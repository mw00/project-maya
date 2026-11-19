// src/hy4/hy4_model.cu - maya-hy4's model (hy4_model.hpp): loading, placement over the GPUs, and the forward.
//
// A layer, for a window of T tokens (the GGUF's hy_v4_flash graph, AngelSlim's hy-v4-flash.cpp):
//   [Engram, layers 1 11 21 31]  x[s] += sigmoid-gated rms(value) gamma (gate[s] + hc_scale[s]) from 16 n-gram rows
//   cur, post = hc_pre(x);  xa = rms(cur) w
//   qr = rms(xa Wq_a) w;  q = qr Wq_b [H][nope | rope];  kv = xa Wkv_a [kv_lora | rope] -> the cache (normed | rotated)
//   q_nope through k_b into the latent space, q_pe rotated;  the keys: the last n_swa positions (sliding-window layers)
//   or the indexer's picks (top_k / kpool - 1 pooled groups by score, then the token's own group up to itself)
//   o = softmax(q.k / sqrt(k_mla), sink) . latent;  o through v_b;  * sigmoid(xa Wgate);  Wo;  rms w
//   x = hc_post(x, that, post)
//   cur, post = hc_pre(x);  xf = rms(cur) w;  layer 0: the dense SwiGLU;  else the router (sigmoid, + bias, top 16,
//   renormalised, * scale), the routed experts in the latent space (xf Wdown_r, experts, Wup_r), + the shared expert
//   x = hc_post(x, rms(that) w, post)
// and logits = rms(hc_head(x)) w . Wout.
#include "hy4_model.hpp"

#include "hy4_kernels.cuh"
#include "hy4_lane.hpp"
#include "hy4_tp.hpp"
#include "hy4_pack.hpp"
#include "hy4_plan.hpp"
#include "../ds4/ds4_iq.cuh"

#include <cublas_v2.h>
#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

namespace hy4 {

namespace iq = strata::ds4::gguf::iq;

extern "C" void quantize_row_q4_K_ref(const float* x, void* y, int64_t k);   // (ggml-quants: ggml's own Q4_K rounding)
// (ggml-quants' row decoders: a token's embedding row in the K types - the host reads one row a token)
extern "C" void dequantize_row_q4_K(const void* x, float* y, int64_t k);
extern "C" void dequantize_row_q5_K(const void* x, float* y, int64_t k);
extern "C" void dequantize_row_q6_K(const void* x, float* y, int64_t k);

namespace {

constexpr int kF32 = 0, kF16 = 1, kBF16 = 30, kF8 = 43;
constexpr int W = k::kMaxWin;

int type_id(const std::string& s) {
    static const std::pair<const char*, int> names[] = {
        {"F32", 0},      {"F16", 1},     {"Q8_0", 8},     {"Q2_K", 10},  {"Q3_K", 11},    {"Q4_K", 12},
        {"Q5_K", 13},    {"Q6_K", 14},   {"IQ2_XXS", 16}, {"IQ2_XS", 17}, {"IQ3_XXS", 18}, {"IQ1_S", 19},
        {"IQ4_NL", 20},  {"IQ3_S", 21},  {"IQ2_S", 22},   {"IQ4_XS", 23}, {"IQ1_M", 29},   {"BF16", 30},
        {"F8_E4M3", 43}};
    for (const auto& [n, i] : names)
        if (s == n) return i;
    return -1;
}
size_t row_bytes(int type, int64_t k) {
    if (type == kF32) return (size_t) k * 4;
    if (type == kF16 || type == kBF16) return (size_t) k * 2;
    if (type == kF8) return (size_t) k;
    if (!iq::mv_supported(type) || k % iq::block_values(type)) return 0;
    return iq::row_bytes(type, k);
}
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while (!(mm & 0x400));
            f = s | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7f800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}
// A token's embedding row (k values) from the file's bytes: Q8_0 as it always was read (d * q, block by block), the K
// types by ggml's own decoders, F32 / BF16 / F16 as they are.  false: a type the host does not decode
bool tok_type_ok(int type) { return type == 8 || type == 12 || type == 13 || type == 14 || type == 0 || type == 30 || type == 1; }
void tok_row(int type, const uint8_t* row, int k, float* dst) {
    if (type == 8) {
        for (int b = 0; b < k / 32; ++b) {
            uint16_t dh;
            std::memcpy(&dh, row + (size_t) b * 34, 2);
            const float d = half_to_float(dh);
            const int8_t* qv = (const int8_t*) (row + (size_t) b * 34 + 2);
            for (int i = 0; i < 32; ++i) dst[b * 32 + i] = d * (float) qv[i];
        }
    } else if (type == 12) {
        dequantize_row_q4_K(row, dst, k);
    } else if (type == 13) {
        dequantize_row_q5_K(row, dst, k);
    } else if (type == 14) {
        dequantize_row_q6_K(row, dst, k);
    } else if (type == 0) {
        std::memcpy(dst, row, (size_t) k * 4);
    } else if (type == 30) {
        for (int i = 0; i < k; ++i) {
            uint16_t h;
            std::memcpy(&h, row + (size_t) i * 2, 2);
            const uint32_t u = (uint32_t) h << 16;
            std::memcpy(&dst[i], &u, 4);
        }
    } else {
        for (int i = 0; i < k; ++i) {
            uint16_t h;
            std::memcpy(&h, row + (size_t) i * 2, 2);
            dst[i] = half_to_float(h);
        }
    }
}

// BF16 values as F32 (exact: a BF16 is an F32's upper half)
void bf16_to_f32(const uint16_t* src, size_t n, float* dst) {
    for (size_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, (const uint8_t*) src + i * 2, 2);
        const uint32_t u = (uint32_t) h << 16;
        std::memcpy(dst + i, &u, 4);
    }
}
// ggml's quantize_row_q8_0_ref over rows of k BF16 values (k % 32 == 0), on `threads` threads: the MTP block's BF16
// matrices and experts as the GEMVs and the experts' kernels read them (Q8_0: 34 bytes a 32 values)
void bf16_to_q8_0(const uint16_t* src, size_t rows, int k, uint8_t* dst, int threads = 16) {
    auto work = [&](size_t r0, size_t r1) {
        float v[32];
        for (size_t r = r0; r < r1; ++r)
            for (int b = 0; b < k / 32; ++b) {
                const uint16_t* x = src + r * (size_t) k + (size_t) b * 32;
                float am = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const uint32_t u = (uint32_t) x[i] << 16;
                    std::memcpy(&v[i], &u, 4);
                    am = std::max(am, std::fabs(v[i]));
                }
                const float d = am / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
                uint8_t* o = dst + (r * (size_t) (k / 32) + (size_t) b) * 34;
                // F32 -> F16 (round to nearest even), as ggml's GGML_FP32_TO_FP16
                uint32_t f;
                std::memcpy(&f, &d, 4);
                const uint32_t sign = (f >> 16) & 0x8000u;
                int32_t e = (int32_t) ((f >> 23) & 0xff) - 127 + 15;
                uint32_t m = f & 0x7fffffu;
                uint16_t h;
                if (e <= 0) {   // (a subnormal F16, or zero)
                    if (e < -10) {
                        h = (uint16_t) sign;
                    } else {
                        m |= 0x800000u;
                        const int sh = 14 - e;
                        uint32_t hm = m >> sh;
                        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
                        if (rem > half || (rem == half && (hm & 1))) ++hm;
                        h = (uint16_t) (sign | hm);
                    }
                } else if (e >= 31) {
                    h = (uint16_t) (sign | 0x7c00u);
                } else {
                    uint32_t hm = m >> 13;
                    const uint32_t rem = m & 0x1fffu;
                    uint32_t he = (uint32_t) e;
                    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) {
                        if (++hm == 0x400u) {
                            hm = 0;
                            ++he;
                        }
                    }
                    h = (uint16_t) (sign | (he << 10) | hm);
                }
                std::memcpy(o, &h, 2);
                for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * id);
            }
    };
    std::vector<std::thread> th;
    const size_t per = (rows + (size_t) threads - 1) / (size_t) threads;
    for (int t = 0; t < threads; ++t) {
        const size_t r0 = (size_t) t * per, r1 = std::min(rows, r0 + per);
        if (r0 < r1) th.emplace_back(work, r0, r1);
    }
    for (auto& x : th) x.join();
}
// (load_layer on the MTP block's thread: its BF16 matrices rounded to Q8_0 by mat())
thread_local bool t_round_bf16 = false;

}  // namespace

#define CK(x)                                                                              \
    do {                                                                                   \
        const cudaError_t e_ = (x);                                                        \
        if (e_ != cudaSuccess) {                                                           \
            err = std::string(#x) + ": " + cudaGetErrorString(e_);                         \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

/// A matrix on the device: rows of k inputs (ld bytes apart), n outputs; batch matrices of n rows each (3D weights)
struct Mat {
    const uint8_t* w = nullptr;
    int type = -1, k = 0, n = 0, batch = 1;
    size_t ld = 0;
    explicit operator bool() const { return w != nullptr; }
};

struct Layer {
    int il = 0, gpu = -1, H = 0;
    bool swa = false, ix = false, dense = false;
    int eg = -1;                 // its Engram module (-1: none)
    float base = 1e4f;
    float *attn_norm = nullptr, *ffn_norm = nullptr, *post_attn = nullptr, *post_ffn = nullptr;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr, *sinks = nullptr;
    Mat q_a, q_b, kv_a, k_b, v_b, wo, gate;
    float *hca_fn = nullptr, *hca_base = nullptr, *hca_scale = nullptr;
    float *hcf_fn = nullptr, *hcf_base = nullptr, *hcf_scale = nullptr;
    Mat ix_q_b, ix_k, ix_val, ix_gate, ix_proj;
    float *ix_k_norm = nullptr, *ix_ape = nullptr, *ix_pnorm = nullptr;
    Mat eg_kv;
    float *eg_kv_b = nullptr, *eg_gate = nullptr, *eg_lookup = nullptr, *eg_gamma = nullptr, *eg_hcs = nullptr;
    Mat ffn_gate, ffn_up, ffn_down;
    Mat router, rdown, rup, sh_gate, sh_up, sh_down;
    float* router_b = nullptr;
    uint8_t* experts = nullptr;  // the VRAM experts, a blob a slot
    size_t blob = 0, tr[3] = {0, 0, 0};
    int tg = 0, td = 0;
    int n_res = 0;               // (the plan) experts in VRAM; the others on the CPU lane
    std::vector<int> slot_of;    // each expert's slot (-1: the CPU lane's), and on the device
    int* slot_d = nullptr;
    CpuExperts cex;              // the CPU lane's view of the layer's experts (the file's tensors)
    int ffe = 0;                 // an expert's width here (a tensor-parallel member's slice: its share of n_ff_exp)
    int tpm = -1;                // the tensor-parallel member whose slice this is (-1: the whole layer)
    bool partial() const { return tpm < 0 && !dense && !mtp && n_res < (int) slot_of.size(); }
    // the MTP block (il == n_layer): its input norms and projection, and its output's norm (no hyper-connections)
    bool mtp = false;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    Mat eh_proj;
    uint16_t* kv = nullptr;      // the latent cache: rows of kv_lora + rope F16
    int ring = 0;                // its rows (0: a row a position)
    float* pooled = nullptr;     // the indexer's pooled keys [max_ctx / kpool][ix_dim]
    float* ivg = nullptr;        // ... and each position's value | gate [ring][2 ix_dim]
    size_t bytes = 0;            // (the plan: its VRAM)
};

struct Gpu {
    int dev = 0;
    int sH = 64, sFF = 0;       // (its scratch's most heads and widest FFN: a tensor-parallel member's share; 0: n_ff)
    double bw = 0;              // its memory's measured GB/s (a copy within it)
    cudaStream_t s = nullptr;
    cudaEvent_t ev = nullptr;   // (the CPU lane's: its inputs copied)
    std::vector<void*> allocs;
    size_t used = 0;
    // a window's activations
    float *x = nullptr, *cur = nullptr, *post = nullptr, *xa = nullptr, *qa = nullptr, *qr = nullptr, *q = nullptr;
    float *qcat = nullptr, *kvr = nullptr, *o = nullptr, *ob = nullptr, *g = nullptr, *y = nullptr, *xf = nullptr;
    float *gu = nullptr, *hd = nullptr, *rl = nullptr, *rw = nullptr, *lat = nullptr, *latout = nullptr;
    int* ids = nullptr;
    int8_t* xq = nullptr;
    float* xd = nullptr;
    // the routed experts
    int8_t *x2q = nullptr, *ehq = nullptr;
    float *x2d = nullptr, *eh = nullptr, *ehd = nullptr, *part = nullptr;
    unsigned long long *grp_ptr = nullptr, *plan = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
    // attention
    float* ws = nullptr;
    int *pos = nullptr, *lo = nullptr, *nk = nullptr, *keys = nullptr, *knk = nullptr;
    // the indexer
    float *iq = nullptr, *ik = nullptr, *ivt = nullptr, *wt = nullptr, *score = nullptr;
    // Engram
    uint8_t* eg_rows = nullptr;
    float *eg_valid = nullptr, *eg_r = nullptr, *eg_kvo = nullptr;
    float* eg_sc = nullptr;   // (the rows' scales [T][cols]: only with a scales file)
    // the head
    float *hcur = nullptr, *logits = nullptr;
    float* route = nullptr;     // (tensor parallel: member 0's and the helpers' routing rows [lat | ids | weights])
    float* route2 = nullptr;    // (... a staggered chunk's second half's)
    int half = 0;               // (a staggered chunk: this view's half, 0 / 1)
    int row0 = 0;               // (... its rows' start: where its partials and its sums sit in the host's buffers)
    uint16_t *h16 = nullptr, *p16 = nullptr;   // (a prompt chunk's F16 partial out, its F16 sum in: [Tc][E])
    cudaStream_t s3 = nullptr;                 // (a staggered half's sums read back here, beside the other's work)
    cudaEvent_t evh0 = nullptr, evh1 = nullptr, evh = nullptr;   // (... each half's read-back done; a view's own)
    float *sc16 = nullptr, *psc16 = nullptr;   // (... their rows' factors)
    bool defer = false;         // (... its sums published only - read back by its own later step)
    cudaStream_t s2 = nullptr;  // (... a member's side stream: the routing copied out beside its experts)
    cudaEvent_t ev2 = nullptr;
    unsigned* cnt2 = nullptr;   // (... that copy's publish tally)
    // the prompt path (a chunk of more than kMaxWin tokens): F16 activations and weight bands for cuBLAS, the experts'
    // sorted rows
    cublasHandle_t blas = nullptr;
    uint16_t *xh16 = nullptr, *wband = nullptr, *sh16 = nullptr;
    size_t wband_n = 0;   // (values)
    int *bounds = nullptr, *stok = nullptr, *srow = nullptr;
    float* sy = nullptr;
    float* cpo = nullptr;   // (the CPU lane's sums, [Tc][n_lat])
    uint8_t* stage = nullptr;   // (a prompt's streamed experts: kStage slots of the largest blob)
    double h2d_gbs = 10.0;      // (... the copies to it, GB/s: measured at the load)
    int* pmap = nullptr;        // (... each expert's slot in them, -1: not this pass's)
    // ... and its attention as GEMMs: the scores, the probabilities (F16), a ring's keys in order, the picks' mask
    float *sS = nullptr, *dmask = nullptr;
    uint16_t *sP = nullptr, *kbuf = nullptr;

    template <typename T>
    bool alloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        const size_t b = std::max<size_t>(n * sizeof(T), 256);
        const cudaError_t e = cudaMalloc(&v, b);
        if (e != cudaSuccess) {
            err = "GPU " + std::to_string(dev) + ": cudaMalloc " + std::to_string(b >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
        allocs.push_back(v);
        used += b;
        p = (T*) v;
        return true;
    }
};

struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    std::vector<Layer> L;
    std::vector<Gpu> G;
    int head_gpu = 0;
    Mat out;
    float *out_norm = nullptr, *hh_fn = nullptr, *hh_base = nullptr, *hh_scale = nullptr;
    // the trunk's final state of each position (ring rows of E: the MTP block's input a position later), and the MTP
    // block (Options::mtp) with its buffers on the head GPU
    float* hring = nullptr;
    Layer M;
    Mat out_draft;   // (the output head rounded to Q8_0 for the drafts' top token: half the bytes a draft step reads)
    float *mtp_in = nullptr, *mtp_cur = nullptr, *mtp_hin = nullptr, *mtp_out = nullptr, *mtp_top = nullptr;
    float* h_emb = nullptr;              // pinned [Tc][E]: the MTP block's embeddings
    // tensor parallel: a prompt chunk's MTP pass on its own scratch and stream of the head's GPU (hm), queued by its own
    // thread, beside the next chunk's layers: ev_hin its inputs ready, ev_mtp the pass done, ev_emb its embeddings
    // copied (HY4_MTP_SYNC=1: in line, as before)
    Gpu hm;
    bool mtp_async = false;
    bool mtp_own = false;                // (the plan's: that scratch is counted on the head's GPU - tensor_split)
    std::vector<double> free0;           // each GPU's free VRAM as the plan first read it (a plan run again: the same)
    std::atomic<bool> emb_pending{false};   // (a queued pass's copy of h_emb maybe still in flight)
    cudaEvent_t ev_hin = nullptr, ev_mtp = nullptr, ev_emb = nullptr;
    std::unique_ptr<Crew> mtp_crew;
    std::string mtp_err;
    std::vector<int> mtp_ids;
    bool mtp_pending = false;   // (a pass queued since the last settle - a wait for nothing costs a draft round ~1 ms)
    void mtp_settle() {   // (the queued pass enqueued, and the head's stream after it)
        if (!mtp_async || !mtp_pending) return;
        mtp_pending = false;
        mtp_crew->wait();
        cudaSetDevice(G[(size_t) head_gpu].dev);
        cudaStreamWaitEvent(G[(size_t) head_gpu].s, ev_mtp, 0);
    }
    // the CPU lane (the plan put some experts there): its threads, and the pinned rows the FFNs hand it
    std::unique_ptr<Lane> lane;
    std::thread placer;               // (place_experts in the background: the CPU lane's experts over the NUMA nodes)
    std::atomic<bool> placer_stop{false};
    int* h_ids = nullptr;
    float *h_w = nullptr, *h_lat = nullptr, *h_cpu = nullptr;
    uint8_t* h_stage = nullptr;   // (pinned: a pass of streamed experts, kStage blobs)
    const double ev_poll_s = (double) lane_waits().ev_poll_us * 1e-6;   // (0: asleep on the event at once)
    double cpu_us_et = 15.0;      // the CPU lane's microseconds an expert-token on a prompt chunk (measured, EMA)
    // the expert profile: uint32 [n_layer][n_expert] routings - read (Options::profile: the most routed stay in VRAM),
    // counted where the routes reach the host (the CPU lane's layers, every layer with Options::profile_out) and
    // written back (with the read ones) by save_profile
    std::vector<uint32_t> prof_in, prof_cnt;
    double copy_gbs = 15.0;       // ... the copies into the bounce, GB/s (measured, EMA)
    int* h_pmap = nullptr;
    std::vector<CpuEntry> lane_entries;
    // tensor parallel (Options::tp > 1, hy4_tp.hpp): the members (indexes into G), each one's layer slices; the pinned
    // mapped buffers - each member's partial [Tc][E] and its counter (kLine words apart), the host's sum and counter,
    // member 0's final state for a head GPU of its own and its counter - and each member's publish tally (device)
    static constexpr int kLine = 16;
    int tp = 0;
    std::vector<int> mem;
    std::vector<std::vector<Layer>> TL;
    float *tp_part = nullptr, *tp_sum = nullptr, *tp_xo = nullptr;
    unsigned *tp_fout = nullptr, *tp_fin = nullptr, *tp_fx = nullptr;
    // a prompt chunk's sums by blocks of rows (kBlk at most): each part's block b's counter (tp_bout, [part][kBlk]),
    // the host's (tp_bin, [kBlk]) - a block summed as soon as every part's has come, read back as soon as it is summed
    static constexpr int kBlk = 16;
    static constexpr int kSumThreads = 16;   // (the host's sums: threads at most)
    unsigned *tp_bout = nullptr, *tp_bin = nullptr;
    // ... in F16 (half the bytes over PCIe and through the host's memory): each part's rows [part][Tc][E] and their
    // factors [part][Tc], the sum's and its factors (HY4_TP_F32: the F32 sums)
    uint16_t *tp_part16 = nullptr, *tp_sum16 = nullptr;
    float *tp_pscale = nullptr, *tp_sscale = nullptr;
    const bool f16_sums = std::getenv("HY4_TP_F32") == nullptr;
    int blk_rows() const { return std::max(64, (Tc + kBlk - 1) / kBlk); }
    const bool blocked = std::getenv("HY4_TP_ONE_BLOCK") == nullptr;   // (HY4_TP_ONE_BLOCK: a chunk's sum in one piece)
    std::vector<unsigned*> tp_cnt;
    unsigned tp_seq = 0, tp_xseq = 0;
    bool tp_broken = false;
    std::atomic<bool> tp_abort{false};
    std::unique_ptr<Crew> crew;
    std::unique_ptr<Summer> summer;
    bool head_member() const { return tp > 1 && head_gpu == mem[0]; }
    // the expert helpers (the GPUs outside the members - the main one among them): w units of every MoE layer's expert
    // slices each (its memory's speed against a member's), their layers [unit][layer]; the units over all (members 1
    // each): U.  Member 0 hands them each MoE layer's routing (tp_route, its counter tp_froute): their latent sums join
    // the FFN's (only its latent part)
    struct Helper {
        int gi = 0, w = 0, unit0 = 0;
        std::vector<std::vector<Layer>> L;
        unsigned* cnt = nullptr;
    };
    std::vector<Helper> hel;
    int U = 0;
    float* tp_route = nullptr;
    unsigned* tp_froute = nullptr;
    float* tp_route2 = nullptr;      // (a staggered chunk's second half's routing, and its counter)
    unsigned* tp_froute2 = nullptr;
    const uint8_t* tok_embd = nullptr;   // (host: the file's rows, decoded a token at a time - tok_row)
    int tok_type = -1;
    size_t tok_ld = 0;
    const uint8_t* eg_table = nullptr;   // (host: the Engram table's F8 rows)
    size_t eg_bytes = 0;
    std::thread eg_populator;            // (its pages mapped in the background)
    int ring = 0;
    int Tc = W;                          // a forward's chunk at most (the prompt path's; W: the token path's alone)
    int64_t end = 0;
    // the MTP block's cache rows made from the trunk's states: positions [0, mtp_done) (a token window leaves its rows to
    // the next draft's first step, which makes them together with its own: one MTP forward a round, not two)
    int64_t mtp_done = 0;
    std::vector<int> tokens;             // the sequence so far (Engram's n-gram history)
    std::unique_ptr<Vision> vis;         // the vision tower (Options::vision)
    VisionGeo vgeo;
    double vis_room = 0;                 // its room for the largest picture, kept free on the main GPU
    int vis_tok = -1, vis_tok_map = -1;  // (the image token's n-gram map entry before the tower took it)
    std::vector<Model::Image> imgs;      // the request's pictures (their positions' rows)
    // the embeddings of positions p0 .. p0 + T - 1 that are a picture's image tokens: its rows (dst [T][E])
    void image_rows(int64_t p0, int T, float* dst) const {
        const int E = g.n_embd;
        for (const Model::Image& im : imgs) {
            const int64_t n = (int64_t) (im.rows.size() / (size_t) E);
            const int64_t a = std::max(p0, im.start), b = std::min(p0 + T, im.start + n);
            if (a < b)
                std::memcpy(dst + (size_t) (a - p0) * E, im.rows.data() + (size_t) (a - im.start) * E,
                            sizeof(float) * (size_t) (b - a) * E);
        }
    }
    Stats st;
    std::vector<std::string> info;
    // host staging
    float* h_x = nullptr;                // pinned [Tc][hc][E]
    uint8_t* h_eg = nullptr;             // pinned [modules][Tc][cols][dim]
    float* h_valid = nullptr;            // pinned [Tc]
    // The Engram norms' eps.  The reference graph's norms have none set and take the machine epsilon of the type they run
    // in: FP32's 1.1920929e-07 (AngelSlim's runtime, the default here), BF16's 0.0078125 when the release runs in BF16
    // (hy4.txt's engram_eps, or HY4_ENGRAM_EPS)
    float eg_eps = 1.1920929e-07f;
    // The table's row scales (optional): a table of per-row scaled codes, e4m3(x / s) with s = max|row| / 448, has its
    // BF16 s a row in a file beside it (hy4.txt's engram_scales); the rows' values are then the release's own.  Read
    // from the disk a looked-up row at a time, as the table
    const uint8_t* eg_scale = nullptr;
    float* h_egs = nullptr;              // pinned [modules][Tc][cols]
    std::FILE* eg_dump = nullptr;        // (HY4_EG_DUMP=<file>, tests: every looked-up row - see engram_row)
    std::mutex eg_dump_mu;
    int* h_pos = nullptr;                // pinned [3][Tc]: pos, lo, nk

    ~Impl() {
        if (mtp_crew) mtp_crew->wait();
        mtp_crew.reset();
        if (hm.s) {
            cudaSetDevice(hm.dev);
            cudaStreamSynchronize(hm.s);
            for (void* p : hm.allocs) cudaFree(p);
            if (hm.blas) cublasDestroy(hm.blas);
            cudaStreamDestroy(hm.s);
            for (cudaEvent_t e : {ev_hin, ev_mtp, ev_emb})
                if (e) cudaEventDestroy(e);
        }
        placer_stop.store(true);
        if (placer.joinable()) placer.join();
        if (eg_populator.joinable()) eg_populator.join();
        for (auto& gp : G) {
            cudaSetDevice(gp.dev);
            for (void* p : gp.allocs) cudaFree(p);
            if (gp.blas) cublasDestroy(gp.blas);
            if (gp.ev) cudaEventDestroy(gp.ev);
            if (gp.ev2) cudaEventDestroy(gp.ev2);
            if (gp.s2) cudaStreamDestroy(gp.s2);
            if (gp.evh0) cudaEventDestroy(gp.evh0);
            if (gp.evh1) cudaEventDestroy(gp.evh1);
            if (gp.s3) cudaStreamDestroy(gp.s3);
            if (gp.s) cudaStreamDestroy(gp.s);
        }
        if (h_x) cudaFreeHost(h_x);
        if (h_eg) cudaFreeHost(h_eg);
        if (h_valid) cudaFreeHost(h_valid);
        if (h_egs) cudaFreeHost(h_egs);
        if (eg_dump) std::fclose(eg_dump);
        if (h_pos) cudaFreeHost(h_pos);
        if (h_emb) cudaFreeHost(h_emb);
        if (h_ids) cudaFreeHost(h_ids);
        if (h_w) cudaFreeHost(h_w);
        if (h_lat) cudaFreeHost(h_lat);
        if (h_cpu) cudaFreeHost(h_cpu);
        if (h_stage) cudaFreeHost(h_stage);
        if (h_pmap) cudaFreeHost(h_pmap);
        crew.reset();
        summer.reset();
        for (void* p : {(void*) tp_part, (void*) tp_sum, (void*) tp_xo, (void*) tp_fout, (void*) tp_fin, (void*) tp_fx,
                        (void*) tp_route, (void*) tp_froute, (void*) tp_bout, (void*) tp_bin, (void*) tp_route2,
                        (void*) tp_froute2, (void*) tp_part16, (void*) tp_sum16, (void*) tp_pscale, (void*) tp_sscale})
            if (p) cudaFreeHost(p);
    }

    // ------------------------------------------------------------ loading
    const Tensor* need(const std::string& n, std::string& err) {
        const Tensor* t = pk.get(n);
        if (!t) err = "the pack has no " + n;
        return t;
    }
    bool upload(Gpu& gp, const Tensor& t, void*& dst, std::string& err) {
        if (!gp.alloc((uint8_t*&) dst, t.bytes, err)) return false;
        CK(cudaMemcpy(dst, pk.data(t), t.bytes, cudaMemcpyHostToDevice));
        return true;
    }
    bool f32(Gpu& gp, const std::string& n, float*& p, std::string& err, int64_t want = -1) {
        const Tensor* t = need(n, err);
        if (!t) return false;
        if (t->type != "F32") {
            err = n + ": " + t->type + ", F32 expected";
            return false;
        }
        if (want >= 0 && (int64_t) (t->bytes / 4) != want) {
            err = n + ": " + std::to_string(t->bytes / 4) + " values, " + std::to_string(want) + " expected";
            return false;
        }
        void* v = nullptr;
        if (!upload(gp, *t, v, err)) return false;
        p = (float*) v;
        return true;
    }
    bool mat(Gpu& gp, const std::string& n, Mat& m, std::string& err, int k = -1, int nout = -1, int batch = 1) {
        const Tensor* t = need(n, err);
        if (!t) return false;
        m.type = type_id(t->type);
        m.k = (int) t->n(0);
        m.n = (int) t->n(1);
        m.batch = (int) t->n(2);
        m.ld = row_bytes(m.type, m.k);
        if (m.type == kBF16 && t_round_bf16 && m.k % 32 == 0 && m.ld * (size_t) m.n * (size_t) m.batch == t->bytes) {
            const size_t rows = (size_t) m.n * m.batch;
            std::vector<uint8_t> q(rows * (size_t) (m.k / 32) * 34);
            bf16_to_q8_0((const uint16_t*) pk.data(*t), rows, m.k, q.data());
            m.type = 8;
            m.ld = row_bytes(8, m.k);
            uint8_t* d = nullptr;
            if (!gp.alloc(d, q.size(), err)) return false;
            CK(cudaMemcpy(d, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = d;
            if ((k >= 0 && m.k != k) || (nout >= 0 && m.n != nout) || m.batch != batch) {
                err = n + ": not the shape the MTP block's geometry gives";
                return false;
            }
            return true;
        }
        if (m.ld == 0 || (k >= 0 && m.k != k) || (nout >= 0 && m.n != nout) || m.batch != batch ||
            m.ld * (size_t) m.n * (size_t) m.batch != t->bytes) {
            err = n + ": " + t->type + " [" + std::to_string(m.k) + ", " + std::to_string(m.n) + ", " +
                  std::to_string(m.batch) + "], expected [" + std::to_string(k) + ", " + std::to_string(nout) + ", " +
                  std::to_string(batch) + "] in a type maya-hy4 reads";
            return false;
        }
        // a BF16 matrix (not the output head, which has its own BF16 paths): the same values as F32 in VRAM - the token
        // path's GEMV sums them in the same order, and the prompt path's GEMM takes F32 weights (BF16 it does not)
        if (m.type == kBF16 && n != "output.weight") {
            const size_t nv = (size_t) m.k * (size_t) m.n * (size_t) m.batch;
            std::vector<float> f(nv);
            bf16_to_f32((const uint16_t*) pk.data(*t), nv, f.data());
            float* d = nullptr;
            if (!gp.alloc(d, nv, err)) return false;
            CK(cudaMemcpy(d, f.data(), nv * 4, cudaMemcpyHostToDevice));
            m.w = (const uint8_t*) d;
            m.type = kF32;
            m.ld = (size_t) m.k * 4;
            return true;
        }
        void* v = nullptr;
        if (!upload(gp, *t, v, err)) return false;
        m.w = (const uint8_t*) v;
        return true;
    }

    // A tensor-parallel member's part of a matrix: rows [r0, r1) of each batch matrix in [b0, b1), and of those rows the
    // columns [c0, c1) (whole blocks of its type) - the whole matrix when the part is all of it
    bool mat_part(Gpu& gp, const std::string& n, Mat& m, std::string& err, int k, int nout, int batch, int r0, int r1,
                  int c0, int c1, int b0, int b1) {
        if (r0 == 0 && r1 == nout && c0 == 0 && c1 == k && b0 == 0 && b1 == batch) return mat(gp, n, m, err, k, nout, batch);
        const Tensor* t = need(n, err);
        if (!t) return false;
        const int type = type_id(t->type);
        const size_t ld = row_bytes(type, k);
        if (ld == 0 || t->n(0) != k || t->n(1) != nout || t->n(2) != batch || ld * (size_t) nout * batch != t->bytes) {
            err = n + ": " + t->type + " [" + std::to_string(t->n(0)) + ", " + std::to_string(t->n(1)) + ", " +
                  std::to_string(t->n(2)) + "], expected [" + std::to_string(k) + ", " + std::to_string(nout) + ", " +
                  std::to_string(batch) + "] in a type maya-hy4 reads";
            return false;
        }
        const size_t o0 = row_bytes(type, c0), w = row_bytes(type, c1 - c0);
        if ((c0 > 0 && o0 == 0) || w == 0) {
            err = n + ": its columns " + std::to_string(c0) + " .. " + std::to_string(c1) + " are not whole " + t->type +
                  " blocks (another --tp)";
            return false;
        }
        const size_t rows = (size_t) (r1 - r0) * (size_t) (b1 - b0);
        std::vector<uint8_t> h(rows * w);
        const uint8_t* src = pk.data(*t);
        for (int b = b0; b < b1; ++b)
            for (int r = r0; r < r1; ++r)
                std::memcpy(h.data() + ((size_t) (b - b0) * (r1 - r0) + (size_t) (r - r0)) * w,
                            src + ((size_t) b * nout + (size_t) r) * ld + o0, w);
        m.k = c1 - c0;
        m.n = r1 - r0;
        m.batch = b1 - b0;
        if (type == kBF16) {   // (as mat(): a BF16 matrix's part as F32 in VRAM)
            const size_t nv = h.size() / 2;
            std::vector<float> f(nv);
            bf16_to_f32((const uint16_t*) h.data(), nv, f.data());
            float* d = nullptr;
            if (!gp.alloc(d, nv, err)) return false;
            CK(cudaMemcpy(d, f.data(), nv * 4, cudaMemcpyHostToDevice));
            m.w = (const uint8_t*) d;
            m.type = kF32;
            m.ld = (size_t) m.k * 4;
            return true;
        }
        uint8_t* d = nullptr;
        if (!gp.alloc(d, h.size(), err)) return false;
        CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
        m.w = d;
        m.type = type;
        m.ld = w;
        return true;
    }
    // ... of an F32 vector: values [i0, i1) of its `want`
    bool f32_part(Gpu& gp, const std::string& n, float*& p, std::string& err, int64_t want, int64_t i0, int64_t i1) {
        if (i0 == 0 && i1 == want) return f32(gp, n, p, err, want);
        const Tensor* t = need(n, err);
        if (!t) return false;
        if (t->type != "F32" || (int64_t) (t->bytes / 4) != want) {
            err = n + ": not F32 of " + std::to_string(want) + " values";
            return false;
        }
        if (!gp.alloc(p, (size_t) (i1 - i0), err)) return false;
        CK(cudaMemcpy(p, (const float*) pk.data(*t) + i0, sizeof(float) * (size_t) (i1 - i0), cudaMemcpyHostToDevice));
        return true;
    }
    // a member's share [f0, f1) of an FFN width ff whose down matrix is of type td: whole blocks, as even as they go
    static void width_part(int ff, int td, int m, int N, int& f0, int& f1) {
        const int bv = iq::block_values(td), nb = ff / bv;
        f0 = (int) ((int64_t) nb * m / N) * bv;
        f1 = (int) ((int64_t) nb * (m + 1) / N) * bv;
    }

    // the latent cache: 8-bit rows (Options::kv: the latent int8 in blocks of 32 with F16 scales, the rotated part F16)
    // or F16, its bytes a position
    bool kv_q8 = false;
    size_t kv_rb() const { return kv_q8 ? k::kv_q8_row_bytes(g.kv_lora, g.rope) : (size_t) g.kv_row() * 2; }
    // (the indexer's pooled keys and its value | gate ring: F16 with 8-bit rows - the reference's own cache type -,
    //  else F32)
    size_t pool_b() const { return kv_q8 ? 2 : 4; }
    // (the 8-bit rows' kernels take a latent of 512 + 64 values)
    bool kv_q8_rows() const { return g.kv_lora == 512 && g.rope == 64; }
    // every layer's caches at this context, in GB, in 8-bit rows or F16 (the plan's INFO line)
    double cache_gb(bool q8) const {
        const double rb = q8 ? (double) k::kv_q8_row_bytes(g.kv_lora, g.rope) : (double) g.kv_row() * 2, pb = q8 ? 2 : 4;
        double b = 0;
        for (int il = 0; il < g.n_layer; ++il)
            b += g.swa[(size_t) il] ? ring * rb
                                    : o.max_ctx * rb + (double) (o.max_ctx / g.ix_kpool + 1) * g.ix_dim * pb +
                                          (double) ring * 2 * g.ix_dim * pb;
        return b / 1e9;
    }
    // a layer's VRAM without its routed experts (its dense matrices and caches), and an expert's (its blob)
    size_t dense_bytes(int il) const {
        size_t b = 0;
        const std::string pre = "blk." + std::to_string(il) + ".";
        for (const auto& [n, t] : pk.t)
            if (n.compare(0, pre.size(), pre) == 0 && n.find("_exps.") == std::string::npos)
                b += t.bytes + 256 + (il < g.n_layer && t.type == "BF16" ? t.bytes : 0);   // (BF16: F32 in VRAM, mat())
        if (g.swa[(size_t) il]) b += (size_t) ring * kv_rb();
        else
            b += (size_t) o.max_ctx * kv_rb() + (size_t) (o.max_ctx / g.ix_kpool + 1) * g.ix_dim * pool_b() +
                 (size_t) ring * 2 * g.ix_dim * pool_b();
        return b;
    }
    size_t expert_bytes(int il) const {
        size_t b = 0;
        const std::string pre = "blk." + std::to_string(il) + ".";
        for (const auto& [n, t] : pk.t)
            if (n.compare(0, pre.size(), pre) == 0 && n.find("_exps.") != std::string::npos) b += t.bytes;
        return g.n_expert ? b / (size_t) g.n_expert + 768 : 0;
    }
    size_t layer_bytes(int il) const {
        size_t b = 0;
        const std::string pre = "blk." + std::to_string(il) + ".";
        for (const auto& [n, t] : pk.t)
            if (n.compare(0, pre.size(), pre) == 0) {
                b += t.bytes + 256;
                // (a trunk layer's BF16 matrices sit in VRAM as F32 - mat(); the MTP block's are rounded to Q8_0)
                if (il < g.n_layer && t.type == "BF16" && n.find("_exps.") == std::string::npos) b += t.bytes;
            }
        // caches
        if (g.swa[(size_t) il]) b += (size_t) ring * kv_rb();
        else
            b += (size_t) o.max_ctx * kv_rb() + (size_t) (o.max_ctx / g.ix_kpool + 1) * g.ix_dim * pool_b() +
                 (size_t) ring * 2 * g.ix_dim * pool_b();
        return b;
    }
    // the buffers a GPU allocates for a forward (scratch()): the chunk's activations (Tc rows), the token path's int8
    // inputs and experts (W rows), the attention / indexer sub-batches' (kSub rows) and the GEMMs' weight band
    static constexpr int kSub = 64;
    // --chunk auto: the chunks tried above kChunkBase, largest first (hy4_plan.hpp's chunk_pick)
    static constexpr int kChunkBase = 1024;
    static constexpr int kChunks[2] = {4096, 2048};
    static constexpr int kStage = 128;   // (a prompt's streamed experts a pass)
    // The indexed layers' keys a dense (tensor-core) prompt sub-batch spans at most: beyond, the gathering kernel.
    // Both read the indexer's same picks (the dense one masks every other key out), so the limit moves no pick - only
    // which kernel sums a token's attention.  The dense path's time grows with the span, the gathering kernel's does
    // not: on V100s (one, and three in tensor parallel) they cross at 23-24 thousand positions.  kDenseBase: what every
    // start's plan holds (the limit before v1.1.68); kDense: the spans tried above it, largest first, taken where the
    // GPUs have tensor cores and the plan keeps what it comes to at kDenseBase (hy4_plan.hpp's dense_pick) - the
    // buffers cost ~75 MB a GPU per 8192 positions more.  HY4_DENSE_MAX=N: that span alone tried (8192: as before).
    static constexpr int kDenseBase = 8192;
    static constexpr int kDense[2] = {24576, 16384};
    static constexpr int kDenseTop = 65536;   // (HY4_DENSE_MAX's most)
    int dense_max = kDenseBase;
    // the prompt path's F16 rows of the cache a layer's dense sub-batches read (kbuf): a window's and a chunk's, or an
    // indexed layer's up to dense_max (an 8-bit cache dequantized there; F16's read in place but where a ring wraps)
    size_t deq_rows() const { return dense_kbuf_rows((size_t) g.n_swa, (size_t) Tc, (size_t) dense_max, kSub); }
    static constexpr size_t kBand = 32u << 20;   // (values: a dense matrix decoded to F16 this much at a time)
    // (H, ff: the most heads and the widest dense / shared FFN a layer gives the GPU - a tensor-parallel member's share)
    size_t scratch_floats(size_t& f16, size_t& ints, size_t H = 64, size_t ff = 0) const {
        if (ff == 0) ff = (size_t) g.n_ff;
        const size_t E = g.n_embd, hc = g.hc, maxk = (size_t) g.ix_top_k + g.ix_kpool, T = (size_t) Tc;
        const size_t lat = g.n_lat, ffe = g.n_ff_exp, npool = (size_t) (o.max_ctx / g.ix_kpool + 1);
        const size_t egc = (size_t) std::max(1, pk.eg.cols()) * std::max(1, pk.eg.head_dim);
        size_t f = T * hc * E + 6 * T * E + T * hc + 2 * T * g.q_lora + T * H * g.k_mla + T * H * g.kv_row() +
                   T * g.kv_row() + T * H * g.kv_lora + 2 * T * H * g.v_mla + 3 * T * ff + T * g.n_expert + T * 16 +
                   2 * T * lat;
        f += W * lat * 2 / 32 + 16 * W * ffe + 16 * W * ffe / 32 + 16 * W * lat;               // (the token path)
        f += k::attn_ws_floats(kSub, (int) H, g.kv_lora, (int) maxk) + kSub * npool;           // (the sub-batches)
        f += T * (g.ix_heads * g.ix_dim + 3 * g.ix_dim + g.ix_heads) + T + T * egc + T * (hc + 1) * E;
        f += 16 * T * lat + W * E + (size_t) W * g.n_vocab;                                     // (the sets' y, head)
        const size_t qk = std::max({H * (size_t) g.v_mla, ff, H * (size_t) g.kv_lora, (size_t) g.q_lora, egc, E});
        f16 = T * qk + kBand + 16 * T * ffe;
        ints = T * 16 + 3 * T + T * maxk + T + 2 * 16 * T + 1024 + 4 * 16 * W + W * qk / 4 + 2 * W * lat / 4 +
               16 * W * ffe / 4 + T * egc / 4;
        return f;
    }
    size_t scratch_bytes(size_t H = 64, size_t ff = 0) const {
        size_t f16 = 0, ints = 0;
        const size_t f = scratch_floats(f16, ints, H, ff);
        const size_t att = dense_att_values(kSub, H, (size_t) g.n_swa, (size_t) dense_max);
        return f * 4 + f16 * 2 + ints * 4 + att * 6 + (size_t) kSub * dense_max * 4 + deq_rows() * g.kv_row() * 2 +
               (96u << 20);
    }

    int main_gi = 0;
    // The drafts' copy of the output head in Q4_K (ggml's own rounding, the BF16 rows a thread a share): a quarter of the
    // BF16 head's bytes for the drafts' top token - the drafts are checked by the model's own head, so only how often
    // they hold depends on it (HY4_DRAFT_HEAD=q8: Q8_0, half the bytes)
    bool draft_head_q4k(Gpu& hg, std::string& err) {
        const Tensor* t = need("output.weight", err);
        if (!t) return false;
        const int k = g.n_embd;
        const size_t rows = (size_t) t->n(1), rb = iq::row_bytes(iq::kQ4_K, k);
        if (k % 256 || rb == 0) {
            err = "the drafts' head: rows of " + std::to_string(k) + " are not whole Q4_K blocks";
            return false;
        }
        std::vector<uint8_t> q(rows * rb);
        const uint16_t* src = (const uint16_t*) pk.data(*t);
        const int nt = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (int w = 0; w < nt; ++w)
            th.emplace_back([&, w] {
                std::vector<float> x((size_t) k);
                for (size_t r = (size_t) w; r < rows; r += (size_t) nt) {
                    for (int i = 0; i < k; ++i) {
                        const uint32_t b = (uint32_t) src[r * (size_t) k + (size_t) i] << 16;
                        std::memcpy(&x[(size_t) i], &b, 4);
                    }
                    quantize_row_q4_K_ref(x.data(), q.data() + r * rb, k);
                }
            });
        for (auto& x : th) x.join();
        uint8_t* d = nullptr;
        if (!hg.alloc(d, q.size(), err)) return false;
        CK(cudaMemcpy(d, q.data(), q.size(), cudaMemcpyHostToDevice));
        out_draft.w = d;
        out_draft.type = iq::kQ4_K;
        out_draft.k = k;
        out_draft.n = (int) rows;
        out_draft.batch = 1;
        out_draft.ld = rb;
        return true;
    }
    // the current GPU's memory bandwidth: a copy of 256 MB within it, timed (GB/s; 0 and err: it failed)
    double measure_bw(std::string& err) {
        const size_t b = 256u << 20;
        void *a = nullptr, *c = nullptr;
        if (cudaMalloc(&a, b) != cudaSuccess || cudaMalloc(&c, b) != cudaSuccess) {
            cudaGetLastError();
            if (a) cudaFree(a);
            return 1.0;   // (no room for the probe: every GPU alike then)
        }
        double best = 0;
        for (int r = 0; r < 3; ++r) {
            const double t0 = now_s();
            if (cudaMemcpy(c, a, b, cudaMemcpyDeviceToDevice) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                err = "a copy within the GPU failed";
                cudaFree(a);
                cudaFree(c);
                return 0;
            }
            best = std::max(best, 2.0 * b / 1e9 / std::max(1e-9, now_s() - t0));
        }
        cudaFree(a);
        cudaFree(c);
        return best;
    }
    bool plan(std::string& err) {
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        std::vector<int> devs = o.gpus;
        if (devs.empty())
            for (int i = 0; i < ndev; ++i) devs.push_back(i);
        for (int d : devs)
            if (d < 0 || d >= ndev) {
                err = "--gpus " + std::to_string(d) + ": " + std::to_string(ndev) + " GPUs visible";
                return false;
            }
        // (the GPUs' streams, their free VRAM and their memory's speed: read once - --chunk auto plans several chunks
        //  against the same readings)
        if (G.empty()) {
            G.resize(devs.size());
            free0.assign(devs.size(), 0.0);
            for (size_t i = 0; i < devs.size(); ++i) {
                G[i].dev = devs[i];
                CK(cudaSetDevice(devs[i]));
                CK(cudaStreamCreateWithFlags(&G[i].s, cudaStreamNonBlocking));
                CK(cudaEventCreateWithFlags(&G[i].ev, cudaEventDisableTiming | cudaEventBlockingSync));
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
#if !defined(_WIN32)
                // a unified-memory GPU (an APU: Strix Halo / Gorgon Halo): its memory is the system's RAM, so what the
                // model may take is what the RAM has (MemAvailable: the model's own file cache counts as available - the
                // kernel drops it as the weights are copied in, the file is never held twice) less a headroom for the
                // system and the other programs, max(8 GiB, a tenth of the RAM) - HY4_RAM_HEADROOM_GB: another one
                int integrated = 0;
                if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, devs[i]) != cudaSuccess) {
                    cudaGetLastError();
                    integrated = 0;
                }
                if (integrated != 0) {
                    const uint64_t avail = ram_available(), ram = ram_total();
                    double head = std::max(8.0 * 1073741824.0, 0.1 * (double) ram);
                    if (const char* h = std::getenv("HY4_RAM_HEADROOM_GB")) head = std::max(0.0, std::atof(h)) * 1073741824.0;
                    if (avail > 0) {
                        // (the GPU's own share - an APU's BIOS carve-out - is not RAM the system counts: on top)
                        cudaDeviceProp pr{};
                        double own = 0.0;
                        if (cudaGetDeviceProperties(&pr, devs[i]) == cudaSuccess)
                            own = (double) gpu_own_free(pr.pciDomainID, pr.pciBusID, pr.pciDeviceID);
                        else
                            cudaGetLastError();
                        const double room = std::max(0.0, (double) avail - head) + own;
                        char ub[256];
                        std::snprintf(ub, sizeof ub, "INFO GPU%zu shares the system's memory: %.1f GB of it available, %.1f GB "
                                      "kept free for the system, %.1f GB for the model", i, (double) avail / 1e9, head / 1e9,
                                      std::min(room, (double) fr) / 1e9);
                        info.push_back(ub);
                        if (room < (double) fr) fr = (size_t) room;
                    }
                }
#endif
                free0[i] = (double) fr;
                G[i].bw = measure_bw(err);
                if (G[i].bw <= 0) return false;
            }
        }
        std::vector<double> free_b(devs.size());
        for (size_t i = 0; i < devs.size(); ++i) free_b[i] = free0[i] - o.reserve_gb * 1e9 - (double) scratch_bytes();
        // the main GPU: the head's, the drafts', the window's coordination - the fastest memory (then the most VRAM)
        if (o.main_gpu >= (int) devs.size()) {
            err = "--main-gpu " + std::to_string(o.main_gpu) + ": " + std::to_string(devs.size()) + " GPUs given";
            return false;
        }
        main_gi = o.main_gpu;
        if (main_gi < 0) {
            main_gi = 0;
            for (size_t i = 1; i < devs.size(); ++i)
                if (G[i].bw > G[(size_t) main_gi].bw * 1.15 ||
                    (G[i].bw > G[(size_t) main_gi].bw / 1.15 && free_b[i] > free_b[(size_t) main_gi] + 1e9))
                    main_gi = (int) i;
        }
        free_b[(size_t) main_gi] -= vis_room;   // (the vision tower's room for a picture: free between pictures)
        // tensor parallel: --tp N, or auto (-1): over the largest group of GPUs of one kind when it holds everything -
        // the layer split otherwise (an INFO line says why)
        if (o.tp != 0 && o.tp != 1) {
            int N = o.tp;
            if (N < 0) {
                std::map<std::string, int> kinds;
                for (int d : devs) {
                    cudaDeviceProp pi{};
                    CK(cudaGetDeviceProperties(&pi, d));
                    N = std::max(N, ++kinds[std::string(pi.name) + "/" + std::to_string(pi.major) + std::to_string(pi.minor)]);
                }
            }
            if (N >= 2) {
                auto undo = [&] {
                    tp = 0;
                    mtp_own = false;
                    mem.clear();
                    hel.clear();
                    TL.clear();
                    for (Gpu& gp : G) {
                        gp.sH = 64;
                        gp.sFF = 0;
                    }
                };
                // the MTP block's pass over a prompt chunk beside the next chunk: on a scratch of its own on the head's
                // GPU, counted here - where a member's plan does not hold it, the pass runs in line on the head's
                // scratch (said), and tensor parallel stays
                const bool want_own = o.mtp && g.n_mtp > 0 && !std::getenv("HY4_MTP_SYNC");
                std::vector<double> fb;
                auto split = [&](std::string& e) {
                    for (int own = want_own ? 1 : 0; own >= 0; --own) {
                        undo();
                        fb = free_b;
                        e.clear();
                        if (!tensor_split(ndev, devs, fb, e, N, own == 1)) continue;
                        if (want_own && !own) {
                            char mb[240];
                            std::snprintf(mb, sizeof mb, "INFO the MTP block's pass in line with the prompt's chunks: its "
                                          "own scratch (%.2f GB at a chunk of %d tokens) does not fit beside a member's "
                                          "plan on the head's GPU", (double) scratch_bytes() / 1e9, Tc);
                            info.push_back(mb);
                        }
                        return true;
                    }
                    return false;
                };
                std::string terr;
                if (split(terr)) return true;
                // (--kv auto: every member holds the whole cache - in 8-bit rows, 58% of its VRAM, when F16 does not fit)
                if (o.kv < 0 && !kv_q8 && kv_q8_rows()) {
                    kv_q8 = true;
                    std::string t8;
                    if (split(t8)) {
                        info.push_back("INFO the latent cache in 8-bit rows (--kv auto): with F16 rows tensor parallel "
                                       "does not hold this context (" + terr + ")");
                        return true;
                    }
                    kv_q8 = false;
                    terr += "; in 8-bit rows: " + t8;
                }
                if (o.tp > 0) {
                    err = terr;
                    return false;
                }
                undo();
                info.push_back("INFO tensor parallel not taken (" + terr + "): the layers split over the GPUs");
            }
        }
        const Tensor* ot = pk.get("output.weight");
        double head_b = (ot ? (double) ot->bytes : 0) + (double) g.n_vocab * W * 4 + (64u << 20) +
                        (double) ring * g.n_embd * 4;   // (the states' ring)
        if (o.mtp && g.n_mtp > 0) head_b += (double) layer_bytes(g.n_layer) * 0.56 + mtp_rows_bytes(Tc, g.n_embd);
        L.resize((size_t) g.n_layer);
        free_b[(size_t) main_gi] -= head_b;   // (the head and the MTP block on the main GPU)
        // the layers over the GPUs in order, each with `rho` of its routed experts in VRAM (the others on the CPU
        // lane): the largest rho the GPUs hold (at most --vram-experts), the GPUs filled in turn
        auto place = [&](double rho, bool apply) {
            int gi = 0;
            double left = free_b[0];
            for (int il = 0; il < g.n_layer; ++il) {
                const bool dense = il < g.n_dense_lead;
                const int n = dense ? 0 : std::min(g.n_expert, (int) std::floor(rho * g.n_expert));
                const double b = (double) dense_bytes(il) + (double) n * (double) expert_bytes(il);
                const double need = b;
                if (!o.layer_split.empty()) {
                    while (gi + 1 < (int) devs.size() && il >= o.layer_split[(size_t) gi]) {
                        ++gi;
                        left = free_b[(size_t) gi];
                    }
                    if (need > left) return false;
                } else {
                    while (need > left) {
                        if (++gi >= (int) devs.size()) return false;
                        left = free_b[(size_t) gi];
                    }
                }
                if (apply) {
                    L[(size_t) il].gpu = gi;
                    L[(size_t) il].n_res = n;
                }
                left -= b;
            }
            return true;
        };
        if (!o.layer_split.empty() && o.layer_split.size() + 1 != devs.size()) {
            err = "--layer-split: " + std::to_string(devs.size() - 1) + " first layers expected";
            return false;
        }
        const double cap = std::max(0.0, std::min(1.0, o.vram_experts));
        double rho = cap;
        if (cap < 1.0 || !place(1.0, false)) {   // (some experts on the CPU lane: room for the prompts' staging)
            double eb = 0;
            for (int il = 0; il < g.n_layer; ++il) eb = std::max(eb, (double) expert_bytes(il));
            for (double& f : free_b) f -= eb * kStage;
        }
        // what the model's own parts fit with (hy4_plan.hpp's kv_pick): as asked - else, in turn, with the cache in
        // 8-bit rows (--kv auto) and a picture's room kept, with the room of a smaller picture (F16 rows, then 8-bit:
        // as many patches as this VRAM holds, a sixteenth of the tower's largest picture at least - the server makes
        // the pictures that size), with F16 rows and no picture's room (pictures not asked for by name), with 8-bit
        // rows and no room.  Pictures go before the rows' precision
        const double vr = vis_room;
        const long long vis_n = vr > 0 ? (long long) vgeo.max_pixels / ((long long) vgeo.patch * vgeo.patch) : 0;
        auto vis_bytes = [&](long long n) {
            VisionGeo vg = vgeo;
            vg.max_pixels = (int) (n * vgeo.patch * vgeo.patch);
            return (double) Vision::room_for(vg);
        };
        const KvPick pick = kv_pick([&](bool q8, double room) {
            const bool was = kv_q8;
            kv_q8 = q8;
            free_b[(size_t) main_gi] += vr - room;
            const bool fit = place(rho, false) || place(0.0, false);
            free_b[(size_t) main_gi] -= vr - room;
            kv_q8 = was;
            return fit;
        }, kv_q8, o.kv < 0 && kv_q8_rows(), vr > 0, o.vision_auto, vis_n,
           std::max<long long>(vis_n / 16, (long long) vgeo.min_pixels / ((long long) vgeo.patch * vgeo.patch) + 1), vis_bytes);
        if (pick.fits && pick.vision && pick.patches < vis_n) {
            // (a picture's room for fewer patches than the tower's largest: the engine says the size, the server
            //  resizes to it - `INFO vision=hy4 ... vision_max_pixels`)
            vgeo.max_pixels = (int) (pick.patches * vgeo.patch * vgeo.patch);
            vis_room = vis_bytes(pick.patches);
            free_b[(size_t) main_gi] += vr - vis_room;
            char sb[320];
            std::snprintf(sb, sizeof sb, "INFO pictures up to %d pixels on this GPU (the tower's largest: %lld): CUDA%d "
                          "has %.0f MB for a picture beside the model's dense parts and caches at this context, the "
                          "largest takes %.0f MB", vgeo.max_pixels, vis_n * vgeo.patch * vgeo.patch,
                          G[(size_t) main_gi].dev, vis_room / 1048576.0, vr / 1048576.0);
            info.push_back(sb);
        }
        if (pick.q8 && !kv_q8) {
            char qb[360];
            std::snprintf(qb, sizeof qb, "INFO the latent cache in 8-bit rows (--kv auto): with F16 rows a context of %lld "
                          "tokens does not fit in the VRAM beside the model's dense parts%s (the cache: %.2f GB in 8-bit "
                          "rows, %.2f GB in F16)", (long long) o.max_ctx, pick.vision ? " and a picture's room" : "",
                          cache_gb(true), cache_gb(false));
            info.push_back(qb);
            kv_q8 = true;
        }
        if (vr > 0 && !pick.vision) {
            // (a card the model's own parts fill: no room kept for a picture - text only, said, rather than no start;
            //  --vision on asks for the tower by name and gets the error below)
            char vb[360];
            std::snprintf(vb, sizeof vb, "NOTE vision off: CUDA%d has no room for a picture (%.0f MB, %.0f MB for the "
                          "smallest it would take) beside the model's dense parts and caches at this context - text only",
                          G[(size_t) main_gi].dev, vis_room / 1048576.0, vis_bytes(std::max(1LL, vis_n / 16)) / 1048576.0);
            info.push_back(vb);
            free_b[(size_t) main_gi] += vis_room;
            vis_room = 0;
            o.vision = false;
            if (vis_tok >= 0) pk.eg.token_map[(size_t) vis_tok] = vis_tok_map;
        }
        if (!place(rho, false)) {
            double lo = 0.0, hi = rho;
            if (!place(0.0, false)) {
                err = std::string("the model's dense parts and caches do not fit in the GPUs' VRAM even with every routed "
                                  "expert on the CPU") + (o.kv < 0 && kv_q8_rows() ? " and the latent cache in 8-bit rows" : "") +
                      " (--max-context smaller, or more GPUs)";
                return false;
            }
            for (int it = 0; it < 30; ++it) {
                const double mid = 0.5 * (lo + hi);
                (place(mid, false) ? lo : hi) = mid;
            }
            rho = lo;
        }
        place(rho, true);
        head_gpu = main_gi;
        return true;
    }

    // Tensor parallel's plan: the last `tp` GPUs the members (of one kind: their copies of the residual must stay the
    // same bit for bit), the first other GPU the head's (the output head, the MTP block) - member 0's when there is none;
    // each member's VRAM checked against its share of every layer
    bool tensor_split(int ndev_all, const std::vector<int>& devs, std::vector<double>& free_b, std::string& err, int n_tp,
                      bool own = false) {
        (void) ndev_all;
        const int N = std::min<int>(n_tp, (int) devs.size());
        if (N < 2) {
            err = "--tp " + std::to_string(n_tp) + ": tensor parallel takes 2 GPUs or more (" +
                  std::to_string(devs.size()) + " given)";
            return false;
        }
        tp = N;
        // the members: N GPUs of the kind there are most of, the main GPU last among them (so it stays out - the head's
        // and the helpers' - where the others are enough; when it is a member, it is member 0)
        std::map<std::string, std::vector<int>> kinds;
        for (size_t i = 0; i < devs.size(); ++i) {
            cudaDeviceProp pi{};
            CK(cudaGetDeviceProperties(&pi, devs[i]));
            kinds[std::string(pi.name) + "/" + std::to_string(pi.major) + std::to_string(pi.minor)].push_back((int) i);
        }
        std::vector<int> cand;
        for (const auto& [k, v] : kinds)
            if (v.size() > cand.size()) cand = v;
        if ((int) cand.size() < N && !std::getenv("HY4_TP_MIXED")) {
            err = "--tp " + std::to_string(N) + ": " + std::to_string(cand.size()) + " GPUs of one kind (the members' "
                  "copies of the model's state must stay the same bit for bit: GPUs of one kind)";
            return false;
        }
        if ((int) cand.size() < N) {   // (HY4_TP_MIXED: any GPUs)
            cand.clear();
            for (size_t i = 0; i < devs.size(); ++i) cand.push_back((int) i);
        }
        std::stable_partition(cand.begin(), cand.end(), [&](int i) { return i != main_gi; });
        mem.assign(cand.begin(), cand.begin() + N);
        head_gpu = main_gi;
        if (std::find(mem.begin(), mem.end(), main_gi) != mem.end()) {
            std::stable_partition(mem.begin(), mem.end(), [&](int i) { return i == main_gi; });
        }
        cudaDeviceProp p0{};
        CK(cudaGetDeviceProperties(&p0, devs[(size_t) mem[0]]));
        for (int i : mem) {
            cudaDeviceProp pi{};
            CK(cudaGetDeviceProperties(&pi, devs[(size_t) i]));
            if ((pi.major != p0.major || pi.minor != p0.minor || std::strcmp(pi.name, p0.name) != 0) &&
                !std::getenv("HY4_TP_MIXED")) {
                err = std::string("--tp: the members are of different kinds (") + p0.name + ", " + pi.name +
                      ") - their copies of the model's state could drift apart; --gpus with GPUs of one kind as the "
                      "last " + std::to_string(N) + " (HY4_TP_MIXED=1 runs them anyway)";
                return false;
            }
        }
        // the members' scratch for their shares (the most heads a layer gives one, the widest FFN share - whole blocks of
        // 256 at most); member 0 the whole model's when it also runs the head and the MTP block
        int sH = 1, sFF = 1;
        for (int il = 0; il < g.n_layer; ++il) sH = std::max(sH, (g.heads[(size_t) il] + N - 1) / N);
        for (int ff : {g.n_ff, g.n_ff_exp * g.n_shared}) sFF = std::max(sFF, ((ff + N - 1) / N + 255) / 256 * 256);
        const double full_s = (double) scratch_bytes(), tp_s = (double) scratch_bytes((size_t) sH, (size_t) sFF);
        for (int i : mem)
            if (!(i == head_gpu && i == mem[0])) {
                G[(size_t) i].sH = sH;
                G[(size_t) i].sFF = sFF;
                free_b[(size_t) i] += full_s - tp_s;
            }
        // the helpers: every other GPU, its units its memory's speed against the slowest member's (HY4_TP_HELPER=0: none,
        // =k: k units each), fewer where its VRAM (beyond the head's on the main GPU) holds fewer
        const Tensor* ot0 = pk.get("output.weight");
        double head_b0 = (ot0 ? (double) ot0->bytes : 0) + (double) g.n_vocab * W * 4 + (64u << 20) +
                         (double) ring * g.n_embd * 4;
        if (o.mtp && g.n_mtp > 0) head_b0 += (double) layer_bytes(g.n_layer) * 0.56 + mtp_rows_bytes(Tc, g.n_embd);
        if (own) head_b0 += (double) scratch_bytes();   // (the MTP pass's own scratch: a whole chunk's buffers)
        double exp_all = 0;   // (every routed expert's bytes)
        for (const auto& [n, t] : pk.t)
            if (n.find("_exps.") != std::string::npos && n.compare(0, 4, "blk.") == 0 &&
                std::atoi(n.c_str() + 4) < g.n_layer)
                exp_all += (double) t.bytes;
        hel.clear();
        const char* he = std::getenv("HY4_TP_HELPER");
        double bw_mem = 1e30;
        for (int i : mem) bw_mem = std::min(bw_mem, G[(size_t) i].bw);
        for (size_t i = 0; i < devs.size() && !(he && std::atoi(he) == 0); ++i) {
            if (std::find(mem.begin(), mem.end(), (int) i) != mem.end()) continue;
            Helper h;
            h.gi = (int) i;
            h.w = he ? std::atoi(he) : std::max(1, (int) std::lround(G[i].bw / std::max(1.0, bw_mem)));
            hel.push_back(h);
        }
        for (;;) {   // (the units, and each helper's VRAM for its share)
            U = N;
            for (const Helper& h : hel) U += h.w;
            bool shrunk = false;
            for (Helper& h : hel) {
                const double have = free_b[(size_t) h.gi] - (h.gi == head_gpu ? head_b0 : 0.0);
                if (h.w > 0 && exp_all * h.w / U * 1.03 > have) {
                    --h.w;
                    shrunk = true;
                }
            }
            hel.erase(std::remove_if(hel.begin(), hel.end(), [](const Helper& h) { return h.w <= 0; }), hel.end());
            if (!shrunk) break;
        }
        int u0 = N;
        for (Helper& h : hel) {
            h.unit0 = u0;
            u0 += h.w;
        }
        // a member's bytes of layer il: its share of the split matrices and of the experts, the rest whole
        auto member_bytes = [&](int il) {
            static const char* split[] = {"attn_q_b.weight", "attn_k_b.weight", "attn_v_b.weight",
                                          "attn_output.weight", "attn_gate.weight", "attn_sinks.weight",
                                          "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                                          "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
            const std::string pre = "blk." + std::to_string(il) + ".";
            double b = 0;
            for (const auto& [n, t] : pk.t) {
                if (n.compare(0, pre.size(), pre) != 0) continue;
                const std::string rest = n.substr(pre.size());
                const bool ex = rest.find("_exps.") != std::string::npos;
                bool sp = false;
                for (const char* x : split) sp |= rest == x;
                b += ex ? (double) t.bytes / U * 1.02 + 4096 : sp ? (double) t.bytes / N * 1.02 + 4096 : (double) t.bytes + 256;
            }
            if (g.swa[(size_t) il]) b += (double) ring * kv_rb();
            else
                b += (double) o.max_ctx * kv_rb() + (double) (o.max_ctx / g.ix_kpool + 1) * g.ix_dim * pool_b() +
                     (double) ring * 2 * g.ix_dim * pool_b();
            return b;
        };
        double need = 0;
        for (int il = 0; il < g.n_layer; ++il) need += member_bytes(il);
        const Tensor* ot = pk.get("output.weight");
        double head_b = (ot ? (double) ot->bytes : 0) + (double) g.n_vocab * W * 4 + (64u << 20) +
                        (double) ring * g.n_embd * 4;
        if (o.mtp && g.n_mtp > 0) head_b += (double) layer_bytes(g.n_layer) * 0.56 + mtp_rows_bytes(Tc, g.n_embd);
        if (own) head_b += (double) scratch_bytes();
        for (int i : mem) {
            const double have = free_b[(size_t) i] - (i == head_gpu ? head_b : 0.0);
            if (need > have) {
                char buf[320];
                std::snprintf(buf, sizeof buf, "--tp %d: a member needs %.2f GB of VRAM, GPU %d has %.2f GB free beyond "
                              "the reserve (a smaller --max-context, or the layer split: --tp 0)", N, need / 1e9,
                              devs[(size_t) i], have / 1e9);
                err = buf;
                return false;
            }
        }
        {
            double have = 1e30;
            for (int i : mem) have = std::min(have, free_b[(size_t) i] - (i == head_gpu ? head_b : 0.0));
            char buf[200];
            std::snprintf(buf, sizeof buf, "INFO tensor parallel: a member's plan %.2f GB of the %.2f GB free beyond the "
                          "reserve (the fullest member)", need / 1e9, have / 1e9);
            info.push_back(buf);
        }
        if (head_gpu != mem[0] && head_b > free_b[(size_t) head_gpu]) {
            err = "--tp: the head's GPU " + std::to_string(devs[(size_t) head_gpu]) + " has too little VRAM free";
            return false;
        }
        mtp_own = own;
        L.assign((size_t) g.n_layer, Layer());
        for (int il = 0; il < g.n_layer; ++il) {
            L[(size_t) il].il = il;
            L[(size_t) il].dense = il < g.n_dense_lead;
            L[(size_t) il].gpu = mem[0];
        }
        TL.assign((size_t) N, std::vector<Layer>((size_t) g.n_layer));
        for (int j = 0; j < N; ++j)
            for (int il = 0; il < g.n_layer; ++il) {
                TL[(size_t) j][(size_t) il].il = il;
                TL[(size_t) j][(size_t) il].gpu = mem[(size_t) j];
            }
        return true;
    }

    // the members' layers (a thread a member), their scratch, the head GPU's scratch, the pinned buffers and threads
    bool load_tp(std::string& err) {
        std::vector<std::string> errs((size_t) tp);
        std::vector<std::thread> th;
        for (int j = 0; j < tp; ++j)
            th.emplace_back([&, j] {
                for (Layer& l : TL[(size_t) j])
                    if (!load_layer(l, errs[(size_t) j], j, tp, U)) return;
                scratch(G[(size_t) mem[(size_t) j]], errs[(size_t) j]);
            });
        std::vector<std::string> herr(hel.size());
        for (size_t h = 0; h < hel.size(); ++h)
            th.emplace_back([&, h] {
                Helper& hp = hel[h];
                hp.L.assign(1, std::vector<Layer>((size_t) g.n_layer));
                for (int il = g.n_dense_lead; il < g.n_layer; ++il) {
                    Layer& l = hp.L[0][(size_t) il];
                    l.il = il;
                    l.gpu = hp.gi;
                    if (!helper_layer(l, hp.unit0, hp.w, herr[h])) return;
                }
            });
        for (auto& t : th) t.join();
        for (auto& e : herr)
            if (!e.empty()) {
                err = e;
                return false;
            }
        for (auto& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        if (!head_member() && !scratch(G[(size_t) head_gpu], err)) return false;
        for (const Helper& h : hel)
            if (h.gi != head_gpu && !scratch(G[(size_t) h.gi], err)) return false;
        const size_t E = (size_t) g.n_embd, T = (size_t) Tc;
        const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
        const int H = (int) hel.size();
        // the sums' node - where most members' PCIe slots are - and the partials and the sum in its memory (the
        // host's sums read every part: local reads, the members' copies cross the link instead)
        std::map<int, int> nodes;
        for (int i : mem) ++nodes[gpu_numa_node(G[(size_t) i].dev)];
        int node = -1, best = 0;
        for (const auto& [nd, c] : nodes)
            if (c > best) {
                best = c;
                node = nd;
            }
        const bool bound = bind_memory_to(node);
        CK(cudaHostAlloc(&tp_part, sizeof(float) * (size_t) (tp + H) * tp_stride(), fl));
        CK(cudaHostAlloc(&tp_route, sizeof(float) * T * (size_t) (g.n_lat + 32), fl));
        CK(cudaHostAlloc(&tp_froute, sizeof(unsigned) * kLine, fl));
        std::memset(tp_froute, 0, sizeof(unsigned) * kLine);
        CK(cudaHostAlloc(&tp_route2, sizeof(float) * T * (size_t) (g.n_lat + 32), fl));
        CK(cudaHostAlloc(&tp_froute2, sizeof(unsigned) * kLine, fl));
        std::memset(tp_froute2, 0, sizeof(unsigned) * kLine);
        CK(cudaHostAlloc(&tp_sum, sizeof(float) * tp_stride(), fl));
        if (f16_sums) {
            CK(cudaHostAlloc(&tp_part16, sizeof(uint16_t) * (size_t) (tp + H) * T * E, fl));
            CK(cudaHostAlloc(&tp_sum16, sizeof(uint16_t) * T * E, fl));
            CK(cudaHostAlloc(&tp_pscale, sizeof(float) * (size_t) (tp + H) * T, fl));
            CK(cudaHostAlloc(&tp_sscale, sizeof(float) * T, fl));
        }
        if (bound) unbind_memory();
        CK(cudaHostAlloc(&tp_xo, sizeof(float) * T * (size_t) g.hc * E, fl));
        CK(cudaHostAlloc(&tp_fout, sizeof(unsigned) * (size_t) (tp + H) * kLine, fl));
        CK(cudaHostAlloc(&tp_fin, sizeof(unsigned) * kLine, fl));
        CK(cudaHostAlloc(&tp_bout, sizeof(unsigned) * (size_t) (tp + H) * kBlk * kLine, fl));
        CK(cudaHostAlloc(&tp_bin, sizeof(unsigned) * kBlk * kLine, fl));
        std::memset(tp_bout, 0, sizeof(unsigned) * (size_t) (tp + H) * kBlk * kLine);
        std::memset(tp_bin, 0, sizeof(unsigned) * kBlk * kLine);
        CK(cudaHostAlloc(&tp_fx, sizeof(unsigned) * kLine, fl));
        std::memset(tp_fout, 0, sizeof(unsigned) * (size_t) (tp + H) * kLine);
        std::memset(tp_fin, 0, sizeof(unsigned) * kLine);
        std::memset(tp_fx, 0, sizeof(unsigned) * kLine);
        tp_cnt.assign((size_t) tp, nullptr);
        for (int j = 0; j < tp; ++j) {
            Gpu& gp = G[(size_t) mem[(size_t) j]];
            CK(cudaSetDevice(gp.dev));
            if (!gp.alloc(tp_cnt[(size_t) j], 16, err)) return false;
            CK(cudaMemset(tp_cnt[(size_t) j], 0, 64));
            if (H > 0) {   // (every member: the hand-off goes round them)
                if (!gp.alloc(gp.route, T * (size_t) (g.n_lat + 32), err) || !gp.alloc(gp.cnt2, 16, err) ||
                    !gp.alloc(gp.route2, T * (size_t) (g.n_lat + 32), err))
                    return false;
                CK(cudaMemset(gp.cnt2, 0, 64));
            }
            if (f16_sums && (!gp.alloc(gp.h16, T * E, err) || !gp.alloc(gp.p16, T * E, err) ||
                             !gp.alloc(gp.sc16, T, err) || !gp.alloc(gp.psc16, T, err)))
                return false;
            // (the side stream: the routing's copy out, and a prompt chunk's partials by blocks; the third: a staggered
            //  half's sums read back)
            CK(cudaStreamCreateWithFlags(&gp.s2, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&gp.ev2, cudaEventDisableTiming));
            CK(cudaStreamCreateWithFlags(&gp.s3, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&gp.evh0, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&gp.evh1, cudaEventDisableTiming));
        }
        for (Helper& h : hel) {
            Gpu& gp = G[(size_t) h.gi];
            CK(cudaSetDevice(gp.dev));
            if (!gp.alloc(h.cnt, 16, err) || !gp.alloc(gp.route, T * (size_t) (g.n_lat + 32), err) ||
                !gp.alloc(gp.route2, T * (size_t) (g.n_lat + 32), err))
                return false;
            CK(cudaMemset(h.cnt, 0, 64));
            if (!gp.s2) {
                CK(cudaStreamCreateWithFlags(&gp.s2, cudaStreamNonBlocking));
                CK(cudaEventCreateWithFlags(&gp.ev2, cudaEventDisableTiming));
            }
            if (f16_sums && !gp.h16 && (!gp.alloc(gp.h16, T * E, err) || !gp.alloc(gp.p16, T * E, err) ||
                                        !gp.alloc(gp.sc16, T, err) || !gp.alloc(gp.psc16, T, err)))
                return false;
        }
        // the sums' threads on that node (a prompt chunk's sums read hundreds of MB: as many as its memory takes)
        std::vector<int> cpus = node_cpus(node);
        const int hw = (int) std::max(1u, std::thread::hardware_concurrency());
        const int helpers = std::max(0, std::min(kSumThreads - 1, (cpus.empty() ? hw : (int) cpus.size()) / 2 - 1));
        {
            const int pn = Lane::node_of_page(tp_part);
            char nb[200];
            std::snprintf(nb, sizeof nb, "INFO tensor parallel: the partials' pinned memory on NUMA node %d (%s)", pn,
                          bound ? "bound" : "not bound");
            info.push_back(nb);
        }
        summer = std::make_unique<Summer>(helpers, cpus);
        crew = std::make_unique<Crew>(tp + H);
        pin_cpus = cpus;
        std::string ms;
        for (int i : mem) ms += (ms.empty() ? "" : ",") + std::to_string(G[(size_t) i].dev);
        char buf[320];
        std::snprintf(buf, sizeof buf, "INFO tensor parallel: every layer over GPUs %s (each its share of the heads and "
                      "the FFNs), the head and the drafts on GPU %d; the sums on %d host threads%s", ms.c_str(),
                      G[(size_t) head_gpu].dev, helpers + 1,
                      node >= 0 ? (" (NUMA node " + std::to_string(node) + ")").c_str() : "");
        info.push_back(buf);
        for (const Helper& h : hel) {
            std::snprintf(buf, sizeof buf, "INFO tensor parallel: GPU %d computes %d of every %d slices of the routed "
                          "experts (its memory %.0f GB/s, a member's %.0f)", G[(size_t) h.gi].dev, h.w, U,
                          G[(size_t) h.gi].bw, G[(size_t) mem[0]].bw);
            info.push_back(buf);
        }
        return true;
    }
    std::vector<int> pin_cpus;
    // HY4_TP_EV: member 0's GPU time by part of each layer (events on its stream: the attention, its sum's wait, the
    // FFN, its sum's wait, the post-mixes), token windows only, printed at the end
    const bool tp_ev = std::getenv("HY4_TP_EV") != nullptr;
    std::vector<cudaEvent_t> evs;   // [layer][7]
    double ev_ms[6] = {0, 0, 0, 0, 0, 0};
    long ev_n = 0;
    std::map<int, std::pair<long, double>> ev_by_t;   // (T: windows, ms in total)
    bool ev_on = false;
    void mark(int m, Gpu& gp, int il, int k) {
        if (!ev_on || m != ev_m) return;
        cudaEventRecord(evs[(size_t) il * 7 + k], gp.s);
    }
    void ev_collect(int T) {
        if (!ev_on) return;
        cudaEventSynchronize(evs[(size_t) (g.n_layer - 1) * 7 + 6]);
        double tot = 0;
        for (int il = 0; il < g.n_layer; ++il)
            for (int k = 0; k < 6; ++k) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, evs[(size_t) il * 7 + k], evs[(size_t) il * 7 + k + 1]) != cudaSuccess) {
                    (void) cudaGetLastError();   // (a mark this path does not record: no time for it)
                    ms = 0;
                }
                ev_ms[k] += ms;
                tot += ms;
            }
        ++ev_n;
        auto& e = ev_by_t[T];
        ++e.first;
        e.second += tot;
    }
    void ev_report() const {
        if (tp_ev && stg_windows > 0)
            std::fprintf(stderr, "HY4_TP_EV member %d, %ld staggered chunks: %.1f ms each, of which %.1f ms in its sums' "
                         "read-backs (the waits and the copies)\n", ev_m, stg_windows, stg_all_ms / stg_windows,
                         stg_wait_ms / stg_windows);
        if (!tp_ev || ev_n == 0) return;
        const char* what[6] = {"attention", "its sum's wait", "post-mix", "FFN", "its sum's wait", "post-mix"};
        std::fprintf(stderr, "HY4_TP_EV member 0, %ld token windows, ms a window:", ev_n);
        for (int k = 0; k < 6; ++k) std::fprintf(stderr, " %s %.2f;", what[k], ev_ms[k] / ev_n);
        std::fprintf(stderr, "\n");
        for (const auto& [T, e] : ev_by_t)
            std::fprintf(stderr, "HY4_TP_EV   windows of %d: %ld, %.2f ms each\n", T, e.first, e.second / e.first);
    }
    // HY4_TP_PROF: the token windows' host time - waiting for the members' attention / FFN partials, the sums, the
    // whole forwards (printed at the end)
    const bool tp_prof = std::getenv("HY4_TP_PROF") != nullptr;
    // ([0]: token windows, [1]: prompt chunks - those also by token)
    double tp_wait[2][2] = {{0, 0}, {0, 0}}, tp_sum_s[2] = {0, 0}, tp_fwd_s[2] = {0, 0}, tp_prep_s[2] = {0, 0},
           tp_post_s[2] = {0, 0}, tp_eg_s[2] = {0, 0}, win_t0 = 0, eg_last_s = 0;
    long tp_fwd[2] = {0, 0}, tp_tok[2] = {0, 0};
    void tp_report() const {
        if (!tp_prof) return;
        for (int q = 0; q < 2; ++q) {
            if (tp_fwd[q] == 0) continue;
            const double f = 1e3 / (double) tp_fwd[q];
            std::fprintf(stderr, "HY4_TP_PROF %ld %s (%ld tokens): %.2f ms each to the last sum; waits for the attention "
                         "partials %.2f ms, the FFN's %.2f ms; the sums %.3f ms; the inputs before %.2f ms (Engram %.2f); "
                         "to the end %.2f ms\n",
                         tp_fwd[q], q ? "prompt chunks" : "token windows", tp_tok[q], tp_fwd_s[q] * f, tp_wait[q][0] * f,
                         tp_wait[q][1] * f, tp_sum_s[q] * f, tp_prep_s[q] * f, tp_eg_s[q] * f, tp_post_s[q] * f);
        }
    }

    // (m, N: tensor-parallel member m of N's slice - its heads, its share of the dense and shared FFNs' widths, its
    //  slices of the routed experts - the rest whole; N 1: the whole layer)
    bool load_layer(Layer& l, std::string& err, int m = 0, int N = 1, int units = 0) {
        Gpu& gp = G[(size_t) l.gpu];
        CK(cudaSetDevice(gp.dev));
        const int il = l.il, E = g.n_embd, hc = g.hc;
        const std::string p = "blk." + std::to_string(il) + ".";
        l.mtp = il >= g.n_layer;
        t_round_bf16 = l.mtp;
        l.tpm = N > 1 ? m : -1;
        const int Hall = g.heads[(size_t) il], h0 = (int) ((int64_t) Hall * m / N), h1 = (int) ((int64_t) Hall * (m + 1) / N);
        l.H = h1 - h0;
        l.swa = g.swa[(size_t) il] != 0;
        l.ix = !l.swa && g.indexer[(size_t) il] != 0 && g.ix_top_k > 0;
        l.dense = !(il >= g.n_layer) && il < g.n_dense_lead;
        l.base = l.swa ? g.rope_base_swa : g.rope_base;
        if (!l.swa && !l.ix) {
            err = "layer " + std::to_string(il) + ": a full-attention layer without an indexer (not run by maya-hy4)";
            return false;
        }
        const int nope = g.nope();
        if (!f32(gp, p + "attn_norm.weight", l.attn_norm, err, E) || !f32(gp, p + "ffn_norm.weight", l.ffn_norm, err, E) ||
            !f32(gp, p + "post_attention_norm.weight", l.post_attn, err, E) ||
            !f32(gp, p + "post_ffw_norm.weight", l.post_ffn, err, E) ||
            !f32(gp, p + "attn_q_a_norm.weight", l.q_a_norm, err, g.q_lora) ||
            !f32(gp, p + "attn_kv_a_norm.weight", l.kv_a_norm, err, g.kv_lora) ||
            !f32_part(gp, p + "attn_sinks.weight", l.sinks, err, Hall, h0, h1))
            return false;
        const int km = g.k_mla, vm = g.v_mla;
        if (!mat(gp, p + "attn_q_a.weight", l.q_a, err, E, g.q_lora) ||
            !mat_part(gp, p + "attn_q_b.weight", l.q_b, err, g.q_lora, Hall * km, 1, h0 * km, h1 * km, 0, g.q_lora, 0, 1) ||
            !mat(gp, p + "attn_kv_a_mqa.weight", l.kv_a, err, E, g.kv_row()) ||
            !mat_part(gp, p + "attn_k_b.weight", l.k_b, err, nope, g.kv_lora, Hall, 0, g.kv_lora, 0, nope, h0, h1) ||
            !mat_part(gp, p + "attn_v_b.weight", l.v_b, err, g.kv_lora, vm, Hall, 0, vm, 0, g.kv_lora, h0, h1) ||
            !mat_part(gp, p + "attn_output.weight", l.wo, err, Hall * vm, E, 1, 0, E, h0 * vm, h1 * vm, 0, 1) ||
            !mat_part(gp, p + "attn_gate.weight", l.gate, err, E, Hall * vm, 1, h0 * vm, h1 * vm, 0, E, 0, 1))
            return false;
        for (const Mat* m : {&l.q_a, &l.q_b, &l.kv_a, &l.k_b, &l.v_b, &l.wo, &l.gate})
            if (m->type == kF32 || m->type == kBF16 || m->type == kF16 || m->type == kF8) {
                err = p + ": an attention matrix in F32 / BF16 / F16 (maya-hy4 runs ggml's block types there)";
                return false;
            }
        if (l.mtp && (!f32(gp, p + "nextn.enorm.weight", l.enorm, err, E) ||
                      !f32(gp, p + "nextn.hnorm.weight", l.hnorm, err, E) ||
                      !f32(gp, p + "nextn.shared_head_norm.weight", l.head_norm, err, E) ||
                      !mat(gp, p + "nextn.eh_proj.weight", l.eh_proj, err, 2 * E, E)))
            return false;
        if (!l.mtp && (!f32(gp, p + "hc_attn_fn.weight", l.hca_fn, err, (int64_t) 2 * hc * hc * E) ||
            !f32(gp, p + "hc_attn_base.weight", l.hca_base, err, 2 * hc) ||
            !f32(gp, p + "hc_attn_scale.weight", l.hca_scale, err, 2) ||
            !f32(gp, p + "hc_ffn_fn.weight", l.hcf_fn, err, (int64_t) 2 * hc * hc * E) ||
            !f32(gp, p + "hc_ffn_base.weight", l.hcf_base, err, 2 * hc) ||
            !f32(gp, p + "hc_ffn_scale.weight", l.hcf_scale, err, 2)))
            return false;
        if (l.ix) {
            if (!mat(gp, p + "indexer.attn_q_b.weight", l.ix_q_b, err, g.q_lora, g.ix_heads * g.ix_dim) ||
                !mat(gp, p + "indexer.attn_k.weight", l.ix_k, err, E, g.ix_dim) ||
                !mat(gp, p + "indexer_compressor_value.weight", l.ix_val, err, g.ix_dim, g.ix_dim) ||
                !mat(gp, p + "indexer_compressor_gate.weight", l.ix_gate, err, g.ix_dim, g.ix_dim) ||
                !mat(gp, p + "indexer.proj.weight", l.ix_proj, err, E, g.ix_heads) ||
                !f32(gp, p + "indexer.k_norm.weight", l.ix_k_norm, err, g.ix_dim) ||
                !f32(gp, p + "indexer_compressor_ape.weight", l.ix_ape, err, (int64_t) g.ix_dim * g.ix_kpool) ||
                !f32(gp, p + "indexer_compressor_norm.weight", l.ix_pnorm, err, g.ix_dim))
                return false;
        }
        l.eg = l.mtp ? -1 : pk.eg.module_of(il);
        if (l.eg >= 0) {
            const int cols = pk.eg.cols();
            if (!mat(gp, p + "engram_kv.weight", l.eg_kv, err, cols * pk.eg.head_dim, E * (hc + 1)) ||
                !f32(gp, p + "engram_kv_b.weight", l.eg_kv_b, err, (int64_t) E * (hc + 1)) ||
                !f32(gp, p + "engram_gate.weight", l.eg_gate, err, (int64_t) E * hc) ||
                !f32(gp, p + "engram_lookup_norm.weight", l.eg_lookup, err, (int64_t) cols * pk.eg.head_dim) ||
                !f32(gp, p + "engram_pre_gamma.weight", l.eg_gamma, err, E) ||
                !f32(gp, p + "engram_hc_scale.weight", l.eg_hcs, err, (int64_t) E * hc))
                return false;
        }
        if (l.dense) {
            const Tensor* tdn = need(p + "ffn_down.weight", err);
            if (!tdn) return false;
            int f0 = 0, f1 = g.n_ff;
            if (N > 1) width_part(g.n_ff, type_id(tdn->type), m, N, f0, f1);
            if (!mat_part(gp, p + "ffn_gate.weight", l.ffn_gate, err, E, g.n_ff, 1, f0, f1, 0, E, 0, 1) ||
                !mat_part(gp, p + "ffn_up.weight", l.ffn_up, err, E, g.n_ff, 1, f0, f1, 0, E, 0, 1) ||
                !mat_part(gp, p + "ffn_down.weight", l.ffn_down, err, g.n_ff, E, 1, 0, E, f0, f1, 0, 1))
                return false;
        } else {
            const int lat = g.n_lat, ff = g.n_ff_exp, fs = ff * g.n_shared;
            const Tensor* tsd = need(p + "ffn_down_shexp.weight", err);
            if (!tsd) return false;
            int s0 = 0, s1 = fs;
            if (N > 1) width_part(fs, type_id(tsd->type), m, N, s0, s1);
            if (!mat(gp, p + "ffn_gate_inp.weight", l.router, err, E, g.n_expert) ||
                !f32(gp, p + "exp_probs_b.bias", l.router_b, err, g.n_expert) ||
                !mat(gp, p + "ffn_routed_down.weight", l.rdown, err, E, lat) ||
                !mat(gp, p + "ffn_routed_up.weight", l.rup, err, lat, E) ||
                !mat_part(gp, p + "ffn_gate_shexp.weight", l.sh_gate, err, E, fs, 1, s0, s1, 0, E, 0, 1) ||
                !mat_part(gp, p + "ffn_up_shexp.weight", l.sh_up, err, E, fs, 1, s0, s1, 0, E, 0, 1) ||
                !mat_part(gp, p + "ffn_down_shexp.weight", l.sh_down, err, fs, E, 1, 0, E, s0, s1, 0, 1))
                return false;
            if (l.router.type != kF32) {
                err = p + "ffn_gate_inp: not F32";
                return false;
            }
            const Tensor *tg = need(p + "ffn_gate_exps.weight", err), *tu = tg ? need(p + "ffn_up_exps.weight", err) : nullptr,
                         *td = tu ? need(p + "ffn_down_exps.weight", err) : nullptr;
            if (!td) return false;
            l.tg = type_id(tg->type);
            l.td = type_id(td->type);
            const bool round = t_round_bf16 && l.tg == kBF16 && l.td == kBF16 && type_id(tu->type) == kBF16;
            if (round) l.tg = l.td = 8;   // (the MTP block's BF16 experts, rounded to Q8_0 below)
            if (!iq::expert_supported(l.tg) || (!round && type_id(tu->type) != l.tg) || !iq::expert_supported(l.td) ||
                tg->n(0) != lat || tg->n(1) != ff || tg->n(2) != g.n_expert || td->n(0) != ff || td->n(1) != lat ||
                td->n(2) != g.n_expert) {
                err = p + ": routed experts " + tg->type + " / " + tu->type + " / " + td->type +
                      " (maya-hy4 runs ggml's expert types, gate and up of one type)";
                return false;
            }
            const size_t gb = iq::row_bytes(l.tg, lat) * (size_t) ff, db = iq::row_bytes(l.td, ff) * (size_t) lat;
            auto al = [](size_t b) { return (b + 255) / 256 * 256; };
            if (N > 1 && !l.mtp)
                return expert_slices(l, gp, err, m, units > 0 ? units : N, pk.data(*tg), pk.data(*tu), pk.data(*td)) &&
                       caches(l, gp, err);
            l.tr[0] = 0;
            l.tr[1] = al(gb);
            l.tr[2] = 2 * al(gb);
            l.blob = al(2 * al(gb) + db);
            l.ffe = ff;
            if (l.mtp || round) l.n_res = g.n_expert;   // (the MTP block's: always all in VRAM - rounded from BF16)
            l.n_res = std::max(0, std::min(l.n_res, g.n_expert));
            // the experts in VRAM: the most routed ones by the profile (Options::profile), else the first n_res
            l.slot_of.assign((size_t) g.n_expert, -1);
            std::vector<int> order((size_t) g.n_expert);
            for (int e = 0; e < g.n_expert; ++e) order[(size_t) e] = e;
            if (!l.mtp && (size_t) (il + 1) * g.n_expert <= prof_in.size()) {
                const uint32_t* c = prof_in.data() + (size_t) il * g.n_expert;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return c[a] > c[b]; });
            }
            std::vector<int> res(order.begin(), order.begin() + l.n_res);
            std::sort(res.begin(), res.end());
            for (int s2 = 0; s2 < l.n_res; ++s2) l.slot_of[(size_t) res[(size_t) s2]] = s2;
            if (!gp.alloc(l.experts, l.blob * (size_t) std::max(1, l.n_res), err) ||
                !gp.alloc(l.slot_d, (size_t) g.n_expert, err))
                return false;
            CK(cudaMemcpy(l.slot_d, l.slot_of.data(), sizeof(int) * g.n_expert, cudaMemcpyHostToDevice));
            const uint8_t *sg = pk.data(*tg), *su = pk.data(*tu), *sd = pk.data(*td);
            if (l.partial()) {   // (the CPU lane reads the others where they lie)
                if (!l.cex.init(l.tg, l.td, lat, ff, err)) {
                    err = p + ": the CPU lane does not take these experts: " + err;
                    return false;
                }
                l.cex.gate = sg;
                l.cex.up = su;
                l.cex.down = sd;
                l.cex.limit = g.limit[(size_t) il];
            }
            // an expert's three parts side by side in one blob (one copy a role; a pinned staging buffer a GPU would be
            // faster - the load reads the file at the page cache's pace anyway); the MTP block's BF16 ones rounded
            const size_t sgb = round ? (size_t) lat * ff * 2 : gb, sdb = round ? (size_t) ff * lat * 2 : db;
            std::vector<uint8_t> host(l.blob * 64);
            for (int s0 = 0; s0 < l.n_res; s0 += 64) {   // (slot s0 + i holds expert res[s0 + i])
                const int ne = std::min(64, l.n_res - s0);
                for (int i = 0; i < ne; ++i) {
                    uint8_t* b = host.data() + (size_t) i * l.blob;
                    const size_t e = (size_t) res[(size_t) (s0 + i)];
                    if (round) {
                        bf16_to_q8_0((const uint16_t*) (sg + e * sgb), (size_t) ff, lat, b + l.tr[0]);
                        bf16_to_q8_0((const uint16_t*) (su + e * sgb), (size_t) ff, lat, b + l.tr[1]);
                        bf16_to_q8_0((const uint16_t*) (sd + e * sdb), (size_t) lat, ff, b + l.tr[2]);
                        continue;
                    }
                    std::memcpy(b + l.tr[0], sg + e * gb, gb);
                    std::memcpy(b + l.tr[1], su + e * gb, gb);
                    std::memcpy(b + l.tr[2], sd + e * db, db);
                }
                CK(cudaMemcpy(l.experts + (size_t) s0 * l.blob, host.data(), l.blob * (size_t) ne,
                              cudaMemcpyHostToDevice));
            }
        }
        return caches(l, gp, err);
    }
    bool caches(Layer& l, Gpu& gp, std::string& err) {
        if (l.swa) {
            l.ring = ring;
            if (!gp.alloc(l.kv, (size_t) ring * kv_rb() / 2, err)) return false;
        } else {
            l.ring = 0;
            if (!gp.alloc(l.kv, (size_t) o.max_ctx * kv_rb() / 2, err) ||
                !gp.alloc(l.pooled, (size_t) (o.max_ctx / g.ix_kpool + 1) * g.ix_dim * pool_b() / 4, err) ||
                !gp.alloc(l.ivg, (size_t) ring * 2 * g.ix_dim * pool_b() / 4, err))
                return false;
            CK(cudaMemset(l.pooled, 0, (size_t) (o.max_ctx / g.ix_kpool + 1) * g.ix_dim * pool_b()));
        }
        return true;
    }
    // A helper's layer: its unit's slices of the routed experts, nothing else
    bool helper_layer(Layer& l, int unit, int w, std::string& err) {
        Gpu& gp = G[(size_t) l.gpu];
        CK(cudaSetDevice(gp.dev));
        const std::string p = "blk." + std::to_string(l.il) + ".";
        const Tensor *tg = need(p + "ffn_gate_exps.weight", err), *tu = tg ? need(p + "ffn_up_exps.weight", err) : nullptr,
                     *td = tu ? need(p + "ffn_down_exps.weight", err) : nullptr;
        if (!td) return false;
        l.tg = type_id(tg->type);
        l.td = type_id(td->type);
        l.tpm = unit;
        if (!iq::expert_supported(l.tg) || type_id(tu->type) != l.tg || !iq::expert_supported(l.td)) {
            err = p + ": routed experts the helper's kernels do not take";
            return false;
        }
        if (model_sum && !mat(gp, p + "ffn_routed_up.weight", l.rup, err, g.n_lat, g.n_embd)) return false;
        return expert_slices(l, gp, err, unit, U, pk.data(*tg), pk.data(*tu), pk.data(*td), w);
    }
    // A tensor-parallel member's slices of the layer's routed experts: block j of expert e's width (a block of the down
    // matrix's type, its gate / up rows and down columns) is member (e B + j) % N's (B blocks an expert), so an expert's
    // blocks go round the members and a member holds c(e) of them - ceil(B / N) at most; its slice of e is those blocks
    // side by side, zero blocks after them up to ceil(B / N) (a zero row adds nothing), an expert of none of its blocks
    // left out (slot -1).  The SwiGLU is a value's own, so the slices' sums are the expert's.
    // (units [m, m + w): a helper's several units in one slice - its blocks of each expert side by side)
    bool expert_slices(Layer& l, Gpu& gp, std::string& err, int m, int N, const uint8_t* sg, const uint8_t* su,
                       const uint8_t* sd, int w = 1) {
        const int lat = g.n_lat, ff = g.n_ff_exp, bv = iq::block_values(l.td), B = ff / bv;
        const size_t rg = iq::row_bytes(l.tg, lat), gb = rg * (size_t) ff;      // (a gate / up row, an expert's)
        const size_t bd = iq::row_bytes(l.td, bv), rd = iq::row_bytes(l.td, ff), db = rd * (size_t) lat;
        auto al = [](size_t b) { return (b + 255) / 256 * 256; };
        std::vector<std::vector<int>> mine((size_t) g.n_expert);
        int cmax = 0;
        for (int e = 0; e < g.n_expert; ++e) {
            for (int j = 0; j < B; ++j) {
                const int u = (int) (((int64_t) e * B + j) % N);
                if (u >= m && u < m + w) mine[(size_t) e].push_back(j);
            }
            cmax = std::max(cmax, (int) mine[(size_t) e].size());
        }
        cmax = std::max(cmax, 1);
        const size_t sgb = rg * (size_t) cmax * bv, sdb = bd * (size_t) cmax * lat;   // (a slice's gate, its down)
        l.ffe = cmax * bv;
        l.tr[0] = 0;
        l.tr[1] = al(sgb);
        l.tr[2] = 2 * al(sgb);
        l.blob = al(2 * al(sgb) + sdb);
        l.slot_of.assign((size_t) g.n_expert, -1);
        l.n_res = 0;
        for (int e = 0; e < g.n_expert; ++e)
            if (!mine[(size_t) e].empty()) l.slot_of[(size_t) e] = l.n_res++;
        if (!gp.alloc(l.experts, l.blob * (size_t) std::max(1, l.n_res), err) || !gp.alloc(l.slot_d, (size_t) g.n_expert, err))
            return false;
        CK(cudaMemcpy(l.slot_d, l.slot_of.data(), sizeof(int) * g.n_expert, cudaMemcpyHostToDevice));
        std::vector<uint8_t> host(l.blob * 64);
        int s0 = 0;
        std::vector<int> batch;
        auto flush = [&]() -> bool {
            if (batch.empty()) return true;
            CK(cudaMemcpy(l.experts + (size_t) s0 * l.blob, host.data(), l.blob * batch.size(), cudaMemcpyHostToDevice));
            s0 += (int) batch.size();
            batch.clear();
            return true;
        };
        for (int e = 0; e < g.n_expert; ++e) {
            if (l.slot_of[(size_t) e] < 0) continue;
            uint8_t* b = host.data() + batch.size() * l.blob;
            std::memset(b, 0, l.blob);
            const auto& js = mine[(size_t) e];
            for (size_t c = 0; c < js.size(); ++c) {
                const size_t j = (size_t) js[c];
                // gate / up: the block's bv rows; down: each of its lat rows' block j
                std::memcpy(b + l.tr[0] + c * bv * rg, sg + (size_t) e * gb + j * bv * rg, bv * rg);
                std::memcpy(b + l.tr[1] + c * bv * rg, su + (size_t) e * gb + j * bv * rg, bv * rg);
                for (int r = 0; r < lat; ++r)
                    std::memcpy(b + l.tr[2] + (size_t) r * cmax * bd + c * bd, sd + (size_t) e * db + (size_t) r * rd + j * bd,
                                bd);
            }
            batch.push_back(e);
            if (batch.size() == 64 && !flush()) return false;
        }
        return flush();
    }

    bool scratch(Gpu& gp, std::string& err) {
        CK(cudaSetDevice(gp.dev));
        const size_t E = (size_t) g.n_embd, hc = (size_t) g.hc, H = (size_t) gp.sH, maxk = (size_t) g.ix_top_k + g.ix_kpool;
        const size_t lat = (size_t) g.n_lat, ff = (size_t) (gp.sFF ? gp.sFF : g.n_ff), ffe = (size_t) g.n_ff_exp,
                     T = (size_t) Tc;
        const size_t egc = (size_t) std::max(1, pk.eg.cols()) * std::max(1, pk.eg.head_dim);
        const size_t qk = std::max({H * (size_t) g.v_mla, ff, H * (size_t) g.kv_lora, (size_t) g.q_lora, egc, E});
        const size_t npool = (size_t) (o.max_ctx / g.ix_kpool + 1);
        const size_t att = dense_att_values(kSub, H, (size_t) g.n_swa, (size_t) dense_max);
        if (cublasCreate(&gp.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(gp.blas, gp.s) != CUBLAS_STATUS_SUCCESS) {
            err = "GPU " + std::to_string(gp.dev) + ": cuBLAS did not start";
            return false;
        }
        gp.wband_n = kBand;
        return gp.alloc(gp.x, T * hc * E, err) && gp.alloc(gp.cur, T * E, err) && gp.alloc(gp.post, T * hc, err) &&
               gp.alloc(gp.xa, T * E, err) && gp.alloc(gp.qa, T * g.q_lora, err) && gp.alloc(gp.qr, T * g.q_lora, err) &&
               gp.alloc(gp.q, T * H * g.k_mla, err) && gp.alloc(gp.qcat, T * H * g.kv_row(), err) &&
               gp.alloc(gp.kvr, T * g.kv_row(), err) && gp.alloc(gp.o, T * H * g.kv_lora, err) &&
               gp.alloc(gp.ob, T * H * g.v_mla, err) && gp.alloc(gp.g, T * H * g.v_mla, err) &&
               gp.alloc(gp.y, T * E, err) && gp.alloc(gp.xf, T * E, err) && gp.alloc(gp.gu, T * ff * 2, err) &&
               gp.alloc(gp.hd, T * ff, err) && gp.alloc(gp.rl, T * g.n_expert, err) && gp.alloc(gp.rw, T * 16, err) &&
               gp.alloc(gp.ids, T * 16, err) && gp.alloc(gp.lat, T * lat, err) && gp.alloc(gp.latout, T * (lat + (gp.sFF ? E : 0)), err) &&
               gp.alloc(gp.xq, W * qk, err) && gp.alloc(gp.xd, W * qk / 32, err) &&
               gp.alloc(gp.x2q, 2 * W * lat, err) && gp.alloc(gp.x2d, 2 * W * lat / 32, err) &&
               gp.alloc(gp.eh, 16 * W * ffe, err) && gp.alloc(gp.ehq, 16 * W * ffe, err) &&
               gp.alloc(gp.ehd, 16 * W * ffe / 32, err) && gp.alloc(gp.part, 16 * W * lat, err) &&
               gp.alloc(gp.grp_ptr, 16 * W + 1, err) && gp.alloc(gp.plan, 16, err) &&
               gp.alloc(gp.grp_start, 16 * W + 2, err) && gp.alloc(gp.grp_ent, 16 * W, err) &&
               gp.alloc(gp.n_groups, 1, err) &&
               gp.alloc(gp.ws, k::attn_ws_floats(kSub, (int) H, g.kv_lora, (int) maxk), err) &&
               gp.alloc(gp.pos, 3 * T, err) && gp.alloc(gp.keys, T * maxk, err) && gp.alloc(gp.knk, T, err) &&
               gp.alloc(gp.iq, T * g.ix_heads * g.ix_dim, err) && gp.alloc(gp.ik, T * g.ix_dim, err) &&
               gp.alloc(gp.ivt, T * 2 * g.ix_dim, err) && gp.alloc(gp.wt, T * g.ix_heads, err) &&
               gp.alloc(gp.score, (size_t) kSub * npool, err) && gp.alloc(gp.eg_rows, T * egc, err) &&
               gp.alloc(gp.eg_valid, T, err) && gp.alloc(gp.eg_r, T * egc, err) &&
               (eg_scale == nullptr || gp.alloc(gp.eg_sc, T * (size_t) std::max(1, pk.eg.cols()), err)) &&
               gp.alloc(gp.eg_kvo, T * (hc + 1) * E, err) && gp.alloc(gp.hcur, T * E, err) &&
               gp.alloc(gp.xh16, T * qk, err) && gp.alloc(gp.wband, kBand, err) && gp.alloc(gp.sh16, 16 * T * ffe, err) &&
               gp.alloc(gp.sy, 16 * T * lat, err) && gp.alloc(gp.cpo, T * lat, err) && gp.alloc(gp.bounds, 1025, err) && gp.alloc(gp.stok, 16 * T, err) &&
               gp.alloc(gp.srow, 16 * T, err) && gp.alloc(gp.sS, att, err) && gp.alloc(gp.sP, att, err) &&
               gp.alloc(gp.dmask, (size_t) kSub * dense_max, err) &&
               gp.alloc(gp.kbuf, deq_rows() * g.kv_row(), err) &&
               (gp.lo = gp.pos + T, gp.nk = gp.pos + 2 * T, true);
    }

    bool load(const Options& opt, std::string& err) {
        const double t0 = now_s();
        o = opt;
        if (!pk.load(o.pack, err)) return false;
        g = pk.g;
        if (o.max_ctx <= 0 || o.max_ctx > g.context) o.max_ctx = g.context;
        o.max_ctx = (o.max_ctx + g.ix_kpool - 1) / g.ix_kpool * g.ix_kpool;
        auto set_chunk = [&](int c) {
            Tc = std::max(W, c);
            ring = (int) std::min<int64_t>(o.max_ctx, (int64_t) g.n_swa + std::max(o.ring_slack, Tc));
        };
        const bool chunk_auto = o.prefill_chunk <= 0;
        set_chunk(chunk_auto ? kChunkBase : o.prefill_chunk);
        kv_q8 = o.kv == 1;
        if (kv_q8 && (g.kv_lora % 32 || g.kv_lora != 512 || g.rope != 64)) {
            err = "--kv q8: a latent row of " + std::to_string(g.kv_lora) + " + " + std::to_string(g.rope) +
                  " values (the 8-bit rows take 512 + 64)";
            return false;
        }
        // the vision tower's geometry (its room is planned on the main GPU; it loads after the model)
        if (o.vision && Vision::in_pack(pk)) {
            std::string verr;
            if (Vision::geometry(pk, vgeo, verr)) {
                vis_room = (double) Vision::room_for(vgeo);
                // an image token's position has no n-gram (Engram's reference masks it, and a later token's history
                // pads it): its id maps to none
                if (pk.eg.modules > 0 && vgeo.image_token < pk.eg.vocab) {
                    vis_tok = vgeo.image_token;
                    vis_tok_map = pk.eg.token_map[(size_t) vis_tok];
                    pk.eg.token_map[(size_t) vis_tok] = -1;
                }
            } else if (!o.vision_auto) {
                err = verr;
                return false;
            } else {
                info.push_back("NOTE vision off: " + verr + " - text only");
                o.vision = false;
            }
        } else if (o.vision && !o.vision_auto) {
            err = "--vision on: the pack has no vision tower (pack the model again with its mmproj file: "
                  "tools/hy4/pack_gguf.py)";
            return false;
        } else {
            o.vision = false;
        }
        {
            // The plan, tried more than once: each try is the whole plan against the same readings of the VRAM, from
            // the state before it.
            // --chunk auto (no --chunk): the largest of kChunks whose plan keeps what the plan at kChunkBase comes to -
            // the layout, the cache's rows, the pictures (hy4_plan.hpp's chunk_pick).
            // Then the dense prompt path's span (dense_max, above): the largest of kDense whose plan keeps what the
            // plan at kDenseBase comes to with that chunk (dense_pick) - so the span costs no chunk, no picture, no
            // layout, no routed expert its place in VRAM (so a card that holds a part of the experts keeps kDenseBase,
            // its decoding and its answers as they were), and never a start.
            const VisionGeo vgeo0 = vgeo;
            const double room0 = vis_room;
            const bool vision0 = o.vision;
            const size_t info0 = info.size();
            auto again = [&](int c, int d) {
                tp = 0;
                mtp_own = false;
                mem.clear();
                hel.clear();
                TL.clear();
                L.clear();
                for (Gpu& gp : G) {
                    gp.sH = 64;
                    gp.sFF = 0;
                }
                kv_q8 = o.kv == 1;
                vgeo = vgeo0;
                vis_room = room0;
                o.vision = vision0;
                if (vision0 && vis_tok >= 0) pk.eg.token_map[(size_t) vis_tok] = -1;
                info.resize(info0);
                dense_max = d;
                set_chunk(c);
            };
            auto plan_at = [&](int ch, int d) {
                again(ch, d);
                std::string e;
                ChunkTry t;
                t.ok = plan(e);
                t.tp = tp;
                t.q8 = kv_q8;
                t.vision = o.vision;
                t.pixels = o.vision ? vgeo.max_pixels : 0;
                t.own = mtp_own;
                for (const Layer& l : L) t.res += l.n_res;   // (the layer split's; tensor parallel holds them all)
                return t;
            };
            const int c = !chunk_auto ? o.prefill_chunk
                                      : chunk_pick([&](int ch) { return plan_at(ch, kDenseBase); }, kChunks, kChunkBase);
            again(c, kDenseBase);
            if (!plan(err)) return false;   // (the start as it was before the span: its own error)
            // the spans tried: kDense on GPUs with tensor cores (compute capability 7.0 and later: on the others the
            // F16 products of a longer span were not measured against the gathering kernel), or HY4_DENSE_MAX's own
            std::vector<int> spans;
            const char* de = std::getenv("HY4_DENSE_MAX");
            if (de) {
                const int d = std::min(kDenseTop, std::atoi(de)) / 8 * 8;
                if (d > kDenseBase) spans.push_back(d);
            } else {
                bool tensor_cores = !G.empty();
                for (Gpu& gp : G) {
                    cudaDeviceProp pi{};
                    CK(cudaGetDeviceProperties(&pi, gp.dev));
                    tensor_cores = tensor_cores && pi.major >= 7;
                }
#if defined(STRATA_USE_HIP)
                tensor_cores = false;   // (AMD: measured slower past kDenseBase on a 16K prompt - the span stays)
#endif
                if (tensor_cores) spans.assign(std::begin(kDense), std::end(kDense));
            }
            if (!spans.empty()) {
                // (HY4_DENSE_MAX names a span: taken where the plan holds it, at the price of experts too)
                const int d = dense_pick([&](int dd) {
                    ChunkTry t = plan_at(c, dd);
                    if (de) t.res = 0;
                    return t;
                }, spans, kDenseBase);
                again(c, d);
                if (!plan(err)) return false;
            }
            char cb[320];
            if (chunk_auto) {
                std::snprintf(cb, sizeof cb, "INFO prompt chunk: %d tokens (auto: the largest of %d / %d / %d this VRAM "
                              "holds beside the model, its cache and the pictures; --chunk N sets one)", c, kChunks[0],
                              kChunks[1], kChunkBase);
                info.push_back(cb);
            }
            if (dense_max != kDenseBase || de) {
                std::snprintf(cb, sizeof cb, "INFO prompts: the indexed layers' attention on the tensor cores up to %d "
                              "positions%s", dense_max,
                              dense_max != kDenseBase ? "" : " (a longer span does not fit beside this start's plan)");
                info.push_back(cb);
            }
        }
        if (o.kv == 1) info.push_back("INFO the latent cache in 8-bit rows (--kv q8)");
        if (!o.profile.empty()) {
            std::ifstream f(o.profile, std::ios::binary);
            std::vector<uint32_t> v((size_t) g.n_layer * g.n_expert);
            if (f && f.read((char*) v.data(), (std::streamsize) (v.size() * 4))) prof_in = std::move(v);
            else info.push_back("INFO the expert profile " + o.profile + " was not read (missing, or another model's)");
        }
        // the host's parts: the token embeddings and the Engram table, read in place
        const Tensor* te = need("token_embd.weight", err);
        if (!te) return false;
        tok_type = type_id(te->type);
        tok_ld = row_bytes(tok_type, g.n_embd);
        if (!tok_type_ok(tok_type) || tok_ld == 0 || te->n(0) != g.n_embd || te->n(1) < g.n_vocab ||
            tok_ld * (size_t) te->n(1) != te->bytes) {
            err = "token_embd: " + te->type + " (maya-hy4 reads Q8_0, Q4_K, Q5_K, Q6_K, F32, BF16 or F16 embeddings)";
            return false;
        }
        tok_embd = pk.data(*te);
        if (pk.eg.modules > 0) {
            const Tensor* tt = need("per_layer_token_embd.weight", err);
            if (!tt) return false;
            if (tt->type != "F8_E4M3" || tt->n(0) != pk.eg.head_dim || tt->n(1) < pk.eg.rows) {
                err = "per_layer_token_embd: " + tt->type + " (F8_E4M3 rows of the Engram head dim expected)";
                return false;
            }
            eg_table = pk.data(*tt);
            eg_bytes = tt->bytes;
            if (pk.has("engram_eps")) eg_eps = (float) pk.num("engram_eps");
            if (const char* ee = std::getenv("HY4_ENGRAM_EPS")) eg_eps = (float) std::atof(ee);
            if (!(eg_eps > 0.0f) || eg_eps > 1.0f) {
                err = "the Engram norms' eps " + std::to_string(eg_eps) + " (engram_eps / HY4_ENGRAM_EPS)";
                return false;
            }
            const char* es = std::getenv("HY4_ENGRAM_SCALES");
            if (pk.has("engram_scales") && !(es && std::atoi(es) == 0)) {
                const std::string sp0 = pk.str("engram_scales");
                const std::string sp = !sp0.empty() && sp0[0] == '/' ? sp0 : pk.model_dir + "/" + sp0;
                size_t sz = 0;
                eg_scale = mm3::os::map_file(sp, sz);
                if (eg_scale == nullptr || sz < (size_t) tt->n(1) * 2) {
                    err = sp + ": not the Engram table's row scales (BF16, one a row)";
                    return false;
                }
                CK(cudaMallocHost(&h_egs, sizeof(float) * (size_t) pk.eg.modules * Tc * pk.eg.cols()));
            }
            if (const char* ed = std::getenv("HY4_EG_DUMP")) eg_dump = std::fopen(ed, "wb");
            {
                char buf[200];
                std::snprintf(buf, sizeof buf, "INFO Engram: %d layers, the table read from the disk a row at a time; "
                              "norm eps %.9g; row scales %s", pk.eg.modules, (double) eg_eps, eg_scale ? "on" : "none");
                info.push_back(buf);
            }
            // its pages mapped into this process in the background (MADV_POPULATE_READ, else a read a page): a
            // window's lookups land on random pages of tens of GB, and each first touch is a page fault - faults that
            // queue on the process's memory map.  Off unless asked (HY4_EG_POPULATE=1): the table stays on the disk and
            // a looked-up row's page is read at its first use - Maya never pulls an n-gram table into RAM by itself
            const char* ep = std::getenv("HY4_EG_POPULATE");
            if (ep && std::atoi(ep) != 0) {
                eg_populator = std::thread([this] {
#if defined(__linux__)
                    (void) nice(10);
                    const size_t pg = 4096, chunk = (size_t) 256 << 20;
                    uint8_t* a = (uint8_t*) ((uintptr_t) eg_table & ~(uintptr_t) (pg - 1));
                    uint8_t* end = (uint8_t*) eg_table + eg_bytes;
                    for (uint8_t* c = a; c < end && !placer_stop.load(); c += chunk) {
                        const size_t n = (size_t) std::min<ptrdiff_t>((ptrdiff_t) chunk, end - c);
#if defined(MADV_POPULATE_READ)
                        if (madvise(c, n, MADV_POPULATE_READ) == 0) continue;
#endif
                        volatile uint8_t sink = 0;
                        for (size_t b = 0; b < n; b += pg) sink ^= c[b];
                    }
#endif
                });
            }
        }
        // every GPU's layers, a thread a GPU (tensor parallel: every member's slices)
        if (tp > 1) {
            if (!load_tp(err)) return false;
        } else {
            for (int il = 0; il < g.n_layer; ++il) L[(size_t) il].il = il;
            std::vector<std::string> errs(G.size());
            std::vector<std::thread> th;
            for (size_t gi = 0; gi < G.size(); ++gi)
                th.emplace_back([&, gi] {
                    for (auto& l : L)
                        if (l.gpu == (int) gi && !load_layer(l, errs[gi])) return;
                    bool used = (int) gi == head_gpu;
                    for (auto& l : L) used |= l.gpu == (int) gi;
                    if (used) scratch(G[gi], errs[gi]);
                });
            for (auto& t : th) t.join();
            for (auto& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        // the head
        Gpu& hg = G[(size_t) head_gpu];
        CK(cudaSetDevice(hg.dev));
        if (!mat(hg, "output.weight", out, err, g.n_embd, g.n_vocab) || !f32(hg, "output_norm.weight", out_norm, err, g.n_embd) ||
            !f32(hg, "output_hc_fn.weight", hh_fn, err, (int64_t) g.hc * g.hc * g.n_embd) ||
            !f32(hg, "output_hc_base.weight", hh_base, err, g.hc) || !f32(hg, "output_hc_scale.weight", hh_scale, err, 1) ||
            !hg.alloc(hg.logits, (size_t) W * g.n_vocab, err) || !hg.alloc(hring, (size_t) ring * g.n_embd, err))
            return false;
        if (o.mtp && g.n_mtp > 0) {   // (after the head: its BF16 rounding is the MTP block's alone)
            const double tm = now_s();
            M.il = g.n_layer;
            M.gpu = head_gpu;
            bool ok = load_layer(M, err);
            const char* dh = std::getenv("HY4_DRAFT_HEAD");
            if (ok && out.type == kBF16 && !(dh && std::string(dh) == "q8")) {
                ok = draft_head_q4k(hg, err);
            } else if (ok && out.type == kBF16) {
                t_round_bf16 = true;
                ok = mat(hg, "output.weight", out_draft, err, g.n_embd, g.n_vocab);
            }
            t_round_bf16 = false;
            if (!ok) return false;
            if (!out_draft) out_draft = out;
            const size_t E = (size_t) g.n_embd, T = (size_t) Tc;
            if (!hg.alloc(mtp_in, T * 2 * E, err) || !hg.alloc(mtp_cur, T * E, err) || !hg.alloc(mtp_hin, T * E, err) ||
                !hg.alloc(mtp_out, T * E, err) || !hg.alloc(mtp_top, 2, err))
                return false;
            CK(cudaMallocHost(&h_emb, sizeof(float) * T * E));
            if (tp > 1 && mtp_own) {   // (the prompt chunks' MTP pass beside the next chunk: the plan counted its scratch)
                hm.dev = hg.dev;
                CK(cudaStreamCreateWithFlags(&hm.s, cudaStreamNonBlocking));
                std::string e2;
                if (scratch(hm, e2)) {
                    CK(cudaEventCreateWithFlags(&ev_hin, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&ev_mtp, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&ev_emb, cudaEventDisableTiming));
                    CK(cudaEventRecord(ev_emb, hm.s));
                    CK(cudaEventRecord(ev_mtp, hm.s));
                    mtp_crew = std::make_unique<Crew>(1);
                    mtp_async = true;
                } else {
                    info.push_back("INFO the MTP block's pass in line with the chunks (no VRAM for its own scratch: " +
                                   e2 + ")");
                    (void) cudaGetLastError();
                    // (what it did get goes back: a part of a scratch must not take the room the kernels' own
                    //  buffers are allocated from at the warm-up)
                    for (void* p : hm.allocs) cudaFree(p);
                    hm.allocs.clear();
                    hm.used = 0;
                    if (hm.blas) cublasDestroy(hm.blas);
                    hm.blas = nullptr;
                }
            }
            char buf[160];
            std::snprintf(buf, sizeof buf, "INFO the MTP block (drafts): its BF16 matrices rounded to Q8_0 in %.1f s",
                          now_s() - tm);
            info.push_back(buf);
        }
        if (g.n_mtp == 0)   // (a set without the MTP block - the public release's layout: every token by the model itself)
            info.push_back(std::string("INFO no MTP draft block in this model's files: decoding without drafts") +
                           (o.mtp ? " (--mtp has nothing to load)" : ""));
        if (out.type != kBF16 && out.type != kF32 && !iq::mv_supported(out.type)) {
            err = "output.weight: a type maya-hy4 does not read";
            return false;
        }
        // host staging
        CK(cudaMallocHost(&h_x, sizeof(float) * Tc * g.hc * g.n_embd));
        CK(cudaMallocHost(&h_eg, (size_t) std::max(1, pk.eg.modules) * Tc * std::max(1, pk.eg.cols()) *
                                     std::max(1, pk.eg.head_dim)));
        CK(cudaMallocHost(&h_valid, sizeof(float) * Tc));
        CK(cudaMallocHost(&h_pos, sizeof(int) * 3 * Tc));
        int n_partial = 0, n_res = 0, n_all = 0;
        for (const Layer& l : L)
            if (!l.dense) {
                n_partial += l.partial();
                n_res += l.n_res;
                n_all += g.n_expert;
            }
        if (n_partial == 0 && !o.profile_out.empty()) CK(cudaMallocHost(&h_ids, sizeof(int) * Tc * 16));
        if (n_partial > 0) {
            lane = std::make_unique<Lane>(o.cpu_threads);
            CK(cudaMallocHost(&h_ids, sizeof(int) * Tc * 16));
            CK(cudaMallocHost(&h_w, sizeof(float) * Tc * 16));
            CK(cudaMallocHost(&h_lat, sizeof(float) * Tc * g.n_lat));
            CK(cudaMallocHost(&h_cpu, sizeof(float) * Tc * g.n_lat));
            size_t blob = 0;
            for (const Layer& l : L) blob = std::max(blob, l.blob);
            CK(cudaMallocHost(&h_stage, blob * kStage));
            CK(cudaMallocHost(&h_pmap, sizeof(int) * g.n_expert));
            for (size_t gi = 0; gi < G.size(); ++gi) {   // (the staging slots where a GPU holds such layers)
                bool any = false;
                for (const Layer& l : L) any |= l.gpu == (int) gi && l.partial();
                if (!any) continue;
                CK(cudaSetDevice(G[gi].dev));
                if (!G[gi].alloc(G[gi].stage, blob * kStage, err) || !G[gi].alloc(G[gi].pmap, (size_t) g.n_expert, err))
                    return false;
                // the GPU's PCIe as streaming sees it: the pinned bounce to the staging slots
                CK(cudaMemcpy(G[gi].stage, h_stage, blob * kStage, cudaMemcpyHostToDevice));
                const double t0 = now_s();
                for (int r = 0; r < 2; ++r)
                    CK(cudaMemcpy(G[gi].stage, h_stage, blob * kStage, cudaMemcpyHostToDevice));
                G[gi].h2d_gbs = 2.0 * blob * kStage / 1e9 / std::max(1e-6, now_s() - t0);
            }
            char buf[240];
            std::snprintf(buf, sizeof buf, "INFO CPU lane: %d of %d routed experts in VRAM (%.0f%%), the rest on %d CPU "
                          "threads from the model's file%s", n_res, n_all, 100.0 * n_res / std::max(1, n_all),
                          lane->threads() + 1, lane->nodes() > 1 ? " (their pages spread over the NUMA nodes in the "
                                                                   "background)" : "");
            info.push_back(buf);
            if (lane->nodes() > 1) {
                const int nodes = lane->nodes();
                placer = std::thread([this, nodes] {
#if defined(__linux__)
                    (void) nice(10);
#endif
                    for (const Layer& l : L) {
                        if (!l.partial()) continue;
                        std::vector<int> cpu;
                        for (int e = 0; e < (int) l.slot_of.size(); ++e)
                            if (l.slot_of[(size_t) e] < 0) cpu.push_back(e);
                        place_experts(l.cex, cpu, nodes, placer_stop);
                    }
                });
            } else {
                // one node: the lane's experts read into this process in the background when the RAM holds them
                // all (file cache included) - a decode's first touch of an expert then pays no page fault a page,
                // nor the disk after a cold start.  Asked for (HY4_LANE_WARM=1): the thread takes a
                // minute or two a 40 GB of experts on Windows and slows a decode beside it a little meanwhile
                uint64_t lane_bytes = 0;
                for (const Layer& l : L)
                    if (l.partial())
                        lane_bytes += (uint64_t) (g.n_expert - l.n_res) * (2 * l.cex.g_stride + l.cex.d_stride);
                const char* we = std::getenv("HY4_LANE_WARM");
                const uint64_t avail = ram_available();
                if (we && std::atoi(we) != 0 && avail > 0 && (double) avail > 1.1 * (double) lane_bytes + 2e9) {
                    info.push_back("INFO CPU lane: its experts are read into memory in the background");
                    placer = std::thread([this] {
#if defined(__linux__)
                        (void) nice(10);
#endif
                        for (const Layer& l : L) {
                            if (!l.partial()) continue;
                            std::vector<int> cpu;
                            for (int e = 0; e < (int) l.slot_of.size(); ++e)
                                if (l.slot_of[(size_t) e] < 0) cpu.push_back(e);
                            warm_experts(l.cex, cpu, placer_stop);
                        }
                    });
                }
            }
        }
        st.experts_vram = n_res;
        st.experts_all = n_all;
        // peer access between neighbours (a hop a GPU boundary)
        for (size_t i = 0; i + 1 < G.size(); ++i) {
            int can = 0;
            cudaDeviceCanAccessPeer(&can, G[i + 1].dev, G[i].dev);
            if (can) {
                cudaSetDevice(G[i + 1].dev);
                (void) cudaDeviceEnablePeerAccess(G[i].dev, 0);
                (void) cudaGetLastError();
            }
        }
        if (o.vision) {   // the vision tower: host memory (the pack's mapping), a picture's room on the main GPU
            vis = std::make_unique<Vision>();
            std::string verr;
            const int vdev = G[(size_t) head_gpu].dev;
            if (!vis->load(pk, vdev, verr)) {
                if (!o.vision_auto) {
                    err = verr;
                    return false;
                }
                info.push_back("NOTE vision off: the tower did not load on CUDA" + std::to_string(vdev) + " (" + verr +
                               ") - text only");
                vis.reset();
                cudaGetLastError();
            } else {
                char vb[320];
                std::snprintf(vb, sizeof vb, "INFO vision tower: %zu MB in host memory (the pack's mapping), a picture "
                              "at a time on CUDA%d (up to %.0f MB, kept free there between pictures), up to %d pixels "
                              "a picture (a square one: %d image tokens)", vis->host_bytes() >> 20, vdev,
                              vis_room / 1048576.0, vgeo.max_pixels,
                              vgeo.tokens((int) std::sqrt((double) vgeo.max_pixels) / vgeo.factor() * vgeo.factor(),
                                          (int) std::sqrt((double) vgeo.max_pixels) / vgeo.factor() * vgeo.factor()));
                info.push_back(vb);
            }
        }
        for (auto& gp : G) st.vram_gb += (double) gp.used / 1e9;
        st.load_s = now_s() - t0;
        char buf[512];
        std::snprintf(buf, sizeof buf, "INFO maya-hy4: Hy4-flash, %d layers, %d experts (%d a token), context %lld, %.1f GB "
                                       "of VRAM over %zu GPUs, loaded in %.1f s",
                      g.n_layer, g.n_expert, g.n_used, (long long) o.max_ctx, st.vram_gb, G.size(), st.load_s);
        info.push_back(buf);
        if (tp > 1) return true;
        std::string split = "INFO layers by GPU:";
        for (size_t gi = 0; gi < G.size(); ++gi) {
            int a = -1, b = -1;
            for (auto& l : L)
                if (l.gpu == (int) gi) {
                    if (a < 0) a = l.il;
                    b = l.il;
                }
            if (a >= 0) split += " GPU" + std::to_string(G[gi].dev) + ":" + std::to_string(a) + "-" + std::to_string(b);
        }
        info.push_back(split);
        return true;
    }

    // ------------------------------------------------------------ the forward
    // HY4_DUMP=1: the sums of the steps the reference graph names (its eval callback's), for the comparison - rows of
    // `width` values at `off` in rows ld apart
    const bool dump = std::getenv("HY4_DUMP") != nullptr;
    void dbg(const char* name, int il, Gpu& gp, const float* p, size_t rows, size_t ld, size_t width = 0,
             size_t off = 0) {
        if (!dump) return;
        if (width == 0) width = ld;
        cudaStreamSynchronize(gp.s);
        std::vector<float> h(rows * ld);
        cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost);
        double s = 0;
        for (size_t r = 0; r < rows; ++r)
            for (size_t i = 0; i < width; ++i) s += h[r * ld + off + i];
        std::fprintf(stderr, "DUMP %-24s sum = %.6f\n", (std::string(name) + "-" + std::to_string(il)).c_str(), s);
    }
    // HY4_INJECT=<dir> (tests): the reference's own tensors (a dump's files: int32 4, int64 ne[4], F32 data) as each
    // layer's input, and its outputs measured against them - every layer's error alone
    // A file holds the whole prompt's rows (a dump in one ubatch): a window reads its own rows (cur_p0 on).
    // HY4_INJECT_FROM=<pos>: only the windows from that position are measured, the per-name sums printed at the end
    // of each forward (inject_report); one window prompts print every comparison as it is made.
    const char* inject = std::getenv("HY4_INJECT");
    // (tests: the layers' time by kind, synced at each step; the names with a leading space are parts of "ffn, experts")
    const bool prof = (lane_prof().on = std::getenv("HY4_PROF") != nullptr);
    std::map<std::string, double> prof_acc;
    void prof_report() {
        double all = 0;
        for (const auto& [k2, v] : prof_acc)
            if (k2[0] != ' ') all += v;
        for (const auto& [k2, v] : prof_acc)
            std::fprintf(stderr, "PROF %-36s %8.3f s  %5.1f%%\n", k2.c_str(), v, 100.0 * v / std::max(all, 1e-9));
        const LaneProf& lp = lane_prof();
        if (lp.calls > 0)
            std::fprintf(stderr, "PROF lane calls %lld, entries %lld: grouping %.3f s, gate/up %.3f s, down %.3f s, sums %.3f s\n",
                         lp.calls, lp.entries, lp.prep_s, lp.gate_up_s, lp.down_s, lp.sum_s);
    }
    int64_t cur_p0 = 0;
    std::map<std::string, std::vector<float>> ref_cache;
    std::map<std::string, std::array<double, 3>> ref_acc;   // name -> d2, r2, max
    const std::vector<float>& ref_load(const std::string& name) {
        auto it = ref_cache.find(name);
        if (it != ref_cache.end()) return it->second;
        std::vector<float>& v = ref_cache[name];
        std::ifstream f(std::string(inject) + "/" + name + ".bin", std::ios::binary);
        if (!f) return v;
        int32_t nd;
        int64_t ne[4];
        f.read((char*) &nd, 4);
        f.read((char*) ne, 32);
        v.resize((size_t) (ne[0] * ne[1] * ne[2] * ne[3]));
        f.read((char*) v.data(), (std::streamsize) (v.size() * 4));
        return v;
    }
    // the window's rows of a file of rows of `n / T` values (null: not there, or not that shape)
    const float* ref_rows(const std::string& name, size_t n, int T) {
        const std::vector<float>& r = ref_load(name);
        const size_t row = n / (size_t) T;
        if (r.empty() || r.size() % row || (size_t) (cur_p0 + T) * row > r.size()) return nullptr;
        return r.data() + (size_t) cur_p0 * row;
    }
    void ref_cmp(const char* name, int il, Gpu& gp, const float* p, size_t n, int T = -1) {
        const std::string key = std::string(name) + (il >= 0 ? "-" + std::to_string(il) : "");
        const std::vector<float>& full = ref_load(key);
        if (full.empty()) return;
        const float* r = T > 0 ? ref_rows(key, n, T) : (full.size() == n ? full.data() : nullptr);
        if (!r) {
            std::fprintf(stderr, "CMP %s: %zu values, ours %zu\n", key.c_str(), full.size(), n);
            return;
        }
        cudaStreamSynchronize(gp.s);
        std::vector<float> h(n);
        cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost);
        if (std::getenv("HY4_SAVE")) {
            std::ofstream f(std::string(inject) + "/ours_" + key + ".bin", std::ios::binary);
            f.write((const char*) h.data(), (std::streamsize) (n * 4));
        }
        double d2 = 0, r2 = 0, mx = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) h[i] - r[i];
            d2 += d * d;
            r2 += (double) r[i] * r[i];
            mx = std::max(mx, std::fabs(d));
        }
        const char* from = std::getenv("HY4_INJECT_FROM");
        if (from) {
            if (cur_p0 < std::atoll(from)) return;
            auto& a = ref_acc[key];
            a[0] += d2;
            a[1] += r2;
            a[2] = std::max(a[2], mx);
            return;
        }
        std::fprintf(stderr, "CMP %-22s rel %.2e  max %.3e\n", key.c_str(), std::sqrt(d2 / std::max(r2, 1e-30)), mx);
    }
    void inject_report() {
        for (const auto& [k, a] : ref_acc)
            std::fprintf(stderr, "CMP %-22s rel %.2e  max %.3e\n", k.c_str(), std::sqrt(a[0] / std::max(a[1], 1e-30)),
                         a[2]);
        ref_acc.clear();
    }
    // y = M x for T columns (x rows ldx apart, y rows ldy apart): GEMVs over int8 inputs up to W columns (ds4_iq's), else
    // the prompt path's GEMMs - the weights decoded to F16 a band at a time, cuBLAS with F32 sums (F32 weights: SGEMM)
    bool mm(Gpu& gp, const Mat& M, const float* x, size_t ldx, int T, float* y, size_t ldy, float alpha = 1.0f) {
        if (T > W) return gemm(gp, M, x, ldx, T, y, ldy, alpha);
        if (M.type == kF32) {
            k::gemv_f32w((const float*) M.w, M.ld / 4, M.k, M.n, x, ldx, T, y, ldy, alpha, gp.s);
            return true;
        }
        if (M.type == kBF16) {
            k::gemv_bf16w((const uint16_t*) M.w, M.ld / 2, M.k, M.n, x, ldx, T, y, ldy, alpha, gp.s);
            return true;
        }
        k::q8_rows(x, ldx, T, M.k, gp.xq, gp.xd, gp.s);
        iq::MvJob j;
        j.w = M.w;
        j.ld = M.ld;
        j.type = M.type;
        j.k = M.k;
        j.n = M.n;
        j.ncol = T;
        j.xq = gp.xq;
        j.xd = gp.xd;
        j.y = y;
        j.ycs = ldy;
        j.alpha = alpha;
        return iq::mv(&j, 1, gp.s);
    }
    // Several matrices over the same input (y_i = M_i x): on the token path its int8 rows made once and the GEMVs in one
    // launch (ds4_iq's jobs, by type); else each one's own mm
    struct Out {
        const Mat* m;
        float* y;
        size_t ldy;
    };
    bool mm_multi(Gpu& gp, std::initializer_list<Out> outs, const float* x, size_t ldx, int T) {
        bool quant = T <= W;
        int k = -1;
        for (const Out& o : outs) {
            quant = quant && o.m->type != kF32 && o.m->type != kBF16 && o.m->type != kF16;
            if (k < 0) k = o.m->k;
            quant = quant && o.m->k == k;
        }
        if (!quant || outs.size() > 64) {
            for (const Out& o : outs)
                if (!mm(gp, *o.m, x, ldx, T, o.y, o.ldy)) return false;
            return true;
        }
        k::q8_rows(x, ldx, T, k, gp.xq, gp.xd, gp.s);
        std::vector<iq::MvJob> jobs;
        for (const Out& o : outs) {
            iq::MvJob j;
            j.w = o.m->w;
            j.ld = o.m->ld;
            j.type = o.m->type;
            j.k = o.m->k;
            j.n = o.m->n;
            j.ncol = T;
            j.xq = gp.xq;
            j.xd = gp.xd;
            j.y = o.y;
            j.ycs = o.ldy;
            jobs.push_back(j);
        }
        return iq::mv(jobs.data(), (int) jobs.size(), gp.s);
    }
    bool gemm(Gpu& gp, const Mat& M, const float* x, size_t ldx, int T, float* y, size_t ldy, float alpha) {
        const float beta = 0.0f;
        if (M.type == kF32) {
            return cublasSgemm(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, M.n, T, M.k, &alpha, (const float*) M.w,
                               (int) (M.ld / 4), x, (int) ldx, &beta, y, (int) ldy) == CUBLAS_STATUS_SUCCESS;
        }
        if (M.type == kBF16 || M.type == kF16 || !iq::mv_supported(M.type)) return false;
        k::to_f16(x, ldx, T, M.k, gp.xh16, (size_t) M.k, gp.s);
        const int band = (int) std::max<size_t>(1, gp.wband_n / (size_t) M.k);
        for (int r0 = 0; r0 < M.n; r0 += band) {
            const int nb = std::min(band, M.n - r0);
            if (!iq::f16_rows(M.type, M.w + (size_t) r0 * M.ld, M.k, nb, gp.wband, M.k, gp.s)) return false;
            if (cublasGemmEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, nb, T, M.k, &alpha, gp.wband, CUDA_R_16F, M.k, gp.xh16,
                             CUDA_R_16F, M.k, &beta, y + r0, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        return true;
    }
    // a 3D matrix head by head: y head h = M[h] x head h; x [T][H][..] (ldt, ldh), y + h ldyh, columns ldyt apart
    bool mm_heads(Gpu& gp, const Mat& M, const float* x, size_t ldt, size_t ldh, int T, int H, float* y, size_t ldyt,
                  size_t ldyh) {
        if (T > W) {   // (one strided batched GEMM over the heads: every head's matrix decoded to F16 at once)
            const size_t per = (size_t) M.n * M.k;
            if (per * H > gp.wband_n) return false;
            if (!iq::f16_rows(M.type, M.w, M.k, M.n * H, gp.wband, M.k, gp.s)) return false;
            k::to_f16(x, ldt, T, (int) (ldh * (H - 1) + M.k), gp.xh16, ldt, gp.s);
            const float one = 1.0f, zero = 0.0f;
            return cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, M.n, T, M.k, &one, gp.wband,
                                              CUDA_R_16F, M.k, (long long) per, gp.xh16, CUDA_R_16F, (int) ldt,
                                              (long long) ldh, &zero, y, CUDA_R_32F, (int) ldyt, (long long) ldyh, H,
                                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
        }
        k::q8_heads(x, ldt, ldh, T, H, M.k, gp.xq, gp.xd, gp.s);
        std::vector<iq::MvJob> jobs((size_t) H);
        for (int h = 0; h < H; ++h) {
            iq::MvJob& j = jobs[(size_t) h];
            j.w = M.w + (size_t) h * M.n * M.ld;
            j.ld = M.ld;
            j.type = M.type;
            j.k = M.k;
            j.n = M.n;
            j.ncol = T;
            j.xq = gp.xq + (size_t) h * T * M.k;
            j.xd = gp.xd + (size_t) h * T * M.k / 32;
            j.y = y + (size_t) h * ldyh;
            j.ycs = ldyt;
        }
        for (int h0 = 0; h0 < H; h0 += 64)
            if (!iq::mv(jobs.data() + h0, std::min(64, H - h0), gp.s)) return false;
        return true;
    }

    // A prompt sub-batch's attention (tokens t0 .. t0 + ts - 1 of the chunk) on the tensor cores: S = Q K^T over every
    // key position the sub-batch may see, the softmax over each token's own (its window, or the indexer's picks:
    // `all` - every position up to it, as long as the picks are every pool), O = P V.  false: an indexed layer whose
    // keys span more than dense_max positions (the caller's gathering kernel instead).
    // (an 8-bit cache: the rows [base, hi) of a layer's call dequantized in gp.kbuf so far - each sub-batch adds the
    //  rows it needs beyond them, so a chunk's rows are dequantized once a layer)
    struct Deq {
        int64_t base = -1, hi = 0;
    };
    bool attn_prompt(Layer& l, Gpu& gp, int t0, int ts, int64_t p0, bool all, float scale, Deq& dq) {
        const int H = l.H, KR = g.kv_row(), KL = g.kv_lora, maxk = g.ix_top_k + g.ix_kpool;
        const int64_t P0 = p0 + t0, k1 = P0 + ts;
        const int64_t k0 = l.swa ? std::max<int64_t>(0, P0 - g.n_swa + 1) : 0;
        const int nkr = (int) (k1 - k0);
        if (!l.swa && !all && nkr > dense_max) return false;
        const int window = l.swa ? g.n_swa : all ? (1 << 30) : 0;
        const uint16_t* K = l.kv + (size_t) (l.ring > 0 ? k0 % l.ring : k0) * KR;
        if (kv_q8) {
            if (dq.base < 0 || k0 < dq.base || k1 - dq.base > (int64_t) deq_rows()) dq.base = dq.hi = k0;
            if (k1 > dq.hi) {
                k::kv_deq(l.kv, l.ring, KL, g.rope, (int) dq.hi, (int) (k1 - dq.hi),
                          gp.kbuf + (size_t) (dq.hi - dq.base) * KR, gp.s);
                dq.hi = k1;
            }
            K = gp.kbuf + (size_t) (k0 - dq.base) * KR;
        } else if (l.ring > 0 && k0 % l.ring + nkr > l.ring) {   // (the window wraps round the ring: its rows in order)
            k::kv_gather(l.kv, l.ring, KR, (int) k0, nkr, gp.kbuf, gp.s);
            K = gp.kbuf;
        }
        k::to_f16(gp.qcat + (size_t) t0 * H * KR, KR, ts * H, KR, gp.xh16, KR, gp.s);
        const int ldS = (nkr + 7) / 8 * 8;
        const float one = 1.0f, zero = 0.0f;
        if (cublasGemmEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, nkr, ts * H, KR, &one, K, CUDA_R_16F, KR, gp.xh16,
                         CUDA_R_16F, KR, &zero, gp.sS, CUDA_R_32F, ldS, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            return false;
        if (!l.swa && !all)
            k::dsa_mask(gp.keys + (size_t) t0 * maxk, maxk, gp.knk + t0, ts, (int) k0, nkr, gp.dmask, (size_t) dense_max,
                        gp.s);
        k::attn_softmax(gp.sS, ldS, ts, H, nkr, (int) k0, gp.pos + t0, window, gp.dmask, (size_t) dense_max, scale, l.sinks,
                        gp.sP, ldS, gp.s);
        return cublasGemmEx(gp.blas, CUBLAS_OP_N, CUBLAS_OP_N, KL, ts * H, nkr, &one, K, CUDA_R_16F, KR, gp.sP,
                            CUDA_R_16F, ldS, &zero, gp.o + (size_t) t0 * H * KL, CUDA_R_32F, KL, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
    }

    // A trunk layer's attention half: the hyper-connections' pre-mix, the norm, attn_core, the post-mix
    // (tensor parallel: member tpm's partial output summed over the members - sync v - before the post-mix)
    bool attention(Layer& l, Gpu& gp, int T, int64_t p0, std::string& err, int tpm = -1, unsigned v = 0) {
        const int E = g.n_embd;
        cudaStream_t s = gp.s;
        if (inject) k::set_mix_out(gp.rl);
        k::hc_pre(gp.x, T, g.hc, E, l.hca_fn, l.hca_scale, l.hca_base, g.eps, g.hc_eps, g.hc_mag, gp.xa, gp.post, s,
                  l.attn_norm, g.eps);
        if (inject) {
            k::set_mix_out(nullptr);
            ref_cmp("hc_mixes", l.il, gp, gp.rl, (size_t) T * 2 * g.hc, T);
        }
        dbg("attn_norm", l.il, gp, gp.xa, T, E);
        if (inject) ref_cmp("attn_norm", l.il, gp, gp.xa, (size_t) T * E, T);
        if (!attn_core(l, gp, T, p0, err)) return false;
        if (tpm >= 0) {
            mark(tpm, gp, l.il, 1);
            all_reduce(tpm, gp, T, v);
            mark(tpm, gp, l.il, 2);
        }
        k::hc_post_norm(gp.x, T, g.hc, E, gp.y, gp.post, l.post_attn, g.eps, s, dump || inject ? gp.cur : nullptr);
        if (dump || inject) {   // (the normed output where the read-outs look for it)
            cudaMemcpyAsync(gp.y, gp.cur, sizeof(float) * T * E, cudaMemcpyDeviceToDevice, s);
            dbg("attn_out", l.il, gp, gp.y, T, E);
        }
        dbg("hc_attn_out", l.il, gp, gp.x, T, (size_t) g.hc * E);
        return true;
    }
    // The attention itself (the trunk's layers and the MTP block's): xa [T][E] -> y [T][E] (before its post norm)
    bool attn_core(Layer& l, Gpu& gp, int T, int64_t p0, std::string& err) {
        const int E = g.n_embd, H = l.H, KL = g.kv_lora, R = g.rope, KR = g.kv_row(), nope = g.nope();
        cudaStream_t s = gp.s;
        bool ok = l.ix ? mm_multi(gp, {{&l.q_a, gp.qa, (size_t) g.q_lora}, {&l.kv_a, gp.kvr, (size_t) KR},
                                       {&l.gate, gp.g, (size_t) H * g.v_mla}, {&l.ix_k, gp.ik, (size_t) g.ix_dim}},
                                  gp.xa, E, T)
                       : mm_multi(gp, {{&l.q_a, gp.qa, (size_t) g.q_lora}, {&l.kv_a, gp.kvr, (size_t) KR},
                                       {&l.gate, gp.g, (size_t) H * g.v_mla}}, gp.xa, E, T);
        k::rmsnorm(gp.qa, g.q_lora, T, g.q_lora, l.q_a_norm, g.eps, gp.qr, g.q_lora, s);
        dbg("q_resid", l.il, gp, gp.qr, T, g.q_lora);
        if (inject) ref_cmp("q_resid", l.il, gp, gp.qr, (size_t) T * g.q_lora, T);
        ok = ok && mm(gp, l.q_b, gp.qr, g.q_lora, T, gp.q, (size_t) H * g.k_mla);
        k::kv_store(gp.kvr, T, KL, R, l.kv_a_norm, g.eps, gp.pos, l.base, l.kv, l.ring, s, kv_q8);
        Deq dq;
        // q: q_pe rotated in place, q_nope into the latent space through k_b, both side by side in qcat
        k::rope(gp.q, T, H, g.k_mla, (size_t) H * g.k_mla, nope, R, gp.pos, l.base, s);
        dbg("q_pe", l.il, gp, gp.q, (size_t) T * H, g.k_mla, R, nope);
        ok = ok && mm_heads(gp, l.k_b, gp.q, (size_t) H * g.k_mla, g.k_mla, T, H, gp.qcat, (size_t) H * KR, KR);
        k::copy_cols(gp.q + nope, g.k_mla, gp.qcat + KL, KR, T * H, R, s);
        const float scale = 1.0f / std::sqrt((float) g.k_mla);
        if (l.ix) {
            // the indexer: its query, the window's pooling rows, the groups they complete, every token's picks
            const int D = g.ix_dim, IH = g.ix_heads, KP = g.ix_kpool;
            ok = ok && mm(gp, l.ix_q_b, gp.qr, g.q_lora, T, gp.iq, (size_t) IH * D);
            k::rope(gp.iq, T, IH, D, (size_t) IH * D, D - R, R, gp.pos, l.base, s);
            k::rmsnorm(gp.ik, D, T, D, l.ix_k_norm, g.eps, gp.ik, D, s);
            dbg("indexer_k", l.il, gp, gp.ik, T, D);
            ok = ok && mm_multi(gp, {{&l.ix_val, gp.ivt, (size_t) 2 * D}, {&l.ix_gate, gp.ivt + D, (size_t) 2 * D}}, gp.ik,
                                D, T);
            dbg("indexer_gate", l.il, gp, gp.ivt, T, 2 * D, D, D);
            dbg("indexer_q", l.il, gp, gp.iq, T, (size_t) IH * D);
            k::store_rows(gp.ivt, T, 2 * D, l.ivg, ring, gp.pos, s, kv_q8);
            // the groups whose last member is in the window: [p0 / KP, (last + 1) / KP)
            const int64_t last = p0 + T - 1, g0 = p0 / KP, g1 = (last + 1) / KP;
            if (g1 > g0)
                k::ix_pool(l.ivg, ring, (int) g0, (int) (g1 - g0), KP, D, l.ix_ape, l.ix_pnorm, g.eps, R, l.base,
                           l.pooled, s, kv_q8);
            ok = ok && mm(gp, l.ix_proj, gp.xa, E, T, gp.wt, IH, 1.0f / std::sqrt((float) D * IH));
            dbg("indexer_weights", l.il, gp, gp.wt, T, IH);
            if (!kv_q8) dbg("indexer_pool_k", l.il, gp, l.pooled, (size_t) g1, D);
            const int npool_ld = (int) (o.max_ctx / KP + 1), maxk = g.ix_top_k + KP;
            const int top = g.ix_top_k / KP - 1;
            for (int t0 = 0; t0 < T; t0 += kSub) {   // (the scores of kSub tokens at a time)
                const int ts = std::min(kSub, T - t0);
                const int n_pool = (int) ((p0 + t0 + ts - 1) / KP);
                // the picks are every pool the tokens see until a token sees more than `top` of them
                const bool all = n_pool <= top;
                if (T > W && all && attn_prompt(l, gp, t0, ts, p0, true, scale, dq)) continue;
                k::ix_score(gp.iq + (size_t) t0 * IH * D, gp.wt + (size_t) t0 * IH, ts, IH, D, l.pooled, gp.pos + t0,
                            KP, n_pool, gp.score, npool_ld, s, kv_q8);
                k::ix_select(gp.score, npool_ld, ts, gp.pos + t0, KP, top, gp.keys + (size_t) t0 * maxk, maxk,
                             gp.knk + t0, s);
                if (T > W && attn_prompt(l, gp, t0, ts, p0, false, scale, dq)) continue;
                k::attention(gp.qcat + (size_t) t0 * H * KR, ts, H, KL, R, l.kv, 0, gp.keys + (size_t) t0 * maxk, maxk,
                             nullptr, gp.knk + t0, maxk, scale, l.sinks, gp.ws, gp.o + (size_t) t0 * H * KL, s, kv_q8);
            }
        } else {
            for (int t0 = 0; t0 < T; t0 += kSub) {
                const int ts = std::min(kSub, T - t0);
                if (T > W && attn_prompt(l, gp, t0, ts, p0, false, scale, dq)) continue;
                k::attention(gp.qcat + (size_t) t0 * H * KR, ts, H, KL, R, l.kv, l.ring, nullptr, 0, gp.lo + t0,
                             gp.nk + t0, g.n_swa, scale, l.sinks, gp.ws, gp.o + (size_t) t0 * H * KL, s, kv_q8);
            }
        }
        ok = ok && mm_heads(gp, l.v_b, gp.o, (size_t) H * KL, KL, T, H, gp.ob, (size_t) H * g.v_mla, g.v_mla);
        dbg("attn_kqv", l.il, gp, gp.ob, T, (size_t) H * g.v_mla);
        if (inject) ref_cmp("attn_kqv", l.il, gp, gp.ob, (size_t) T * H * g.v_mla, T);
        k::sigmoid_mul(gp.ob, gp.g, (size_t) T * H * g.v_mla, s);
        dbg("attn_gated", l.il, gp, gp.ob, T, (size_t) H * g.v_mla);
        ok = ok && mm(gp, l.wo, gp.ob, (size_t) H * g.v_mla, T, gp.y, E);
        if (!ok) err = "layer " + std::to_string(l.il) + ": a GEMV the kernels do not take";
        return ok;
    }

    // A window's routed experts on the token path (T <= W): gp.lat's int8 rows through ds4_iq's plan (one token) or
    // grouped (a few) form, each token's weighted sum -> out [T][n_lat]; slot: the GPU's expert set (its weights in
    // gp.rw zeroed for the others)
    bool routed_token(Layer& l, Gpu& gp, int T, const int* slot, float* out) {
        const int lat = g.n_lat, ffe = l.ffe, nu = g.n_used;
        cudaStream_t s = gp.s;
        bool ok = true;
        k::q8_rows(gp.lat, lat, T, lat, gp.xq, gp.xd, s);
        iq::Args a;
        a.tr[0] = l.tr[0];
        a.tr[1] = l.tr[1];
        a.tr[2] = l.tr[2];
        a.n_embd = lat;
        a.n_ff = ffe;
        a.limit = g.limit[(size_t) l.il];
        a.h = gp.eh;
        a.part = gp.part;
        a.tg = l.tg;
        a.td = l.td;
        a.act = iq::kActSwiglu;
        int groups, ne_max;
        if (T == 1) {
            k::plan_experts(gp.ids, nu, l.experts, l.blob, gp.plan, s, slot, gp.rw);
            a.plan_ptr = gp.plan;
            a.xq = gp.xq;
            a.xd = gp.xd;
            groups = nu;
            ne_max = 1;
        } else {
            k::dup_rows_q8(gp.xq, gp.xd, T, lat, gp.x2q, gp.x2d, s);
            k::group_experts(gp.ids, T, nu, g.n_expert, l.experts, l.blob, gp.grp_ptr, gp.grp_start, gp.grp_ent,
                             gp.n_groups, s, slot, gp.rw);
            a.grp_ptr = gp.grp_ptr;
            a.grp_start = gp.grp_start;
            a.grp_ent = gp.grp_ent;
            a.n_groups = gp.n_groups;
            a.xq = gp.x2q;
            a.xd = gp.x2d;
            groups = std::min(nu * T, g.n_expert);
            ne_max = T;
        }
        ok = ok && iq::gate_up(a, groups, ne_max, s);
        // h's rows: every entry's (the plan form: nu of them; the grouped form: t 16 + j)
        const int rows = T == 1 ? nu : 16 * T;
        k::q8_rows(gp.eh, ffe, rows, ffe, gp.ehq, gp.ehd, s);
        a.hq = gp.ehq;
        a.hd = gp.ehd;
        ok = ok && iq::down(a, groups, ne_max, s);
        k::moe_sum(gp.part, T, nu, lat, gp.rw, out, s);
        return ok;
    }

    // A chunk's routed experts (more than W tokens): the entries sorted by expert, ds4_iq's sets over every expert's
    // rows (gate / up -> F16 h, down -> F32 rows), then each token's weighted sum -> latout
    // (slot: each expert's slot in wbase - null: every expert, wbase the layer's; n_slots of them; accumulate: latout +=)
    // the routes in h_ids (T tokens of layer il) into the profile's counts
    void count_routes(int il, int T) {
        if (prof_cnt.empty()) prof_cnt.assign((size_t) g.n_layer * g.n_expert, 0);
        if (il >= g.n_layer) return;
        uint32_t* c = prof_cnt.data() + (size_t) il * g.n_expert;
        for (int i = 0; i < T * g.n_used; ++i) ++c[h_ids[i]];
    }
    bool experts_sets(Layer& l, Gpu& gp, int T, std::string& err, const int* slot, const uint8_t* wbase = nullptr,
                      int n_slots = -1, bool accumulate = false) {
        const int lat = g.n_lat, nu = g.n_used, ne = slot ? (n_slots >= 0 ? n_slots : l.n_res) : g.n_expert;
        cudaStream_t s = gp.s;
        k::moe_sort(gp.ids, T, nu, g.n_expert, gp.bounds, gp.stok, gp.srow, s, slot, nullptr, ne);
        if (ne == 0) {   // (no expert of the set here: the zeros, or nothing to add)
            if (!accumulate) k::moe_combine(gp.sy, gp.srow, gp.rw, T, nu, lat, gp.latout, s);
            return true;
        }
        std::vector<int> b((size_t) ne + 1);
        CK(cudaMemcpyAsync(b.data(), gp.bounds, sizeof(int) * (ne + 1), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        int max_rows = 0;
        for (int e = 0; e < ne; ++e) max_rows = std::max(max_rows, b[(size_t) e + 1] - b[(size_t) e]);
        iq::SetArgs a;
        a.wbase = wbase ? wbase : l.experts;
        a.stride = l.blob;
        a.tr[0] = l.tr[0];
        a.tr[1] = l.tr[1];
        a.tr[2] = l.tr[2];
        a.bounds = gp.bounds;
        a.row0 = 0;
        a.max_rows = max_rows;
        a.n_exp = ne;
        a.n_embd = lat;
        a.n_ff = l.ffe;
        a.limit = g.limit[(size_t) l.il];
        a.xf = gp.lat;
        a.idx = gp.stok;
        a.ldx = lat;
        a.h = gp.sh16;
        a.ldo = l.ffe;
        a.tg = l.tg;
        a.td = l.td;
        a.act = iq::kActSwiglu;
        if (!iq::set_gate_up(a, s)) {
            err = "layer " + std::to_string(l.il) + ": the experts' gate / up sets did not run";
            return false;
        }
        iq::SetArgs d = a;
        d.xf = nullptr;
        d.idx = nullptr;
        d.xh = gp.sh16;
        d.ldx = l.ffe;
        d.h = nullptr;
        d.y = gp.sy;
        d.ldo = lat;
        if (!iq::set_down(d, s)) {
            err = "layer " + std::to_string(l.il) + ": the experts' down sets did not run";
            return false;
        }
        k::moe_combine(gp.sy, gp.srow, gp.rw, T, nu, lat, gp.latout, s, accumulate);
        return true;
    }

    // A prompt chunk's experts that are not in VRAM, streamed: the ones the chunk uses, in passes of kStage, each pass
    // copied from the model's file into the pinned bounce (the CPU lane's threads) and on to the GPU's staging slots in
    // one copy, its sets run there and added to latout
    bool experts_streamed(Layer& l, Gpu& gp, int T, std::string& err) {
        const int nu = g.n_used, ne = g.n_expert;
        count_routes(l.il, T);
        cudaStream_t s = gp.s;
        std::vector<char> used((size_t) ne, 0);
        for (int i = 0; i < T * nu; ++i)
            if (l.slot_of[(size_t) h_ids[i]] < 0) used[(size_t) h_ids[i]] = 1;
        std::vector<int> u;
        for (int e = 0; e < ne; ++e)
            if (used[(size_t) e]) u.push_back(e);
        const CpuExperts& cx = l.cex;
        const size_t gb = cx.g_stride, db = cx.d_stride;
        for (size_t p0 = 0; p0 < u.size(); p0 += (size_t) kStage) {
            const int np = (int) std::min(u.size() - p0, (size_t) kStage);
            const double tc = now_s();
            lane->parallel_for(np * 3, [&](int it) {
                const int i = it / 3, role = it % 3, e = u[p0 + (size_t) i];
                uint8_t* dst = h_stage + (size_t) i * l.blob + l.tr[role];
                const uint8_t* src = role == 0 ? cx.gate + (size_t) e * gb : role == 1 ? cx.up + (size_t) e * gb
                                                                                       : cx.down + (size_t) e * db;
                std::memcpy(dst, src, role == 2 ? db : gb);
            });
            st.stream_ms += (now_s() - tc) * 1e3;
            CK(cudaMemcpyAsync(gp.stage, h_stage, (size_t) np * l.blob, cudaMemcpyHostToDevice, s));
            for (int e = 0; e < ne; ++e) h_pmap[e] = -1;
            for (int i = 0; i < np; ++i) h_pmap[u[p0 + (size_t) i]] = i;
            CK(cudaMemcpyAsync(gp.pmap, h_pmap, sizeof(int) * ne, cudaMemcpyHostToDevice, s));
            if (!experts_sets(l, gp, T, err, gp.pmap, gp.stage, np, true)) return false;
            CK(cudaStreamSynchronize(s));   // (the bounce and the map are the next pass's)
            st.streamed += np;
        }
        return true;
    }

    // A trunk layer's FFN half: the pre-mix, the norm, ffn_core, the post-mix
    bool ffn(Layer& l, Gpu& gp, int T, std::string& err, int tpm = -1, unsigned v = 0) {
        const int E = g.n_embd;
        cudaStream_t s = gp.s;
        k::hc_pre(gp.x, T, g.hc, E, l.hcf_fn, l.hcf_scale, l.hcf_base, g.eps, g.hc_eps, g.hc_mag, gp.xf, gp.post, s,
                  l.ffn_norm, g.eps);
        dbg("ffn_norm", l.il, gp, gp.xf, T, E);
        if (inject) ref_cmp("ffn_norm", l.il, gp, gp.xf, (size_t) T * E, T);
        if (!ffn_core(l, gp, T, err, v)) return false;   // (tensor parallel: its sum is ffn_core's)
        (void) tpm;
        k::hc_post_norm(gp.x, T, g.hc, E, gp.y, gp.post, l.post_ffn, g.eps, s, dump || inject ? gp.cur : nullptr);
        if (dump || inject) {
            cudaMemcpyAsync(gp.y, gp.cur, sizeof(float) * T * E, cudaMemcpyDeviceToDevice, s);
            dbg("ffn_out", l.il, gp, gp.y, T, E);
        }
        dbg("l_out", l.il, gp, gp.x, T, (size_t) g.hc * E);
        return true;
    }
    // The FFN itself (the dense one, or the router, the routed experts and the shared expert): xf -> y (before its
    // post norm)
    // (tensor parallel, l a member's slice: its partial summed over the members - sync v - inside: the routed experts'
    //  sum in the latent space beside the shared expert's, so ffn_routed_up projects the whole sum)
    bool ffn_core(Layer& l, Gpu& gp, int T, std::string& err, unsigned v = 0) {
        const int E = g.n_embd;
        cudaStream_t s = gp.s;
        bool ok = true;
        if (l.dense) {
            const int ff = l.ffn_gate.n;   // (a member's slice: its share)
            ok = mm_multi(gp, {{&l.ffn_gate, gp.gu, (size_t) ff}, {&l.ffn_up, gp.gu + (size_t) T * ff, (size_t) ff}}, gp.xf, E,
                          T);
            k::swiglu(gp.gu, gp.gu + (size_t) T * ff, ff, T, ff, gp.hd, ff, s);
            ok = ok && mm(gp, l.ffn_down, gp.hd, ff, T, gp.y, E);
            if (ok && l.tpm >= 0) {
                mark(l.tpm, gp, l.il, 4);
                all_reduce(l.tpm, gp, gp.y, (size_t) T * E, v);
                mark(l.tpm, gp, l.il, 5);
            }
        } else {
            const int lat = g.n_lat, ffe = l.ffe, nu = g.n_used;
            const bool lead = l.tpm <= 0;   // (the stats and the profile: one member's count)
            ok = mm(gp, l.router, gp.xf, E, T, gp.rl, g.n_expert);
            dbg("ffn_moe_logits", l.il, gp, gp.rl, T, g.n_expert);
            k::router(gp.rl, T, g.n_expert, l.router_b, nu, g.w_norm, g.w_scale, gp.ids, gp.rw, s);
            if (dump) {
                cudaStreamSynchronize(s);
                std::vector<int> hid((size_t) T * nu);
                cudaMemcpy(hid.data(), gp.ids, hid.size() * 4, cudaMemcpyDeviceToHost);
                long sum = 0;
                for (int v : hid) sum += v;
                std::fprintf(stderr, "DUMP ffn_moe_topk-%-10d sum = %ld\n", l.il, sum);
            }
            const int sf = l.sh_gate.n;
            // the experts not in VRAM (the CPU lane's): the picks, their weights and the latent input to the host
            // first; the GPU's are queued without them (their weights zeroed there), the lane's sum added after.  On
            // the token path the latent input goes first (its own GEMV) and the shared expert's after the copies, so
            // the GPU computes it while the host waits on the copies' event and the lane works
            const bool lane_on = l.partial();
            if (!lane_on && lead && !o.profile_out.empty()) {   // (the profile: this layer's routes as well)
                CK(cudaMemcpyAsync(h_ids, gp.ids, sizeof(int) * T * nu, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                count_routes(l.il, T);
            }
            // (tensor parallel: a member hands the helpers the routing - member il % N, so the copy goes round them)
            const bool hand = l.tpm >= 0 && !hel.empty() && l.tpm == (hand_rot ? l.il % tp : 0);
            if ((lane_on || hand) && T <= W) ok = ok && mm(gp, l.rdown, gp.xf, E, T, gp.lat, lat);
            else
                ok = ok && mm_multi(gp, {{&l.rdown, gp.lat, (size_t) lat}, {&l.sh_gate, gp.gu, (size_t) sf},
                                         {&l.sh_up, gp.gu + (size_t) T * sf, (size_t) sf}}, gp.xf, E, T);
            if (hand) {   // (the helpers' input - the routing, the latent rows - then the shared expert's gate / up)
                float* rt = gp.half ? gp.route2 : gp.route;   // (a staggered chunk's halves: a buffer each)
                float* trt = gp.half ? tp_route2 : tp_route;
                unsigned* tfr = gp.half ? tp_froute2 : tp_froute;
                k::pack3(gp.lat, (size_t) T * lat, gp.ids, (size_t) T * 16, gp.rw, (size_t) T * 16, rt, s);
                if (gp.s2 && !hand_inline) {   // (copied out on the side stream: member 0's own work goes on meanwhile)
                    CK(cudaEventRecord(gp.ev2, s));
                    CK(cudaStreamWaitEvent(gp.s2, gp.ev2, 0));
                    k::publish(rt, trt, (size_t) T * (lat + 32), gp.cnt2, tfr, v, gp.s2);
                } else {
                    k::publish(rt, trt, (size_t) T * (lat + 32), gp.cnt2, tfr, v, s);
                }
                if (T <= W)
                    ok = ok && mm_multi(gp, {{&l.sh_gate, gp.gu, (size_t) sf}, {&l.sh_up, gp.gu + (size_t) T * sf,
                                                                            (size_t) sf}}, gp.xf, E, T);
            }
            if (lane_on) {
                CK(cudaMemcpyAsync(h_ids, gp.ids, sizeof(int) * T * nu, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpyAsync(h_w, gp.rw, sizeof(float) * T * nu, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpyAsync(h_lat, gp.lat, sizeof(float) * T * lat, cudaMemcpyDeviceToHost, s));
                CK(cudaEventRecord(gp.ev, s));
                if (T <= W)
                    ok = ok && mm_multi(gp, {{&l.sh_gate, gp.gu, (size_t) sf}, {&l.sh_up, gp.gu + (size_t) T * sf,
                                                                                (size_t) sf}}, gp.xf, E, T);
                const double tw = now_s();
                // (a token's copies are there within a millisecond or two: polled that long - a thread that sleeps on
                //  the event comes back late on some hosts, and the lane's threads wait for it - then, a prompt
                //  chunk's long wait, asleep on the event.  HY4_EV_POLL_US: the polled time, 0 = asleep at once)
                for (cudaError_t q;;) {
                    if (ev_poll_s <= 0.0 || (q = cudaEventQuery(gp.ev)) == cudaErrorNotReady) {
                        if (ev_poll_s > 0.0 && now_s() - tw < ev_poll_s) continue;
                        CK(cudaEventSynchronize(gp.ev));
                        break;
                    }
                    CK(q);
                    break;
                }
                if (prof) prof_acc[T > W ? " lane: wait for the routes (prompt)" : " lane: wait for the routes (token)"] += now_s() - tw;
            }
            const int* slot = lane_on || l.tpm >= 0 ? l.slot_d : nullptr;
            bool streamed = false;
            if (T > W) {
                ok = ok && experts_sets(l, gp, T, err, slot);
                // a prompt chunk's experts not in VRAM: streamed to the GPU, or the CPU lane - whichever the measured
                // rates say is faster for this many experts and entries (a PC's PCIe against its CPU)
                if (lane_on) {
                    int n_ent = 0;
                    std::vector<char> used((size_t) g.n_expert, 0);
                    for (int i = 0; i < T * nu; ++i)
                        if (l.slot_of[(size_t) h_ids[i]] < 0) {
                            ++n_ent;
                            used[(size_t) h_ids[i]] = 1;
                        }
                    int n_u = 0;
                    for (char c : used) n_u += c;
                    const double est_cpu = n_ent * cpu_us_et * 1e-3;   // (ms)
                    const double est_stream = (double) n_u * l.blob * (1.0 / (gp.h2d_gbs * 1e6) + 1.0 / (copy_gbs * 1e6));
                    if (est_stream < est_cpu) {
                        const double ts = now_s();
                        const double c0 = st.stream_ms;
                        ok = ok && experts_streamed(l, gp, T, err);
                        streamed = true;
                        const double copy_ms = st.stream_ms - c0;
                        if (copy_ms > 0.1) copy_gbs = 0.8 * copy_gbs + 0.2 * ((double) n_u * l.blob / 1e6 / copy_ms);
                        (void) ts;
                    }
                }
            } else {
                ok = ok && routed_token(l, gp, T, slot, gp.latout);
            }
            if (lane_on && !streamed) {   // (the GPU's experts run meanwhile)
                count_routes(l.il, T);
                lane_entries.clear();
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < nu; ++j) {
                        const int e = h_ids[t * nu + j];
                        if (l.slot_of[(size_t) e] < 0) lane_entries.push_back({t, e, h_w[t * nu + j]});
                    }
                const double tc = now_s();
                cpu_experts(*lane, l.cex, lane_entries, h_lat, T, h_cpu);
                const double ms = (now_s() - tc) * 1e3;
                st.cpu_ms += ms;
                if (prof) prof_acc[T > W ? " lane: the CPU's experts (prompt)" : " lane: the CPU's experts (token)"] += ms * 1e-3;
                if (T > W && !lane_entries.empty())   // (the prompt chunks' rate: what the route choice weighs)
                    cpu_us_et = 0.8 * cpu_us_et + 0.2 * (ms * 1e3 / (double) lane_entries.size());
                st.cpu_experts += (int64_t) lane_entries.size();
                CK(cudaMemcpyAsync(gp.cpo, h_cpu, sizeof(float) * T * lat, cudaMemcpyHostToDevice, s));
                k::add(gp.latout, gp.cpo, (size_t) T * lat, s);
            }
            if (lead) st.experts += (int64_t) T * nu;
            dbg("ffn_moe_out", l.il, gp, gp.latout, T, lat);
            if (inject) ref_cmp("ffn_moe_out", l.il, gp, gp.latout, (size_t) T * lat, T);
            if (l.tpm >= 0 && model_sum) {   // (HY4_TP_MODEL_SUM=1: each member's own up-projection, summed after)
                ok = ok && mm(gp, l.rup, gp.latout, lat, T, gp.y, E);
                k::swiglu(gp.gu, gp.gu + (size_t) T * sf, sf, T, sf, gp.hd, sf, s);
                ok = ok && mm(gp, l.sh_down, gp.hd, sf, T, gp.cur, E);
                k::add(gp.y, gp.cur, (size_t) T * E, s);
                mark(l.tpm, gp, l.il, 4);
                if (ok) all_reduce(l.tpm, gp, gp.y, (size_t) T * E, v);
                mark(l.tpm, gp, l.il, 5);
                if (!ok) err = "layer " + std::to_string(l.il) + ": an FFN kernel did not take its shapes";
                return ok;
            }
            if (l.tpm >= 0) {   // (tensor parallel: [the experts' latent sum | the shared expert's] summed, then up)
                float* sh = gp.latout + (size_t) T * lat;
                k::swiglu(gp.gu, gp.gu + (size_t) T * sf, sf, T, sf, gp.hd, sf, s);
                ok = ok && mm(gp, l.sh_down, gp.hd, sf, T, sh, E);
                mark(l.tpm, gp, l.il, 4);
                if (ok) all_reduce(l.tpm, gp, gp.latout, (size_t) T * (lat + E), v, false);
                mark(l.tpm, gp, l.il, 5);
                ok = ok && mm(gp, l.rup, gp.latout, lat, T, gp.y, E);
                k::add(gp.y, sh, (size_t) T * E, s);
                if (!ok) err = "layer " + std::to_string(l.il) + ": an FFN kernel did not take its shapes";
                return ok;
            }
            ok = ok && mm(gp, l.rup, gp.latout, lat, T, gp.y, E);
            dbg("ffn_moe_up", l.il, gp, gp.y, T, E);
            // the shared expert
            k::swiglu(gp.gu, gp.gu + (size_t) T * sf, sf, T, sf, gp.hd, sf, s);
            ok = ok && mm(gp, l.sh_down, gp.hd, sf, T, gp.cur, E);
            dbg("ffn_shexp", l.il, gp, gp.cur, T, E);
            if (inject) ref_cmp("ffn_shexp", l.il, gp, gp.cur, (size_t) T * E, T);
            k::add(gp.y, gp.cur, (size_t) T * E, s);
        }
        if (!ok) err = "layer " + std::to_string(l.il) + ": an FFN kernel did not take its shapes";
        return ok;
    }

    // the Engram rows of the window's tokens for module e -> h_eg's part e; valid -> h_valid
    void engram_gather(int e, int T, int64_t p0) {
        for (int t = 0; t < T; ++t) engram_row(e, t, p0);
    }
    // ... a token's (t of the window at p0): its n-grams' hashed rows of the table - a random page each, so a window's
    // are looked up on several threads (eg_crew)
    void engram_row(int e, int t, int64_t p0) {
        const Engram& eg = pk.eg;
        const int n = eg.ngram, cols = eg.cols(), hd = eg.head_dim;
        const uint64_t* mult = eg.mult.data() + (size_t) e * n;
        const uint64_t* primes = eg.primes.data() + (size_t) e * cols;
        const uint64_t* offs = eg.offsets.data() + (size_t) e * cols;
        uint8_t* dst = h_eg + (size_t) e * Tc * cols * hd;
        auto cid = [&](int t) -> int64_t {
            return t < 0 || t >= eg.vocab ? -1 : (int64_t) eg.token_map[(size_t) t];
        };
        {
            const int64_t p = p0 + t;
            uint64_t toks[8];
            const int64_t c0 = cid(tokens[(size_t) p]);
            const bool dead = c0 < 0;
            toks[0] = (uint64_t) (dead ? eg.pad : c0);
            for (int sft = 1; sft < n; ++sft) {
                const int64_t c = p - sft < 0 ? -1 : cid(tokens[(size_t) (p - sft)]);
                toks[sft] = (uint64_t) (c < 0 ? eg.pad : c);
            }
            uint64_t rolling = toks[0] * mult[0];
            for (int oo = 1; oo < n; ++oo) {
                rolling ^= toks[oo] * mult[oo];
                for (int h = 0; h < eg.heads; ++h) {
                    const int col = (oo - 1) * eg.heads + h;
                    const uint64_t row = rolling % primes[col] + offs[col];
                    std::memcpy(dst + ((size_t) t * cols + col) * hd, eg_table + row * (uint64_t) hd, (size_t) hd);
                    float sc = 1.0f;
                    if (eg_scale != nullptr) {
                        uint16_t sb;
                        std::memcpy(&sb, eg_scale + row * 2, 2);
                        bf16_to_f32(&sb, 1, &sc);
                        h_egs[((size_t) e * Tc + (size_t) t) * cols + col] = sc;
                    }
                    if (eg_dump != nullptr) {   // (int64 module, position, column, row; float scale; the row's bytes)
                        std::lock_guard<std::mutex> lk(eg_dump_mu);
                        const int64_t rec[4] = {e, p, col, (int64_t) row};
                        std::fwrite(rec, 8, 4, eg_dump);
                        std::fwrite(&sc, 4, 1, eg_dump);
                        std::fwrite(eg_table + row * (uint64_t) hd, 1, (size_t) hd, eg_dump);
                    }
                }
            }
            if (e == 0) h_valid[t] = dead ? 0.0f : 1.0f;
        }
    }
    std::unique_ptr<Crew> eg_crew;   // (the Engram lookups' threads)

    // (r0: the rows' start in the window - a staggered half)
    bool engram(Layer& l, Gpu& gp, int T, std::string& err, int r0 = 0) {
        const Engram& eg = pk.eg;
        const int cols = eg.cols(), hd = eg.head_dim, E = g.n_embd;
        cudaStream_t s = gp.s;
        CK(cudaMemcpyAsync(gp.eg_rows, h_eg + ((size_t) l.eg * Tc + (size_t) r0) * cols * hd, (size_t) T * cols * hd,
                           cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(gp.eg_valid, h_valid + r0, sizeof(float) * T, cudaMemcpyHostToDevice, s));
        const float eps = eg_eps;
        if (eg_scale != nullptr)
            CK(cudaMemcpyAsync(gp.eg_sc, h_egs + ((size_t) l.eg * Tc + (size_t) r0) * cols, sizeof(float) * T * cols,
                               cudaMemcpyHostToDevice, s));
        k::engram_rows(gp.eg_rows, gp.eg_valid, T, cols, hd, l.eg_lookup, eps, gp.eg_r, s, eg_scale ? gp.eg_sc : nullptr);
        dbg("engram_rows_normed", l.il, gp, gp.eg_r, T, (size_t) cols * hd);
        if (!mm(gp, l.eg_kv, gp.eg_r, (size_t) cols * hd, T, gp.eg_kvo, (size_t) (g.hc + 1) * E)) {
            err = "engram_kv: a GEMV the kernels do not take";
            return false;
        }
        dbg("engram_kv(no bias)", l.il, gp, gp.eg_kvo, T, (size_t) (g.hc + 1) * E);
        k::engram_mix(gp.x, gp.eg_kvo, l.eg_kv_b, T, g.hc, E, l.eg_gate, l.eg_gamma, l.eg_hcs, gp.eg_valid, eps, s);
        dbg("engram_out", l.il, gp, gp.x, T, (size_t) g.hc * E);
        return true;
    }

    // The MTP block over T tokens at positions p0 .. (on the head GPU, gp.pos holding them): each token's embedding
    // beside the state a position before it (mtp_hin's rows) -> eh_proj -> its attention and FFN (plain residuals) ->
    // shared_head_norm -> mtp_out; want > 0: the last `want` rows' logits in hg.logits.  Its caches take the positions.
    bool mtp_forward(const int* ids, int T, int64_t p0, int want, std::string& err, const Mat* head = nullptr,
                     Gpu* on = nullptr) {
        Gpu& hg = on ? *on : G[(size_t) head_gpu];
        const int E = g.n_embd;
        cudaStream_t s = hg.s;
        if (emb_pending) {   // (h_emb free: the queued pass's copy of it done - once, after such a pass)
            CK(cudaEventSynchronize(ev_emb));
            emb_pending = false;
        }
        for (int t = 0; t < T; ++t) {   // (the embeddings, host-dequantised as the trunk's)
            const uint8_t* row = tok_embd + (size_t) ids[t] * tok_ld;
            tok_row(tok_type, row, E, h_emb + (size_t) t * E);
        }
        if (!imgs.empty()) image_rows(p0, T, h_emb);   // (a picture's image tokens: its rows, as the trunk's)
        CK(cudaMemcpyAsync(mtp_cur, h_emb, sizeof(float) * T * E, cudaMemcpyHostToDevice, s));
        if (on == &hm) {
            CK(cudaEventRecord(ev_emb, s));
            emb_pending = true;
        }
        k::rmsnorm(mtp_cur, E, T, E, M.enorm, g.eps, mtp_in, 2 * (size_t) E, s);
        k::rmsnorm(mtp_hin, E, T, E, M.hnorm, g.eps, mtp_in + E, 2 * (size_t) E, s);
        if (!mm(hg, M.eh_proj, mtp_in, 2 * (size_t) E, T, mtp_cur, E)) {
            err = "MTP eh_proj: a GEMV the kernels do not take";
            return false;
        }
        k::rmsnorm(mtp_cur, E, T, E, M.attn_norm, g.eps, hg.xa, E, s);
        if (!attn_core(M, hg, T, p0, err)) return false;
        k::rmsnorm(hg.y, E, T, E, M.post_attn, g.eps, hg.y, E, s);
        k::add(mtp_cur, hg.y, (size_t) T * E, s);
        k::rmsnorm(mtp_cur, E, T, E, M.ffn_norm, g.eps, hg.xf, E, s);
        if (!ffn_core(M, hg, T, err)) return false;
        k::rmsnorm(hg.y, E, T, E, M.post_ffn, g.eps, hg.y, E, s);
        k::add(mtp_cur, hg.y, (size_t) T * E, s);
        k::rmsnorm(mtp_cur, E, T, E, M.head_norm, g.eps, mtp_out, E, s);
        if (want > 0 && !mm(hg, head ? *head : out, mtp_out + (size_t) (T - want) * E, E, want, hg.logits, g.n_vocab)) {
            err = "MTP head: a GEMV the kernels do not take";
            return false;
        }
        return true;
    }

    // Up to n drafts after `tok` at position pos (the caches hold every position before it): each MTP step's top token
    // and its probability, the first from the trunk's state at pos - 1 (the ring's), the next from the step before's
    // state; stops after a draft below `floor`.  The MTP cache's rows at pos .. are rewritten by the next forward.
    // HY4_MTP_EAGER=1: every window's MTP rows at its end, as before (the drafts' first step alone)
    const bool mtp_eager = std::getenv("HY4_MTP_EAGER") != nullptr;
    // the MTP rows [mtp_done, upto) from the ring's states (a prompt chunk after token windows that left theirs)
    bool mtp_catch_up(int64_t upto, std::string& err) {
        Gpu& hg = G[(size_t) head_gpu];
        mtp_settle();
        int64_t c0 = mtp_done;
        if (c0 < upto - (int64_t) ring + 1) c0 = std::max<int64_t>(0, upto - (int64_t) ring + 1);
        const size_t rb = sizeof(float) * g.n_embd;
        std::vector<int> pos_save(h_pos, h_pos + 3 * Tc);
        for (int64_t a = c0; a < upto; a += Tc) {
            const int nT = (int) std::min<int64_t>(upto - a, Tc);
            for (int t = 0; t < nT; ++t) h_pos[t] = (int) (a + t);
            CK(cudaMemcpyAsync(hg.pos, h_pos, sizeof(int) * nT, cudaMemcpyHostToDevice, hg.s));
            for (int t = 0; t < nT; ) {
                const int64_t q = a + t;
                if (q == 0) {
                    CK(cudaMemsetAsync(mtp_hin, 0, rb, hg.s));
                    ++t;
                    continue;
                }
                const int64_t r0 = (q - 1) % ring;
                const int run = (int) std::min<int64_t>(nT - t, ring - r0);
                CK(cudaMemcpyAsync(mtp_hin + (size_t) t * g.n_embd, hring + (size_t) r0 * g.n_embd, rb * run,
                                   cudaMemcpyDeviceToDevice, hg.s));
                t += run;
            }
            if (!mtp_forward(tokens.data() + a, nT, a, 0, err)) return false;
            CK(cudaStreamSynchronize(hg.s));   // (h_pos is the window's again below)
        }
        std::copy(pos_save.begin(), pos_save.end(), h_pos);
        CK(cudaMemcpyAsync(hg.pos, h_pos, sizeof(int) * 3 * Tc, cudaMemcpyHostToDevice, hg.s));
        mtp_done = upto;
        return true;
    }
    bool draft(int tok, int64_t pos, int n, float floor, std::vector<int>& d, std::vector<float>& conf,
               std::string& err) {
        d.clear();
        conf.clear();
        if (!M.mtp || n <= 0) return true;
        if (pos + n >= o.max_ctx) n = (int) std::max<int64_t>(0, o.max_ctx - pos - 1);
        Gpu& hg = G[(size_t) head_gpu];
        CK(cudaSetDevice(hg.dev));
        mtp_settle();
        if (!mtp_err.empty()) {
            err = "the MTP block's pass: " + mtp_err;
            return false;
        }
        const size_t rb = sizeof(float) * g.n_embd;
        // the first step: the rows the cache lacks (the positions since mtp_done, the trunk's states before them - those
        // still in the ring) and the new token's, one MTP forward (chunks of Tc), the last row's logits the draft's
        int64_t c0 = std::min(mtp_done, pos);
        if (c0 < pos - (int64_t) ring + 1) c0 = std::max<int64_t>(0, pos - (int64_t) ring + 1);   // (older: gone)
        std::vector<int> rows;
        for (int64_t q = c0; q < pos; ++q) rows.push_back(tokens[(size_t) q]);
        rows.push_back(tok);
        for (size_t a = 0; a < rows.size(); a += (size_t) Tc) {
            const int nT = (int) std::min(rows.size() - a, (size_t) Tc);
            const int64_t q0 = c0 + (int64_t) a;
            for (int t = 0; t < nT; ++t) h_pos[t] = (int) (q0 + t);
            CK(cudaMemcpyAsync(hg.pos, h_pos, sizeof(int) * nT, cudaMemcpyHostToDevice, hg.s));
            for (int t = 0; t < nT; ) {   // (the trunk's state a position before each row: the ring's, in runs)
                const int64_t q = q0 + t;
                if (q == 0) {
                    CK(cudaMemsetAsync(mtp_hin, 0, rb, hg.s));
                    ++t;
                    continue;
                }
                const int64_t r0 = (q - 1) % ring;
                const int run = (int) std::min<int64_t>(nT - t, ring - r0);
                CK(cudaMemcpyAsync(mtp_hin + (size_t) t * g.n_embd, hring + (size_t) r0 * g.n_embd, rb * run,
                                   cudaMemcpyDeviceToDevice, hg.s));
                t += run;
            }
            const bool last = a + (size_t) nT == rows.size();
            if (!mtp_forward(rows.data() + a, nT, q0, last ? 1 : 0, err, &out_draft)) return false;
        }
        mtp_done = pos + 1;
        int cur = tok;
        for (int i = 0; i < n; ++i) {
            if (i > 0) {
                h_pos[0] = (int) (pos + i);
                CK(cudaMemcpyAsync(hg.pos, h_pos, sizeof(int), cudaMemcpyHostToDevice, hg.s));
                if (!mtp_forward(&cur, 1, pos + i, 1, err, &out_draft)) return false;
            }
            k::top1(hg.logits, g.n_vocab, mtp_top, hg.s);
            float r[2];
            CK(cudaMemcpyAsync(r, mtp_top, sizeof r, cudaMemcpyDeviceToHost, hg.s));
            CK(cudaStreamSynchronize(hg.s));
            int id;
            std::memcpy(&id, &r[0], 4);
            d.push_back(id);
            conf.push_back(r[1]);
            if (r[1] < floor || i + 1 == n) break;
            cur = id;
            const int last_row = i == 0 ? (int) ((rows.size() - 1) % (size_t) Tc) : 0;   // (the first step's last row)
            CK(cudaMemcpyAsync(mtp_hin, mtp_out + (size_t) last_row * g.n_embd, rb, cudaMemcpyDeviceToDevice, hg.s));
        }
        if (k::errors()) {
            err = "a kernel failed (above)";
            return false;
        }
        return true;
    }

    bool window(const int* ids, int T, int64_t p0, int want, float* logits_out, std::string& err) {
        const int E = g.n_embd, hc = g.hc;
        win_t0 = now_s();
        cur_p0 = p0;
        // the embeddings, every stream a copy
        for (int t = 0; t < T; ++t) {
            const int id = ids[t];
            if (id < 0 || id >= g.n_vocab) {
                err = "token id " + std::to_string(id) + " outside the vocabulary";
                return false;
            }
            const uint8_t* row = tok_embd + (size_t) id * tok_ld;
            tok_row(tok_type, row, E, h_x + (size_t) t * E);   // ([T][E]: the GPUs copy each row into the hc streams)
        }
        if (!imgs.empty()) image_rows(p0, T, h_x);   // (a picture's image tokens: its rows)
        const double te = now_s();
        const int n_look = pk.eg.modules * T;
        if (n_look >= 8 && !std::getenv("HY4_EG_SERIAL")) {   // (a window's rows on the lookup threads)
            if (!eg_crew) eg_crew = std::make_unique<Crew>(8);
            const int nt = eg_crew->size();
            eg_crew->start([&](int w) {
                for (int i = w; i < n_look; i += nt) engram_row(i / T, i % T, p0);
            });
            eg_crew->wait();
        } else {
            for (int e = 0; e < pk.eg.modules; ++e) engram_gather(e, T, p0);
        }
        eg_last_s = now_s() - te;
        st.engram_ms += eg_last_s * 1e3;
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            h_pos[t] = (int) p;
            h_pos[Tc + t] = (int) std::max<int64_t>(0, p - g.n_swa + 1);
            h_pos[2 * Tc + t] = (int) (p - h_pos[Tc + t] + 1);
        }
        if (tp > 1) return window_tp(ids, T, p0, want, logits_out, err);
        int cur = -1;
        for (Layer& l : L) {
            Gpu& gp = G[(size_t) l.gpu];
            if (l.gpu != cur) {
                CK(cudaSetDevice(gp.dev));
                CK(cudaMemcpyAsync(gp.pos, h_pos, sizeof(int) * 3 * Tc, cudaMemcpyHostToDevice, gp.s));
                if (cur < 0) {
                    CK(cudaMemcpyAsync(gp.cur, h_x, sizeof(float) * T * E, cudaMemcpyHostToDevice, gp.s));
                    k::expand_streams(gp.cur, T, hc, E, gp.x, gp.s);
                } else {
                    Gpu& pg = G[(size_t) cur];
                    CK(cudaStreamSynchronize(pg.s));
                    CK(cudaMemcpyPeerAsync(gp.x, gp.dev, pg.x, pg.dev, sizeof(float) * T * hc * E, gp.s));
                }
                cur = l.gpu;
            }
            const size_t nx = (size_t) T * hc * E;
            if (inject && l.il > 0) {   // (the layer's input: the reference's own)
                const float* r = ref_rows("l_out-" + std::to_string(l.il - 1), nx, T);
                if (r) {
                    cudaMemcpyAsync(gp.x, r, nx * 4, cudaMemcpyHostToDevice, gp.s);
                    cudaStreamSynchronize(gp.s);
                }
            }
            double tp = prof ? (cudaStreamSynchronize(gp.s), now_s()) : 0.0;
            auto lap = [&](const char* what) {
                if (!prof) return;
                cudaStreamSynchronize(gp.s);
                const double t = now_s();
                prof_acc[std::string(what) + (T > W ? " (prompt)" : " (token)")] += t - tp;
                tp = t;
            };
            if (l.eg >= 0 && !engram(l, gp, T, err)) return false;
            if (l.eg >= 0) lap("engram");
            if (inject && l.eg >= 0) ref_cmp("engram_out", l.il, gp, gp.x, nx, T);
            if (!attention(l, gp, T, p0, err)) return false;
            lap(l.swa ? "attention, window" : "attention, indexed");
            if (inject) {
                ref_cmp("attn_out", l.il, gp, gp.y, (size_t) T * E, T);
                ref_cmp("hc_attn_out", l.il, gp, gp.x, nx, T);
                const float* r = ref_rows("hc_attn_out-" + std::to_string(l.il), nx, T);   // (the FFN alone)
                if (r) {
                    cudaMemcpyAsync(gp.x, r, nx * 4, cudaMemcpyHostToDevice, gp.s);
                    cudaStreamSynchronize(gp.s);
                }
            }
            if (!ffn(l, gp, T, err)) return false;
            lap(l.dense ? "ffn, dense" : "ffn, experts");
            if (inject) {
                ref_cmp("ffn_out", l.il, gp, gp.y, (size_t) T * E, T);
                ref_cmp("l_out", l.il, gp, gp.x, nx, T);
            }
        }
        Gpu& hg = G[(size_t) head_gpu];
        CK(cudaSetDevice(hg.dev));
        if (cur != head_gpu) {   // (the final state and the positions: the head's GPU may hold no layer)
            Gpu& pg = G[(size_t) cur];
            CK(cudaStreamSynchronize(pg.s));
            CK(cudaMemcpyAsync(hg.pos, h_pos, sizeof(int) * 3 * Tc, cudaMemcpyHostToDevice, hg.s));
            CK(cudaMemcpyPeerAsync(hg.x, hg.dev, pg.x, pg.dev, sizeof(float) * T * hc * E, hg.s));
        }
        if (!head_tail(ids, T, p0, want, logits_out, err)) return false;
        CK(cudaStreamSynchronize(hg.s));
        if (k::errors()) {
            err = "a kernel failed (above)";
            return false;
        }
        return true;
    }

    // The head, on its GPU's stream (its x the trunk's final state of the window's rows, its pos the positions): every
    // row's final state (hc_head, output_norm) - kept in the ring by position, the MTP block's input a position later -
    // then the wanted rows' logits, and the MTP block over the chunk (its own caches).  Queued, not waited for.
    bool head_tail(const int* ids, int T, int64_t p0, int want, float* logits_out, std::string& err) {
        const int E = g.n_embd, hc = g.hc;
        Gpu& hg = G[(size_t) head_gpu];
        k::hc_head(hg.x, T, hc, E, hh_fn, hh_scale, hh_base, g.eps, g.hc_eps, hg.hcur, hg.s);
        dbg("hc_head", -1, hg, hg.hcur + (size_t) (T - 1) * E, 1, E);
        k::rmsnorm(hg.hcur, E, T, E, out_norm, g.eps, hg.hcur, E, hg.s);
        dbg("result_norm", -1, hg, hg.hcur + (size_t) (T - 1) * E, 1, E);
        k::store_rows(hg.hcur, T, E, hring, ring, hg.pos, hg.s);
        if (want > 0) {
            if (!mm(hg, out, hg.hcur + (size_t) (T - want) * E, E, want, hg.logits, g.n_vocab)) {
                err = "output head: a GEMV the kernels do not take";
                return false;
            }
            dbg("result_output", -1, hg, hg.logits, (size_t) want, g.n_vocab);
            if (dump) {
                std::vector<float> lr((size_t) g.n_vocab);
                cudaMemcpy(lr.data(), hg.logits + (size_t) (want - 1) * g.n_vocab, lr.size() * 4, cudaMemcpyDeviceToHost);
                std::fprintf(stderr, "DUMP logits[last] %.4f %.4f %.4f ... %.4f %.4f %.4f\n", lr[0], lr[1], lr[2],
                             lr[lr.size() - 3], lr[lr.size() - 2], lr[lr.size() - 1]);
            }
            CK(cudaMemcpyAsync(logits_out, hg.logits, sizeof(float) * want * g.n_vocab, cudaMemcpyDeviceToHost, hg.s));
        }
        if (M.mtp && (T > W || mtp_eager)) {
            if (p0 > mtp_done && !mtp_catch_up(p0, err)) return false;
            mtp_settle();   // (mtp_hin and hm.pos rewritten below: the queued pass first)
            // the MTP block's input rows: the state a position before each token (the first: the ring's, the rest: this
            // chunk's own rows shifted by one)
            const size_t rb = sizeof(float) * E;
            if (p0 == 0) CK(cudaMemsetAsync(mtp_hin, 0, rb, hg.s));
            else CK(cudaMemcpyAsync(mtp_hin, hring + (size_t) ((p0 - 1) % ring) * E, rb, cudaMemcpyDeviceToDevice, hg.s));
            if (T > 1) CK(cudaMemcpyAsync(mtp_hin + E, hg.hcur, rb * (T - 1), cudaMemcpyDeviceToDevice, hg.s));
            const double tm = prof ? (cudaStreamSynchronize(hg.s), now_s()) : 0.0;
            if (mtp_async && T > W) {   // (its own scratch and stream, queued by its thread: beside the next chunk)
                CK(cudaMemcpyAsync(hm.pos, hg.pos, sizeof(int) * 3 * Tc, cudaMemcpyDeviceToDevice, hg.s));
                CK(cudaEventRecord(ev_hin, hg.s));
                mtp_ids.assign(ids, ids + T);
                mtp_pending = true;
                mtp_crew->start([this, T, p0](int) {
                    cudaSetDevice(hm.dev);
                    cudaStreamWaitEvent(hm.s, ev_hin, 0);
                    std::string e2;
                    if (!mtp_forward(mtp_ids.data(), T, p0, 0, e2, nullptr, &hm)) mtp_err = e2;
                    cudaEventRecord(ev_mtp, hm.s);
                });
                if (p0 <= mtp_done) mtp_done = std::max(mtp_done, p0 + T);
                return true;
            }
            if (!mtp_forward(ids, T, p0, 0, err)) return false;
            if (p0 <= mtp_done) mtp_done = std::max(mtp_done, p0 + T);
            if (prof) {
                cudaStreamSynchronize(hg.s);
                prof_acc[std::string("mtp, cache") + (T > W ? " (prompt)" : " (token)")] += now_s() - tm;
            }
        }
        return true;
    }

    // ------------------------------------------------------------ tensor parallel's forward
    // member m's partial gp.y [T][E] to the host (sync v), then the members' sum back into gp.y
    void all_reduce(int m, Gpu& gp, int T, unsigned v) { all_reduce(m, gp, gp.y, (size_t) T * g.n_embd, v); }
    // ... of n floats at buf (in place)
    // (rows: n is rows of E - a prompt chunk's sum then goes by blocks of rows)
    void all_reduce(int m, Gpu& gp, float* buf, size_t n, unsigned v, bool rows = true) {
        const size_t E = (size_t) g.n_embd;
        if (rows && blocked && gp.s2 && n > (size_t) W * E && n % E == 0) {
            publish_blocks(m, gp, buf, (int) (n / E), v);
            if (gp.defer) {   // (a staggered half: its read-back queued now on s3, waited for by its own later step)
                if (gp.s3 && !pull_main) {
                    cudaStreamWaitEvent(gp.s3, gp.ev2, 0);   // (after its partial is made - and packed)
                    Gpu q = gp;
                    q.s = gp.s3;
                    pull_blocks(q, buf, (int) (n / E), v);
                    cudaEventRecord(gp.evh, gp.s3);
                }
                return;
            }
            pull_blocks(gp, buf, (int) (n / E), v);
            return;
        }
        k::publish(buf, tp_part + (size_t) m * tp_stride(), n, tp_cnt[(size_t) m], tp_fout + (size_t) m * kLine, v,
                   gp.s);
        k::wait_pull(tp_fin, v, buf, tp_sum, n, gp.s);
    }
    // a prompt chunk's sum read back into buf block by block, each once the host has summed it
    void pull_blocks(Gpu& gp, float* buf, int T, unsigned v) {
        const size_t E = (size_t) g.n_embd, base = (size_t) gp.row0;
        const int R = blk_rows();
        if (f16_sums && gp.p16) {   // (the F16 sum in by blocks, unpacked once)
            for (int b = 0; b * R < T; ++b) {
                const size_t r0 = (size_t) b * R, nr = (size_t) std::min(R, T - b * R);
                k::wait_flag(tp_bin + (size_t) b * kLine, v, gp.s);
                if (!skip_copy &&
                    (cudaMemcpyAsync(gp.p16 + r0 * E, tp_sum16 + (base + r0) * E, nr * E * 2, cudaMemcpyHostToDevice,
                                     gp.s) != cudaSuccess ||
                     cudaMemcpyAsync(gp.psc16 + r0, tp_sscale + base + r0, nr * 4, cudaMemcpyHostToDevice, gp.s) !=
                         cudaSuccess))
                    (void) cudaGetLastError();
            }
            k::unpack16_rows(gp.p16, gp.psc16, T, (int) E, buf, gp.s);
            return;
        }
        for (int b = 0; b * R < T; ++b) {
            const size_t r0 = (size_t) b * R, nr = (size_t) std::min(R, T - b * R);
            k::wait_flag(tp_bin + (size_t) b * kLine, v, gp.s);
            if (!skip_copy && cudaMemcpyAsync(buf + r0 * E, tp_sum + (base + r0) * E, nr * E * sizeof(float), cudaMemcpyHostToDevice,
                                gp.s) !=
                cudaSuccess)
                (void) cudaGetLastError();
        }
    }
    // part j's T rows of E at buf to the host by blocks (its side stream, once buf is made on its main one), each
    // block's counter after it
    void publish_blocks(int j, Gpu& gp, const float* buf, int T, unsigned v) {
        const size_t E = (size_t) g.n_embd, base = (size_t) gp.row0;
        const int R = blk_rows();
        if (f16_sums && gp.h16) {   // (packed to F16 on the main stream, the halves and factors out by blocks)
            k::pack16_rows(buf, T, (int) E, gp.h16, gp.sc16, gp.s);
            cudaEventRecord(gp.ev2, gp.s);
            cudaStreamWaitEvent(gp.s2, gp.ev2, 0);
            for (int b = 0; b * R < T; ++b) {
                const size_t r0 = (size_t) b * R, nr = (size_t) std::min(R, T - b * R);
                if (!skip_copy &&
                    (cudaMemcpyAsync(tp_part16 + ((size_t) j * Tc + base + r0) * E, gp.h16 + r0 * E, nr * E * 2,
                                     cudaMemcpyDeviceToHost, gp.s2) != cudaSuccess ||
                     cudaMemcpyAsync(tp_pscale + (size_t) j * Tc + base + r0, gp.sc16 + r0, nr * 4, cudaMemcpyDeviceToHost,
                                     gp.s2) != cudaSuccess))
                    (void) cudaGetLastError();
                k::set_flag(tp_bout + ((size_t) j * kBlk + b) * kLine, v, gp.s2);
            }
            return;
        }
        cudaEventRecord(gp.ev2, gp.s);
        cudaStreamWaitEvent(gp.s2, gp.ev2, 0);
        for (int b = 0; b * R < T; ++b) {
            const size_t r0 = (size_t) b * R, nr = (size_t) std::min(R, T - b * R);
            if (!skip_copy && cudaMemcpyAsync(tp_part + (size_t) j * tp_stride() + (base + r0) * E, buf + r0 * E, nr * E * sizeof(float),
                                cudaMemcpyDeviceToHost, gp.s2) != cudaSuccess)
                (void) cudaGetLastError();
            k::set_flag(tp_bout + ((size_t) j * kBlk + b) * kLine, v, gp.s2);
        }
    }
    // the FFN's partials summed in the model's space - each member's (and helper's) routed sum projected up by
    // ffn_routed_up first: a third less to send than [the latent sum | the shared expert's], the same projections
    // (HY4_TP_LATENT_SUM=1: the latent sum, projected after)
    const bool model_sum = std::getenv("HY4_TP_LATENT_SUM") == nullptr;
    const bool check_tp = std::getenv("HY4_TP_CHECK") != nullptr;
    // (debugging only - the answers are no longer the model's: HY4_TP_SKIP_SUM the host adds nothing, HY4_TP_SKIP_COPY
    //  a chunk's partials and sums are not copied, only their counters - each bounds what that part costs)
    const bool skip_sum = std::getenv("HY4_TP_SKIP_SUM") != nullptr;
    const bool skip_copy = std::getenv("HY4_TP_SKIP_COPY") != nullptr;
    const bool hand_inline = std::getenv("HY4_TP_HAND_INLINE") != nullptr;   // (a window's routing copy on its stream)
    const bool hand_rot = std::getenv("HY4_TP_HAND0") == nullptr;   // (HY4_TP_HAND0: member 0 hands every layer's)
    const int ev_m = std::getenv("HY4_TP_EV_M") ? std::atoi(std::getenv("HY4_TP_EV_M")) : 0;   // (HY4_TP_EV's member)
    // (HY4_TP_SUM_EACH: each part added as it comes - measured slower on a window of 4: the team's one sum over every
    //  part reads each once, the adds read the sum again each time)
    const bool sum_all = std::getenv("HY4_TP_SUM_EACH") == nullptr;
    size_t tp_stride() const { return (size_t) Tc * (g.n_lat + g.n_embd); }
    // ------------------------------------------------------------ a prompt chunk in two staggered halves
    // A chunk's rows split in two (A, B): each layer's attention and FFN of one half computed while the other half's
    // sum goes through the host - the members' sums hidden behind their own work.  B's attention at a layer follows A's
    // there (its keys are written), so the halves are the chunk's tokens in order.  The weights are read twice a layer -
    // a chunk's GEMMs are bound by their compute, not by the weights.  (HY4_TP_NO_STAGGER: the chunk in one piece)
    const bool stagger = std::getenv("HY4_TP_NO_STAGGER") == nullptr;
    const bool pull_main = std::getenv("HY4_TP_PULL_MAIN") != nullptr;   // (the read-backs on the main stream: before)
    bool staggered(int T) const { return stagger && blocked && model_sum && T > 4 * W; }
    static int half_rows(int T) { return T / 2; }
    // gp's buffers for the rows [r0, ...) of the chunk: every row-indexed buffer advanced by r0 rows at its largest row
    // size (so the halves' rows never meet); the scratch a step uses and finishes before the next is shared
    Gpu view(const Gpu& gp, int r0, int half) const {
        Gpu v = gp;
        v.allocs.clear();
        const size_t r = (size_t) r0, E = (size_t) g.n_embd, hc = (size_t) g.hc, H = (size_t) gp.sH;
        const size_t ff = (size_t) (gp.sFF ? gp.sFF : g.n_ff), lat = (size_t) g.n_lat;
        const size_t maxk = (size_t) g.ix_top_k + g.ix_kpool;
        const size_t egc = (size_t) std::max(1, pk.eg.cols()) * std::max(1, pk.eg.head_dim);
        v.x += r * hc * E;
        v.cur += r * E;
        v.post += r * hc;
        v.xa += r * E;
        v.qa += r * g.q_lora;
        v.qr += r * g.q_lora;
        v.q += r * H * g.k_mla;
        v.qcat += r * H * g.kv_row();
        v.kvr += r * g.kv_row();
        v.o += r * H * g.kv_lora;
        v.ob += r * H * g.v_mla;
        v.g += r * H * g.v_mla;
        v.y += r * E;
        v.xf += r * E;
        v.gu += r * ff * 2;
        v.hd += r * ff;
        v.rl += r * g.n_expert;
        v.rw += r * 16;
        v.ids += r * 16;
        v.lat += r * lat;
        v.latout += r * (lat + (gp.sFF ? E : 0));
        v.pos += r;
        v.lo += r;
        v.nk += r;
        v.keys += r * maxk;
        v.knk += r;
        v.iq += r * (size_t) g.ix_heads * g.ix_dim;
        v.ik += r * (size_t) g.ix_dim;
        v.ivt += r * 2 * (size_t) g.ix_dim;
        v.wt += r * (size_t) g.ix_heads;
        v.eg_rows += r * egc;
        v.eg_valid += r;
        if (v.eg_sc) v.eg_sc += r * (size_t) pk.eg.cols();
        v.eg_r += r * egc;
        v.eg_kvo += r * (hc + 1) * E;
        if (v.h16) {
            v.h16 += r * E;
            v.p16 += r * E;
            v.sc16 += r;
            v.psc16 += r;
        }
        v.evh = half ? gp.evh1 : gp.evh0;
        v.half = half;
        v.row0 = r0;
        v.defer = true;
        return v;
    }
    // a half's attention: the inputs (Engram), the pre-mix, the attention, its partial out
    bool attn_out(Layer& l, Gpu& gv, int T, int64_t p0, int r0, int m, unsigned v, std::string& err) {
        if (l.eg >= 0 && !engram(l, gv, T, err, r0)) return false;
        k::hc_pre(gv.x, T, g.hc, g.n_embd, l.hca_fn, l.hca_scale, l.hca_base, g.eps, g.hc_eps, g.hc_mag, gv.xa, gv.post,
                  gv.s, l.attn_norm, g.eps);
        if (!attn_core(l, gv, T, p0, err)) return false;
        all_reduce(m, gv, gv.y, (size_t) T * g.n_embd, v);   // (published only: gv.defer)
        return true;
    }
    // (HY4_TP_EV with a staggered chunk: member ev_m's GPU time in its sums' read-backs - the waits for the host's sums
    //  and the copies - against the window's whole)
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> stg_ev;
    size_t stg_n = 0;
    double stg_wait_ms = 0, stg_all_ms = 0;
    long stg_windows = 0;
    void stg_pull(int m, Gpu& gv, int T, unsigned v) {
        const bool on = tp_ev && m == ev_m;
        if (on && stg_n >= stg_ev.size()) {
            stg_ev.emplace_back();
            cudaEventCreate(&stg_ev.back().first);
            cudaEventCreate(&stg_ev.back().second);
        }
        if (on) cudaEventRecord(stg_ev[stg_n].first, gv.s);
        if (gv.s3 && !pull_main) cudaStreamWaitEvent(gv.s, gv.evh, 0);   // (queued on s3 when it was published)
        else pull_blocks(gv, gv.y, T, v);
        if (on) cudaEventRecord(stg_ev[stg_n++].second, gv.s);
    }
    void attn_in(Layer& l, Gpu& gv, int T, unsigned v, int m = -1) {
        stg_pull(m, gv, T, v);
        k::hc_post_norm(gv.x, T, g.hc, g.n_embd, gv.y, gv.post, l.post_attn, g.eps, gv.s);
    }
    bool ffn_out(Layer& l, Gpu& gv, int T, unsigned v, std::string& err) {
        k::hc_pre(gv.x, T, g.hc, g.n_embd, l.hcf_fn, l.hcf_scale, l.hcf_base, g.eps, g.hc_eps, g.hc_mag, gv.xf, gv.post,
                  gv.s, l.ffn_norm, g.eps);
        return ffn_core(l, gv, T, err, v);   // (its sum published only: gv.defer)
    }
    void ffn_in(Layer& l, Gpu& gv, int T, unsigned v, int m = -1) {
        stg_pull(m, gv, T, v);
        k::hc_post_norm(gv.x, T, g.hc, g.n_embd, gv.y, gv.post, l.post_ffn, g.eps, gv.s);
    }
    // member m's layers for a staggered chunk: syncs v0 + 4 il + {1: A's attention, 2: B's, 3: A's FFN, 4: B's}
    bool member_layers_staggered(int m, Gpu& gp, int T, int64_t p0, unsigned v0, std::string& err) {
        const int hA = half_rows(T), hB = T - hA;
        if (tp_ev && m == ev_m) {
            stg_n = 0;
            if (stg_ev.empty() || !stg_t0) {
                cudaEventCreate(&stg_t0);
                cudaEventCreate(&stg_t1);
            }
            cudaEventRecord(stg_t0, gp.s);
        }
        Gpu gA = view(gp, 0, 0), gB = view(gp, hA, 1);
        auto& Ls = TL[(size_t) m];
        auto vs = [&](int il, int q) { return v0 + 4u * (unsigned) il + (unsigned) q + 1u; };
        if (!attn_out(Ls[0], gA, hA, p0, 0, m, vs(0, 0), err) || !attn_out(Ls[0], gB, hB, p0 + hA, hA, m, vs(0, 1), err))
            return false;
        for (int il = 0; il < (int) Ls.size(); ++il) {
            if (tp_abort.load(std::memory_order_relaxed)) {
                err = "stopped";
                return false;
            }
            Layer& l = Ls[(size_t) il];
            attn_in(l, gA, hA, vs(il, 0), m);
            if (!ffn_out(l, gA, hA, vs(il, 2), err)) return false;
            attn_in(l, gB, hB, vs(il, 1), m);
            if (!ffn_out(l, gB, hB, vs(il, 3), err)) return false;
            ffn_in(l, gA, hA, vs(il, 2), m);
            if (il + 1 < (int) Ls.size() && !attn_out(Ls[(size_t) il + 1], gA, hA, p0, 0, m, vs(il + 1, 0), err))
                return false;
            ffn_in(l, gB, hB, vs(il, 3), m);
            if (il + 1 < (int) Ls.size() && !attn_out(Ls[(size_t) il + 1], gB, hB, p0 + hA, hA, m, vs(il + 1, 1), err))
                return false;
        }
        if (tp_ev && m == ev_m) cudaEventRecord(stg_t1, gp.s);
        return true;
    }
    cudaEvent_t stg_t0 = nullptr, stg_t1 = nullptr;
    void stg_collect() {   // (after the window: its read-backs' share of member ev_m's time)
        if (!tp_ev || !stg_t1 || stg_n == 0) return;
        cudaEventSynchronize(stg_t1);
        float all = 0;
        cudaEventElapsedTime(&all, stg_t0, stg_t1);
        double w = 0;
        for (size_t i = 0; i < stg_n; ++i) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, stg_ev[i].first, stg_ev[i].second) == cudaSuccess) w += ms;
            else (void) cudaGetLastError();
        }
        stg_wait_ms += w;
        stg_all_ms += all;
        ++stg_windows;
        stg_n = 0;
    }

    // member m's whole window, queued: the inputs, every layer (sync v0 + 1 .. v0 + 2 n_layer), then the final state to
    // the head's GPU (counter xv) - or, the head on member 0's GPU, the head itself
    bool member_forward(int m, const int* ids, int T, int64_t p0, int want, float* logits_out, unsigned v0,
                        unsigned xv, std::string& err) {
        Gpu& gp = G[(size_t) mem[(size_t) m]];
        CK(cudaSetDevice(gp.dev));
        const size_t nx = (size_t) T * g.hc * g.n_embd;
        CK(cudaMemcpyAsync(gp.pos, h_pos, sizeof(int) * 3 * Tc, cudaMemcpyHostToDevice, gp.s));
        CK(cudaMemcpyAsync(gp.cur, h_x, sizeof(float) * T * g.n_embd, cudaMemcpyHostToDevice, gp.s));
        k::expand_streams(gp.cur, T, g.hc, g.n_embd, gp.x, gp.s);
        unsigned v = v0;
        if (staggered(T) && !member_layers_staggered(m, gp, T, p0, v0, err)) return false;
        for (Layer& l : TL[(size_t) m]) {
            if (staggered(T)) break;
            if (tp_abort.load(std::memory_order_relaxed)) {
                err = "stopped";
                return false;
            }
            mark(m, gp, l.il, 0);
            if (l.eg >= 0 && !engram(l, gp, T, err)) return false;
            if (!attention(l, gp, T, p0, err, m, ++v)) return false;
            mark(m, gp, l.il, 3);
            if (!ffn(l, gp, T, err, m, ++v)) return false;
            mark(m, gp, l.il, 6);
        }
        if (m != 0) return true;
        if (head_member()) return head_tail(ids, T, p0, want, logits_out, err);
        k::publish(gp.x, tp_xo, nx, tp_cnt[0], tp_fx, xv, gp.s);
        return true;
    }
    // helper h's window, queued: each MoE layer's routing from member 0 (the FFN's sync value), its units' expert slices,
    // their latent sum to its part
    bool helper_forward(int h, int T, unsigned v0, std::string& err) {
        Helper& hp = hel[(size_t) h];
        Gpu& gp = G[(size_t) hp.gi];
        CK(cudaSetDevice(gp.dev));
        const int lat = g.n_lat, N = tp;
        if (staggered(T)) {   // (each MoE layer's two halves: syncs v0 + 4 il + 3 and + 4)
            const int hA = half_rows(T);
            for (int il = g.n_dense_lead; il < g.n_layer; ++il)
                for (int q = 0; q < 2; ++q) {
                    if (tp_abort.load(std::memory_order_relaxed)) {
                        err = "stopped";
                        return false;
                    }
                    const int Th = q ? T - hA : hA;
                    Gpu gv = view(gp, q ? hA : 0, q);
                    const unsigned v = v0 + 4u * (unsigned) il + 3u + (unsigned) q;
                    const size_t nrh = (size_t) Th * 16;
                    float* rt = q ? gp.route2 : gp.route;
                    k::wait_pull(q ? tp_froute2 : tp_froute, v, rt, q ? tp_route2 : tp_route, (size_t) Th * (lat + 32), gp.s);
                    k::split3(rt, gv.lat, (size_t) Th * lat, gv.ids, nrh, gv.rw, nrh, gp.s);
                    Layer& l = hp.L[0][(size_t) il];
                    if (!experts_sets(l, gv, Th, err, l.slot_d)) return false;
                    if (!mm(gv, l.rup, gv.latout, lat, Th, gv.y, g.n_embd)) {
                        err = "the helper's ffn_routed_up did not take its shapes";
                        return false;
                    }
                    publish_blocks(N + h, gv, gv.y, Th, v);
                }
            return true;
        }
        const size_t nr = (size_t) T * 16;
        for (int il = g.n_dense_lead; il < g.n_layer; ++il) {
            if (tp_abort.load(std::memory_order_relaxed)) {
                err = "stopped";
                return false;
            }
            const unsigned v = v0 + 2u * (unsigned) il + 2u;
            k::wait_pull(tp_froute, v, gp.route, tp_route, (size_t) T * (lat + 32), gp.s);
            k::split3(gp.route, gp.lat, (size_t) T * lat, gp.ids, nr, gp.rw, nr, gp.s);
            Layer& l = hp.L[0][(size_t) il];
            if (T > W) {
                if (!experts_sets(l, gp, T, err, l.slot_d)) return false;
            } else if (!routed_token(l, gp, T, l.slot_d, gp.latout)) {
                err = "the helper's experts did not take their shapes";
                return false;
            }
            if (model_sum) {
                if (!mm(gp, l.rup, gp.latout, lat, T, gp.y, g.n_embd)) {
                    err = "the helper's ffn_routed_up did not take its shapes";
                    return false;
                }
                if (T > W && blocked && gp.s2) publish_blocks(N + h, gp, gp.y, T, v);
                else
                    k::publish(gp.y, tp_part + (size_t) (N + h) * tp_stride(), (size_t) T * g.n_embd, hp.cnt,
                               tp_fout + (size_t) (N + h) * kLine, v, gp.s);
            } else {
                k::publish(gp.latout, tp_part + (size_t) (N + h) * tp_stride(), (size_t) T * lat, hp.cnt,
                           tp_fout + (size_t) (N + h) * kLine, v, gp.s);
            }
        }
        return true;
    }
    bool window_tp(const int* ids, int T, int64_t p0, int want, float* logits_out, std::string& err) {
        if (tp_broken) {
            err = "tensor parallel stopped after an error: restart the engine";
            return false;
        }
        const int E = g.n_embd, N = tp;
        const bool stg = staggered(T);
        const unsigned v0 = tp_seq, xv = ++tp_xseq, n_sync = (stg ? 4u : 2u) * (unsigned) g.n_layer;
        const int hA = half_rows(T);
        const double tp_t0 = now_s();
        tp_abort.store(false);
        if (tp_ev && evs.empty()) {
            CK(cudaSetDevice(G[(size_t) mem[(size_t) ev_m]].dev));
            evs.resize((size_t) g.n_layer * 7);
            for (auto& e : evs) CK(cudaEventCreate(&e));
        }
        ev_on = tp_ev;
        const int pq = T > W ? 1 : 0;   // (HY4_TP_PROF's kind)
        const int H = (int) hel.size();
        std::vector<std::string> errs((size_t) (N + H));
        crew->start([&](int m) {
            const bool ok2 = m < N ? member_forward(m, ids, T, p0, want, logits_out, v0, xv, errs[(size_t) m])
                                   : helper_forward(m - N, T, v0, errs[(size_t) m]);
            if (!ok2) tp_abort.store(true);
        });
        // the sums, in order: each one when every member's counter has passed it
        static thread_local bool pinned = false;
        if (!pinned) {
            pin_to(pin_cpus);
            pinned = true;
        }
        const size_t stride = tp_stride();
        bool ok = true;
        for (unsigned k2 = 1; ok && k2 <= n_sync; ++k2) {
            const unsigned v = v0 + k2;
            const double t0 = now_s();
            // (staggered: per layer A's attention, B's, A's FFN, B's - each its half's rows)
            const int il = (int) ((k2 - 1) / (stg ? 4 : 2)), q4 = (int) ((k2 - 1) % 4);
            const bool is_ffn = stg ? q4 >= 2 : (k2 % 2) == 0;
            const bool moe = is_ffn && il >= g.n_dense_lead;   // (the helpers' parts in it)
            const bool lat_sum = moe && !model_sum;             // (its payload [latent | shared])
            const int Ts = stg ? ((q4 & 1) ? T - hA : hA) : T;  // (the sync's rows)
            const size_t base = stg && (q4 & 1) ? (size_t) hA : 0;
            // each part added as it comes, in member order (then the helpers' latent parts): the sums sum() makes, the
            // adding beside the waiting for the later members
            const size_t n = (size_t) T * (lat_sum ? g.n_lat + E : E), n_help = (size_t) T * (lat_sum ? g.n_lat : E);
            if (blocked && T > W && !lat_sum) {   // (a prompt chunk's: block by block as the parts come)
                const int R = blk_rows(), parts = N + (moe ? H : 0);
                double tw = 0;
                for (int b = 0; ok && b * R < Ts; ++b) {
                    for (int j = 0; j < parts && ok; ++j) {
                        const volatile unsigned* f = tp_bout + ((size_t) j * kBlk + b) * kLine;
                        for (long spin = 0; (int) (*f - v) < 0; ++spin) {
#if defined(__x86_64__)
                            __builtin_ia32_pause();
#endif
                            if ((spin & 0xffff) == 0xffff && (tp_abort.load() || k::errors() || now_s() - t0 > 60.0)) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (!ok) break;
                    std::atomic_thread_fence(std::memory_order_acquire);
                    const double ts = now_s();
                    const size_t r0 = base + (size_t) b * R, nr = (size_t) std::min(R, Ts - b * R);
                    if (skip_sum) {
                    } else if (f16_sums) {   // (members then helpers, in order: sum16 takes the parts as one list)
                        summer->sum16(tp_sum16 + r0 * E, tp_sscale + r0, tp_part16 + r0 * E, tp_pscale + r0,
                                      N + (moe ? H : 0), (size_t) Tc * E, (size_t) Tc, nr, (int) E);
                    } else {
                        summer->sum(tp_sum + r0 * E, tp_part + r0 * E, N, stride, nr * E, moe ? H : 0, nr * E);
                    }
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    *(volatile unsigned*) (tp_bin + (size_t) b * kLine) = v;
                    tw += now_s() - ts;
                }
                if (tp_prof) {
                    tp_wait[pq][is_ffn ? 1 : 0] += now_s() - t0 - tw;
                    tp_sum_s[pq] += tw;
                }
                continue;
            }
            double t1 = t0;
            for (int j = 0; j < N + (moe ? H : 0) && ok; ++j) {
                const volatile unsigned* f = tp_fout + (size_t) j * kLine;
                for (long spin = 0; (int) (*f - v) < 0; ++spin) {
#if defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                    if ((spin & 0xffff) == 0xffff &&
                        (tp_abort.load() || k::errors() || now_s() - t0 > 60.0)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                std::atomic_thread_fence(std::memory_order_acquire);
                if (j == N + (moe ? H : 0) - 1) t1 = now_s();
                if (sum_all) continue;
                summer->add(tp_sum, tp_part + (size_t) j * stride, j < N ? n : n_help, j == 0);
            }
            if (!ok) break;
            if (sum_all) summer->sum(tp_sum, tp_part, N, stride, n, moe ? H : 0, n_help);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) tp_fin = v;
            if (tp_prof) {
                tp_wait[pq][is_ffn ? 1 : 0] += t1 - t0;
                tp_sum_s[pq] += now_s() - t1;
            }
        }
        if (tp_prof) {
            ++tp_fwd[pq];
            tp_tok[pq] += T;
            tp_fwd_s[pq] += now_s() - tp_t0;
            tp_prep_s[pq] += tp_t0 - win_t0;
            tp_eg_s[pq] += eg_last_s;
        }
        if (!ok) {   // (every wait let go, so the members' queues drain; the state is no longer the model's)
            *(volatile unsigned*) tp_froute = v0 + 0x40000000u;
            *(volatile unsigned*) tp_fin = v0 + 0x40000000u;
            for (int b = 0; b < kBlk; ++b) *(volatile unsigned*) (tp_bin + (size_t) b * kLine) = v0 + 0x40000000u;
            *(volatile unsigned*) tp_fx = xv + 0x40000000u;
            tp_broken = true;
        }
        tp_seq = v0 + n_sync;
        if (ok && !head_member()) {
            Gpu& hg = G[(size_t) head_gpu];
            CK(cudaSetDevice(hg.dev));
            CK(cudaMemcpyAsync(hg.pos, h_pos, sizeof(int) * 3 * Tc, cudaMemcpyHostToDevice, hg.s));
            k::wait_pull(tp_fx, xv, hg.x, tp_xo, (size_t) T * g.hc * E, hg.s);
            if (!head_tail(ids, T, p0, want, logits_out, err)) ok = false;
        }
        crew->wait();
        if (ok && ev_on) {
            CK(cudaSetDevice(G[(size_t) mem[(size_t) ev_m]].dev));
            if (stg) stg_collect();
            else ev_collect(T);
        }
        if (!ok || check_tp)   // (else no wait for every member: the head's covers member 0's, and the next window
            for (int i : mem) {   //  queues behind the others' few last kernels)
                CK(cudaSetDevice(G[(size_t) i].dev));
                CK(cudaStreamSynchronize(G[(size_t) i].s));
            }
        CK(cudaSetDevice(G[(size_t) head_gpu].dev));
        CK(cudaStreamSynchronize(G[(size_t) head_gpu].s));
        for (int j = 0; j < N + H; ++j)
            if (!errs[(size_t) j].empty() && errs[(size_t) j] != "stopped") {
                err = "GPU " + std::to_string(G[(size_t) (j < N ? mem[(size_t) j] : hel[(size_t) (j - N)].gi)].dev) +
                      ": " + errs[(size_t) j];
                tp_broken = true;
                return false;
            }
        if (tp_prof) tp_post_s[pq] += now_s() - tp_t0;
        if (!ok || k::errors() || k::wait_timeouts()) {
            if (err.empty()) err = "tensor parallel: a member's sums did not come (a kernel failed, or a GPU stalled)";
            tp_broken = true;
            return false;
        }
        if (check_tp) {   // (debugging: every member's copy of the state the same)
            std::vector<float> a(nx_of(T)), b(nx_of(T));
            CK(cudaSetDevice(G[(size_t) mem[0]].dev));
            CK(cudaMemcpy(a.data(), G[(size_t) mem[0]].x, a.size() * 4, cudaMemcpyDeviceToHost));
            for (int j = 1; j < N; ++j) {
                CK(cudaSetDevice(G[(size_t) mem[(size_t) j]].dev));
                CK(cudaMemcpy(b.data(), G[(size_t) mem[(size_t) j]].x, b.size() * 4, cudaMemcpyDeviceToHost));
                if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0)
                    std::fprintf(stderr, "HY4_TP_CHECK: member %d's state differs from member 0's at position %lld\n", j,
                                 (long long) p0);
            }
        }
        return true;
    }
    size_t nx_of(int T) const { return (size_t) T * g.hc * g.n_embd; }
};

Model::Model() : m(new Impl) {}
Model::~Model() {
    if (m) m->tp_report();
    if (m) m->ev_report();
    if (m && m->prof) m->prof_report();
}

// The load, then a warm-up (HY4_NO_WARMUP=1: none): a prompt chunk, a token window and a draft of a fixed token, so
// the first request does not pay the GPUs' first launches of each kernel (CUDA loads a kernel's module at its first
// launch), cuBLAS's start and the buffers made at their first use; then the state as loaded - position 0, no tokens,
// the MTP cache empty, the counts as they were
bool Model::load(const Options& o, std::string& err) {
    if (!m->load(o, err)) return false;
    if (std::getenv("HY4_NO_WARMUP")) return true;
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const Stats keep = m->st;
    const int T = std::min<int64_t>(m->Tc, m->o.max_ctx - 8), V = m->g.n_vocab;
    if (T < 2) return true;
    std::vector<int> ids((size_t) T, m->g.bos >= 0 && m->g.bos < V ? m->g.bos : 0);
    std::vector<float> lg((size_t) hy4::k::kMaxWin * V);
    bool ok = forward(ids.data(), T, 0, 1, lg.data(), err) && forward(ids.data(), 4, T, 4, lg.data(), err);
    if (ok && has_mtp()) {
        std::vector<int> d;
        std::vector<float> c;
        ok = draft(ids[0], T + 4, 3, 0.0f, d, c, err);
    }
    if (ok && m->vis) {   // (the tower's kernels and cuBLAS's start, and its weights read once: a small grey picture)
        const int f = m->vgeo.factor() * 2;
        std::vector<uint8_t> px((size_t) f * f * 3, 128);
        std::vector<float> rows;
        ok = encode_image(px.data(), f, f, rows, err);
    }
    m->tokens.clear();
    m->end = 0;
    m->mtp_done = 0;
    m->st = keep;
    if (!ok) {
        err = "the warm-up: " + err;
        return false;
    }
    const double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    char buf[160];
    std::snprintf(buf, sizeof buf, "INFO warmed up in %.1f s (a prompt chunk, a token window, a draft%s)", t1 - t0,
                  m->vis ? ", a picture" : "");
    m->info.push_back(buf);
    return true;
}

bool Model::resumable(int64_t pos) const {
    if (pos == 0) return true;
    if (pos > m->end) return false;
    return pos - m->g.n_swa + 1 >= m->end - m->ring;
}
int64_t Model::end() const { return m->end; }

bool Model::forward(const int* ids, int n, int64_t pos0, int want, float* logits_out, std::string& err) {
    if (n <= 0) {
        err = "an empty forward";
        return false;
    }
    if (!resumable(pos0)) {
        err = "position " + std::to_string(pos0) + ": the sliding-window caches no longer hold the window before it";
        return false;
    }
    if (pos0 + n > m->o.max_ctx) {
        err = "the context is full (" + std::to_string(m->o.max_ctx) + " tokens)";
        return false;
    }
    want = std::max(0, std::min(want, n));
    if (want > W) {
        err = "logits of at most " + std::to_string(W) + " positions a call";
        return false;
    }
    m->tokens.resize((size_t) pos0);
    m->mtp_done = std::min(m->mtp_done, pos0);
    m->tokens.insert(m->tokens.end(), ids, ids + n);
    m->end = pos0;
    // the last chunk takes the last min(n, Tc) tokens (the wanted rows among them), the ones before it chunks of Tc
    const int Tc = m->Tc, last = std::min(n, Tc), head = n - last;
    int done = 0;
    while (done < n) {
        const int T = done < head ? std::min(Tc, head - done) : last;
        if (!m->window(ids + done, T, pos0 + done, done + T == n ? want : 0, logits_out, err)) return false;
        done += T;
        m->end = pos0 + done;
        m->st.tokens += T;
    }
    if (m->inject) m->inject_report();
    return true;
}

int Model::vocab() const { return m->g.n_vocab; }
int64_t Model::max_ctx() const { return m->o.max_ctx; }
int Model::chunk() const { return m->Tc; }
bool Model::draft(int tok, int64_t pos, int n, float floor, std::vector<int>& d, std::vector<float>& c,
                  std::string& err) {
    return m->draft(tok, pos, n, floor, d, c, err);
}
bool Model::has_mtp() const { return m->M.mtp; }
bool Model::pin(void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return false;
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
}
void Model::unpin(void* p) {
    if (p && cudaHostUnregister(p) != cudaSuccess) (void) cudaGetLastError();
}
bool Model::save_profile(std::string& err) const {
    const std::string& path = m->o.profile_out;
    if (path.empty()) return true;
    std::vector<uint32_t> v((size_t) m->g.n_layer * m->g.n_expert, 0);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = (i < m->prof_in.size() ? m->prof_in[i] : 0) + (i < m->prof_cnt.size() ? m->prof_cnt[i] : 0);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f.write((const char*) v.data(), (std::streamsize) (v.size() * 4))) {
            err = tmp + ": not written";
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = path + ": not replaced";
        return false;
    }
    return true;
}
int Model::eos() const { return m->g.eos; }
const VisionGeo* Model::vision() const { return m->vis ? &m->vgeo : nullptr; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err) {
    if (!m->vis) {
        err = "no vision tower loaded (--vision on, and a pack with the mmproj file)";
        return false;
    }
    if (!m->vis->encode(rgb, width, height, rows, err)) return false;
    if (rows.size() != (size_t) m->vgeo.tokens(width, height) * (size_t) m->g.n_embd) {
        err = "vision: the tower gave " + std::to_string(rows.size()) + " values for a picture's " +
              std::to_string(m->vgeo.tokens(width, height)) + " rows";
        return false;
    }
    return true;
}
void Model::set_images(std::vector<Image> images) {
    m->mtp_settle();   // (a queued MTP pass may still read the pictures' rows)
    m->imgs = std::move(images);
}
const Stats& Model::stats() const { return m->st; }
std::vector<std::string> Model::info_lines() const { return m->info; }

}  // namespace hy4
