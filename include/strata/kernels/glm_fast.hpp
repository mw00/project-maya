// include/strata/kernels/glm_fast.hpp - the glm5-next FAST decode kernels (one token, fused).
//
// The correctness-first kernels (glm_hc / glm_kda / glm_dsa / router_sigmoid / native_expert_grouped)
// stay the parity reference; this family is what the decode path runs.  Same arithmetic, fused so
// a layer is ~11 launches instead of ~60, every reduction parallel, and every per-token plan read
// from DEVICE memory (the grouped expert path used to read its plan over PCIe from host-mapped
// memory - every block, every layer - which was most of its 0.4 ms per call).
//
// Weights: ggml quant types are dotted against a q8_1 activation (llama.cpp's vecdotq, the same
// expressions as iq_kernels.cu / native_mmvq.cu); the pack's big BF16 rows stay BF16 on the device
// (dense.bin stores them as BF16, so keeping them is lossless and halves their bytes); small
// parameters are F32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::glmf {

constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;

/// y[r] = alpha * (W_r . x) + (bias ? bias[r] : 0) for r in [0, n_out).  Quantized types read
/// `xq` (q8_1, n_in / 32 blocks); F32 / BF16 weights read `xf`.
///
/// `ncol` > 1: the same weights against that many activations (the tokens of a verify window, a draft block) -
/// column c reads xq + c * xq_cstride bytes / xf + c * xf_cstride floats and writes y + c * y_cstride floats.  Each
/// column's sum is the single-column kernel's, bit for bit; a row's weights are read from DRAM once for all of them.
struct MvJob {
    const void* w = nullptr;
    const void* xq = nullptr;
    const float* xf = nullptr;
    float* y = nullptr;
    const float* bias = nullptr;
    float alpha = 1.0f;
    int type = 0;
    int n_in = 0, n_out = 0;
    int ncol = 1;
    int64_t xq_cstride = 0, xf_cstride = 0, y_cstride = 0;
    /// The sums may differ from the single column's in float rounding (a drafter's GEMVs: its drafts are only
    /// proposals): five or more Q4_K / Q8_0 columns then go to the tensor cores (sm_80+; MAYA_GLM_MV_APPROX=0: never).
    bool approx = false;
};
constexpr int kMaxMvCols = 16;   // a verify window's tokens, a draft block's
constexpr int kMaxMvJobs = 8;
/// One launch for up to kMaxMvJobs independent GEMVs (they share nothing but the launch).
bool mv(const MvJob* jobs, int n, cudaStream_t s);
/// The most activations mv_nt() takes at once (the rows of one batched decode step: one a conversation, or a
/// conversation's verify window of several with MAYA_GLM_MULTI_MTP).
constexpr int kMaxMvNt = 16;
/// mv() for nt activations at once (nt <= kMaxMvNt; 1 = mv()), the jobs' ncol 1: each job's xq / xf hold nt activations
/// one after another (n_in values each) and its y nt outputs (n_out each).  Output t is bit for bit what mv() gives
/// for activation t alone, and each weight row is read from memory once for all of them (more than 8 activations: once
/// for each group of 8, EXL3 weights once for all).
bool mv_nt(const MvJob* jobs, int n, int nt, cudaStream_t s);
/// Whether mv() serves this weight type (the Flash / DFlash2 list, Maya v1.1.55).
bool mv_supported(int type);
/// glm-dsa B10.7: mv_supported plus the extra glm-dsa dense types (glmd_mv_ext.cuh).
bool mv_supported_ext(int type);
/// glm-dsa L1-4: mv() for the glm-dsa decode's GEMVs - the one-column Q5_K / Q8_0 / Q4_K / Q6_K / BF16 / F32 jobs of a
/// batch in one launch (one row a warp when K-quants are among them, else two), every output bit for bit mv()'s
/// (glmd_mv_ext.cuh); a batch of K-quants only, and the other jobs, through mv().  The types a device-tuned one-column
/// kernel of mv() serves always stay with mv() (cc 6.0: the P100 Q6_K kernel; MAYA_GLMD_MV_NW2=1: the K-quants), and a
/// HIP build takes Q4_K / Q5_K / Q8_0 only (the RDNA kernel's Q6_K / BF16; F32 not bit for bit there).
/// cfg kMvDsaAuto: the device's default - on where it was measured faster (NVIDIA cc >= 7.0: V100, glmd_dec_bench), mv()
/// alone elsewhere (AMD: gfx1151 within 0-3%, bandwidth-bound; Pascal: not measured); kMvDsaOn: on any device;
/// kMvDsaOff: mv() alone (A/Bs); 1-3, 5: fixed shapes on any device, for glmd_dec_bench (1: one launch of every job it
/// takes, 2: 4-row warps, 3: a launch a type, 5: one-row warps).
constexpr int kMvDsaOff = -1, kMvDsaAuto = 0, kMvDsaOn = 4;
bool mv_dsa(const MvJob* jobs, int n, cudaStream_t s, int cfg = kMvDsaAuto);
/// The calling thread's launches of mv(), moe_gate_up() and moe_down() read `flag` (a device int) first and do nothing
/// while it is non-zero - a pipelined window's tokens after a wrong draft (skip_gate sets it from the host's word);
/// nullptr (the default): no check.
void set_skip(const int* flag);
/// One thread: *flag = (*host_from <= t) - token t of the window is past the first wrong draft (`host_from`: pinned,
/// host-mapped, written by the host while the window runs).
void skip_gate(const volatile int* host_from, int t, int* flag, cudaStream_t s);
/// Whether moe_gate_up() / moe_down() serve routed experts of this type.
bool moe_supported(int type);
/// Kernel launches of this family that failed since the process started (each is also printed); the forward
/// that sees the count grow reports an error instead of computing on.
int launch_errors();
/// Bytes of one row of `type` with n_in values (0 when unsupported).
size_t row_bytes(int type, int64_t n_in);

/// q8_1 of n floats (n % 32 == 0).
void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s);

/// The mHC read half, fused with the previous block's write half:
///   block_out != nullptr: R_new = post_in (x) block_out + comb_in . R_old  (glm_hc_post), then
///   always:               pre/post/comb from R_new (glm_hc_pre incl. the Sinkhorn), mixed =
///                         sum_s pre_s R_new_s, x = rms_norm(mixed) * norm_w, xq = q8_1(x).
/// block_out == nullptr reads R_old as the residual (R_new is ignored).  post/comb may alias
/// post_in/comb_in (they are overwritten only after every block has read them).
struct HcArgs {
    const float* block_out = nullptr;
    const float* R_old = nullptr;
    float* R_new = nullptr;
    const float* post_in = nullptr;
    const float* comb_in = nullptr;
    float* pre = nullptr;
    float* post = nullptr;
    float* comb = nullptr;
    const uint16_t* w_fn = nullptr;   // BF16 [hc_mix_dim][hc * n_embd]
    const float* w_scale = nullptr;   // 3
    const float* w_base = nullptr;    // hc_mix_dim
    const float* norm_w = nullptr;    // n_embd
    float norm_eps = 1e-5f, hc_eps = 1e-6f;
    int iters = 20, n_embd = 4096;
    float* x = nullptr;               // n_embd f32
    void* xq = nullptr;               // n_embd / 32 q8_1 blocks
    float* part = nullptr;            // scratch: (hc*n_embd / 512) * 25 floats
    unsigned int* counter = nullptr;  // zero-initialised once: the single-tail kernel leaves it zero; the grid-
                                      // barrier kernels (hc2 / hc3) only count it up - each barrier waits for the
                                      // next multiple of the block count, so it needs no reset
};
void hc(const HcArgs& a, cudaStream_t s);
/// n rows' hc (each its own args, partials and counter) in one launch where the grid-barrier kernel can hold them all
/// resident (n <= kMaxMvNt), else one launch a row; each row runs hc3's block function on its own args, so its
/// result is meant to be hc()'s (glm_rows_kernels_test compares them bitwise)
void hc_rows(const HcArgs* a, int n, cudaStream_t s);
/// The write half alone (the split boundary / the head): R_new = post (x) block_out + comb . R_old.
void hc_post(const float* block_out, const float* R_old, const float* post, const float* comb, int n_embd,
             float* R_new, cudaStream_t s);
/// The head's prologue: mean over the 4 streams of R, rms_norm * norm_w -> x (f32) and xq (q8_1).
void head_prep(const float* R, const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s);
/// R[s][e] = emb[e] for the 4 streams.
void embed_streams(const float* emb, float* R, int n_embd, cudaStream_t s);

/// KDA, after the projections: the three causal convs (+SiLU, + the history slide), the q/k l2
/// norms, and the two gate GEMVs (f_b with dt bias and the lower-bound squash -> g1; g_b -> g2).
struct KdaPrepArgs {
    const float* proj[3] = {nullptr, nullptr, nullptr};     // q, k, v projections (d_inner)
    const float* conv_w[3] = {nullptr, nullptr, nullptr};   // [d_inner][d_conv] f32
    float* conv_state = nullptr;                            // 3 x d_inner x (d_conv-1), this layer
    float* out[3] = {nullptr, nullptr, nullptr};
    const float* fa = nullptr;                              // head_dim (f_a output)
    const float* ga = nullptr;                              // head_dim (g_a output)
    const uint16_t* f_b = nullptr;                          // BF16 [d_inner][head_dim]
    const uint16_t* g_b = nullptr;
    const float* dt_bias = nullptr;                         // d_inner
    const float* ssm_a = nullptr;                           // n_head
    float lower_bound = -5.0f;
    float* g1 = nullptr;                                    // d_inner
    float* g2 = nullptr;                                    // d_inner
    int n_head = 64, head_dim = 128, d_conv = 4;
};
void kda_prep(const KdaPrepArgs& a, cudaStream_t s);
/// The gated delta recurrence (one token) + the output gate (rms over head_dim * norm_w *
/// sigmoid(g2)) + q8_1 of the gated output (the attn_output GEMV's input).
void kda_rec(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw,
             float* state, const float* g2, const float* norm_w, float eps, int n_head, int head_dim,
             void* out_q8_1, cudaStream_t s);
/// kda_prep then kda_rec of one token through n <= kKdaLayers recurrent layers at once (two launches; each layer's
/// arithmetic the one-layer kernels' bit for bit), the states only unless rec[i].out (a window commit's replay) - or of
/// n rows' tokens through one layer (the batched rows, each row its own scratch and output; the same block
/// functions as kda_prep / kda_rec - glm_rows_kernels_test compares them bitwise).
/// rec[i]'s q / k / v / g1 / g2: prep[i]'s out[0..2] / g1 / g2
constexpr int kKdaLayers = 8;
struct KdaRecArgs {
    const float *q = nullptr, *k = nullptr, *v = nullptr, *g1 = nullptr, *beta_raw = nullptr;
    float* state = nullptr;
    const float *g2 = nullptr, *norm_w = nullptr;
    void* out = nullptr;   // the gated output (q8_1, kda_rec's out_q8_1); null: the state only (a commit's replay)
};
void kda_layers(const KdaPrepArgs* prep, const KdaRecArgs* rec, int n, float eps, cudaStream_t s);
/// The batched rows' KDA inputs kept for a verify window's commit (MAYA_GLM_MULTI_MTP): row i's q | k | v projections
/// (d_inner each), f_a, g_a (head_dim each) and beta (n_head) - read from src[0..5] at row i (rows di / hd / nh apart) -
/// into its conversation's record at dst[i] in the verify window's layout (glm_window.cu's kda_in: each piece tmax
/// rows, row t[i] of it).  dst[i] null: row i is not kept.  One launch for every row of a layer.
struct KdaGatherArgs {
    const float* src[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    float* dst[kMaxMvNt] = {};
    int t[kMaxMvNt] = {};
    int n_rows = 0, tmax = 0, di = 0, hd = 0, nh = 0;
};
void kda_gather(const KdaGatherArgs& a, cudaStream_t s);

/// DSA, after the first projections: q_a norm (-> qr f32 + q8_1), kv_a norm into the latent cache
/// at position p, the indexer key's layer norm into its cache, the compressor gate into its cache,
/// and the completed pool's pooled key when (p + 1) % kpool == 0.
struct DsaPrepArgs {
    const float* qr_raw = nullptr; const float* q_a_norm = nullptr; float* qr = nullptr; void* qr_q = nullptr;
    int q_lora = 1536;
    const float* kv_raw = nullptr; const float* kv_norm = nullptr; uint16_t* lat = nullptr; int kv_lora = 512;
    int lat_q8 = 0;   // the latent cache holds INT8 records (lat8_rec_bytes each) instead of FP16 rows
    // KV streaming (glm_kv_stream.hpp): the row also goes to the host copy, and to its block's VRAM slot when lat_table
    // names one (lat, the identity-layout cache, is null then)
    uint16_t* lat_host = nullptr; uint16_t* lat_slots = nullptr; const int* lat_table = nullptr; int lat_page = 4;
    const float* ik_raw = nullptr; const float* k_norm_w = nullptr; const float* k_norm_b = nullptr;
    float* ik_cache = nullptr;
    const float* ig_raw = nullptr; float* ig_cache = nullptr;
    const float* ape = nullptr; float* pooled = nullptr;
    int idx_key = 128, kpool = 4;
    int ring = 1 << 30;   // the ik / ig caches hold positions modulo ring (a multiple of kpool)
    int p = 0;
    float eps = 1e-5f;
 // the latent page table routing (NULL = incumbent absolute rows; the launchers'
    // route helper maps the write's logical row to the page table's physical row).  "paged_"
    // renamed in the port: upstream's KV streaming (glm_kv_stream.hpp) owns the plain names.
    const int32_t* paged_lat_table = nullptr;
    int paged_page_rows = 0;
    int paged_max_rows = 0;   // the pool's row capacity (the cells route skips entries past it)
};
void dsa_prep(const DsaPrepArgs& a, cudaStream_t s);
/// score[p] = sum_h relu(iq_h . pooled_p) * iw_h for the n_pools completed pools.
/// token: a decode token (fast_dsa, a decode row) - only it takes the take-two kernel MAYA_GLM_DSA_SCORE2 names (a
/// prompt's windows keep the original: their per-token calls run beside the expert copies; glm_attn_bench's override
/// applies anywhere); the scores are the same bits either way.
void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools,
               float* score, cudaStream_t s, bool token = false);
/// The stable top-k of the pools (ties by lower index) expanded to cells, then the tail cells.
/// paged_lat_table routes the cells through the sequence's latent page table (logical token
/// position -> physical row; NULL = incumbent).
void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells,
                cudaStream_t s, const int32_t* paged_lat_table = nullptr, int paged_page_rows = 0,
                int paged_max_rows = 0);
/// dsa_score of up to kMaxMvNt rows of different sequences (the batched rows) in one launch: row r's query, pools,
/// weights, scores and pool count as dsa_score's arguments - each row's scores dsa_score's
struct DsaScoreRows {
    const float* iq[kMaxMvNt];
    const float* pooled[kMaxMvNt];
    const float* iw[kMaxMvNt];
    float* score[kMaxMvNt];
    int n_pools[kMaxMvNt];
};
void dsa_score_rows(const DsaScoreRows& a, int n, int key_dim, int idx_heads, cudaStream_t s);
/// dsa_select of up to kMaxMvNt rows of different sequences (the batched rows) in one launch, a block a row: row r's
/// scores, pools and cells as dsa_select's arguments - each row's cells dsa_select's
struct DsaSelectRows {
    const float* score[kMaxMvNt];
    int* cells[kMaxMvNt];
    int n_vis[kMaxMvNt], top_pools[kMaxMvNt], n_sel[kMaxMvNt], pos[kMaxMvNt];
};
void dsa_select_rows(const DsaSelectRows& a, int n, int kpool, cudaStream_t s);
/// While a token's every pool is in its selection (n_vis <= the top pools, a context of at most idx_top_k positions):
/// dsa_select's cells in position order - no scores needed (the indexer's query and dsa_score are left out).  The
/// decode's paths all take it alike (dsa_select_all_on; MAYA_GLM_DSA_ALL=0: off), so a window's token is the
/// token path's bit for bit
bool dsa_select_all_on();
void dsa_select_all(int n_vis, int kpool, int n_sel, int pos, int* cells, cudaStream_t s);
/// Absorbed MLA for one token, per head: q_abs = wk_b_h . q_h, scores over the selected latents,
/// softmax, ctx, out_h = wv_b_h . ctx; writes the q8_1 of the n_head * v_head output.
/// lat_q8: the cache holds INT8 latent records (see lat8_rec_bytes), else FP16 rows of kv_lora.
/// marks (the stage profiler, MAYA_GLM_PROF): called after each step it launched with that step's name -
/// "mla_q_abs", "mla_attend", "mla_out", or "mla" for the one-kernel form; nullptr: none.  Launches unchanged.
struct MlaMarks {
    void (*fn)(void* ctx, const char* name);
    void* ctx;
};
/// token: the decode's token path - only it takes the take-two split attention MAYA_GLM_MLA_ATTN2 names (as dsa_score).
void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel,
         int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8 = false,
         const MlaMarks* marks = nullptr, bool token = false);
/// The same absorbed MLA in three steps for a verify window (each step mla()'s arithmetic, bit for bit): the head-wise
/// q_abs = wk_b . q of `ncol` tokens at once (q rows q_stride floats apart, q_abs rows qa_stride), one token's attention
/// over its selected cells into ctx (n_head x kv_lora; false: not available here - use mla()), and the head-wise
/// wv_b . ctx of ncol tokens at once into their q8_1 rows (out_stride_bytes apart).  The two GEMVs read wk_b / wv_b once
/// for all the tokens, where mla() per token read them T times.
void mla_q_abs(const float* q, int64_t q_stride, const uint16_t* wk_b, int n_head, int qk_nope, int kv_lora, int ncol,
               float* q_abs, int64_t qa_stride, cudaStream_t s);
bool mla_attend(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head, int qk_nope,
                int kv_lora, float* ctx, cudaStream_t s, bool lat_q8);
/// mla_attend of T tokens in a few launches (up to kMlaTokMax tokens each; the tensor-core attention: NVIDIA sm_80+,
/// kv_lora 512): token t's q_abs at q_abs + t * q_stride, its n_sel[t] cells at cells + t * c_stride, its ctx at
/// ctx + t * ctx_stride - each token's sums mla_attend's bit for bit.  false: not available here (mla_attend per token)
constexpr int kMlaTokMax = 8;
bool mla_attend_batch(const float* q_abs, int64_t q_stride, const uint16_t* lat, const int* cells, int64_t c_stride,
                      const int* n_sel, int T, int n_head, int qk_nope, int kv_lora, float* ctx, int64_t ctx_stride,
                      cudaStream_t s, bool lat_q8);
/// mla_attend_batch for tokens of DIFFERENT sequences (the batched rows of several conversations): token t's latents at
/// lat[t] and its n_sel[t] cells at cells[t] (each its own cache: the resident rows, or a streamed layer's slot rows),
/// the partials in the caller's `part` (mla_rows_part_floats() a token, room for part_tokens tokens: launches of
/// min(part_tokens, kMlaTokMax) tokens) - each token's sums mla_attend's bit for bit.  false: not available here
/// (mla_rows_ok), nothing launched
size_t mla_rows_part_floats(int n_head, int kv_lora);
bool mla_rows_ok(int kv_lora);
bool mla_attend_rows(const float* q_abs, int64_t q_stride, const uint16_t* const* lat, const int* const* cells,
                     const int* n_sel, int T, int n_head, int qk_nope, int kv_lora, float* ctx, int64_t ctx_stride,
                     float* part, int part_tokens, cudaStream_t s, bool lat_q8);
void mla_out(const float* ctx, int64_t ctx_stride, const uint16_t* wv_b, int n_head, int kv_lora, int v_head, int ncol,
             void* out_q8_1, int64_t out_stride_bytes, cudaStream_t s);
/// The INT8 latent cache (MAYA_GLM_KV_INT8): per position kv_lora int8 codes, then one FP16 scale per 32 values
/// (symmetric, absmax / 127) - 544 bytes at kv_lora 512 against the FP16 row's 1024.
__host__ __device__ constexpr inline int lat8_rec_bytes(int kv_lora) { return kv_lora + kv_lora / 16; }

/// SwiGLU with GLM's clamp (gate above, up both sides) and the q8_1 of h, n values.
void swiglu_q8(const float* gate, const float* up, float limit, int n, void* hq, cudaStream_t s);

// ---------------------------------------------------------------- the routed experts
//
// Three tiers, decided ON THE DEVICE per routed expert: tab[key] != 0 -> resident in VRAM (a hit);
// else rtab[key] != 0 -> resident in the pinned RAM tier: the fetch kernel pulls it over PCIe (11.5 GB/s,
// measured) into one of the layer's SPARE slots (it is then resident: the device writes tab itself) or,
// with no spare left, into a scratch slot; else it is only on disk: the host reads it (the one case the
// device waits for).  key = layer * n_expert + expert.  tab, rtab and the spare table are ONE device
// array (tab at [0, N), rtab at [N, 2N), spares at [2N, 2N + n_layers * kSpares)) so a single update
// kernel applies the host's between-token edits.
constexpr int kSpares = 3;
/// The per-device routing state (device memory unless noted).
struct MoeDev {
    unsigned long long* tab = nullptr;         // [3 tables, see above]
    const uint8_t* helper_owned = nullptr;      // optional GLM helper ownership, checked before primary admission
    int n_keys = 0;                            // N = n_layers * n_expert
    unsigned long long* scratch = nullptr;     // [8] VRAM slots for fetched experts that could not be promoted
    unsigned long long* plan_ptr = nullptr;    // [8]
    float* plan_w = nullptr;                   // [8]
    int* plan_id = nullptr;                    // [8]
    unsigned long long* fetch_src = nullptr;   // [8] host-mapped source (0 = nothing to fetch for entry i)
    unsigned long long* pf_src = nullptr;      // [8] the NEXT layer's predicted experts being prefetched: source,
    unsigned long long* pf_dst = nullptr;      // [8] ... their (claimed spare) slot,
    int* pf_n = nullptr;                       // ... and how many (the side-stream fetch kernel reads these)
    unsigned int* seq = nullptr;               // the device's request counter
    unsigned int* wait_seq = nullptr;          // the seq the wait kernel must see answered (0 = none)
    void* ring = nullptr;                      // host-mapped request ring (device pointer)
    const void* resp = nullptr;                // host-mapped response (device pointer)
    float* cpu_part = nullptr;                 // n_embd: the host's contribution (misses computed off-GPU)
    int* cpu_flag = nullptr;                   // 1 when cpu_part holds this layer's contribution
    unsigned int* cpu_seq = nullptr;           // the CPU LANE: the seq whose answer moe_cpu_wait must see (0 = none)
    const void* cpu_ans = nullptr;             // ... host-mapped CpuAnswer (device pointer)
    unsigned int* dcnt = nullptr;              // ... [n_keys] routes per key (halved every 4096 routes): the coldest
                                               //     RAM-tier experts go to the host, the hot ones are promoted
    volatile int* route_error = nullptr;       // host-mapped, first invalid route: 4 * layer + kind (1..3)
    int* route_err_d = nullptr;                // ... and its device copy, which the route kernel reads (a read of the
                                               //     host-mapped one crossed PCIe at every route's start); null: none
    const int* skip = nullptr;                 // set_skip's flag: non-zero = the experts' kernels do nothing
    // moe_experts_multi: mt_T tokens in one launch, token t's plan at plan_ptr / plan_w + 8 t and its rows at these
    // strides (q8_1 blocks: its FFN input, its h, its shared expert's h; floats: its output, its CPU part)
    int mt_T = 1;
    unsigned int mt_xq_s = 0, mt_hq_s = 0, mt_sh_hq_s = 0, mt_out_s = 0, mt_cpu_s = 0;
    // the routing ring's shape (ring_entry below): ring_n entries (a power of two) of ring_stride bytes; 0, 0: kRingSize
    // entries of sizeof(MoeRequest)
    unsigned int ring_n = 0, ring_stride = 0;
};
/// The widest residual the host-mapped routing structs below carry (glm5-next 4096, glm-dsa 6144).
constexpr int kMaxEmbd = 8192;
/// moe_route's hard limits: n_embd <= kMaxEmbd (the request's x), k <= 8 (the plan arrays), n_expert <= 512 (its
/// shared score tables).  nullptr when a geometry fits, else why not (fast_setup and moe_route refuse it).
inline const char* moe_limits_error(int n_embd, int k, int n_expert) {
    if (n_embd <= 0 || n_embd > kMaxEmbd) return "n_embd outside 1..kMaxEmbd (8192, the routing ring's width)";
    if (k < 1 || k > 8) return "experts per token outside 1..8";
    if (n_expert < 1 || n_expert > 512) return "n_expert outside 1..512";
    return nullptr;
}
/// The CPU LANE's answer: the weighted sum of the RAM-tier experts the host computed for route `seq`.
struct CpuAnswer {
    volatile unsigned int seq;
    unsigned int pad[3];
    float part[kMaxEmbd];
};
/// Ring entries: a non-waiting route can be lapped; 256 covers a 76-route glm-dsa token (B2.5)
constexpr int kRingSize = 256;
/// LOOKAHEAD: a route may also carry the top-k of the next (up to) kAhead layers' routers applied to its FFN input.
constexpr int kAhead = 4;
/// One routing request, written by the device into host-mapped memory.
struct MoeRequest {
    volatile unsigned int seq;   // written LAST (after a system fence)
    int layer;
    int error;                   // invalid route: no experts or table edits; the host still retires its sequence
    unsigned int miss_mask;      // experts the host must provide (only on disk) - the device waits for these
    unsigned int fetch_mask;     // experts fetched from the RAM tier
    unsigned int promo_mask;     // ... of which promoted into a spare slot: promo_ptr[i] is the slot
    int ids[8];
    float w[8];
    unsigned long long promo_ptr[8];
    int pred[8];                 // the NEXT layer's router applied to this layer's FFN input (-1: none)
    int pf_n;                    // of which prefetched into the next layer's spares: pf_ids / pf_ptr (slot)
    int pf_ids[8];
    unsigned long long pf_ptr[8];
    short ahead[kAhead][8];      // layer + 1 + d's router on this layer's FFN input: its top-k (-1: none)
    unsigned int cpu_mask;       // the CPU LANE: RAM-tier experts the host computes (the device skips them) ...
    unsigned int skip_mask;      // DISK SKIP: disk-only experts left out of this token (MAYA_GLM_DISK_SKIP_W / _FROM)
    unsigned long long cpu_src[8];   // ... their RAM-tier blobs
    short near_ids[16];          // MAYA_GLM_ROUTE_LOG only: the route's next ranks after the top k (-1: none)
    int win;                     // ... and the answer slot they go to: a verify window's token (0: the lone answer)
    int win_n;                   // a batched verify window's tokens (0: not one): their CPU lanes may run at once
    int win_rows;                // ... a batched rows step's group instead (MAYA_GLM_GROUP_EXPERTS): win the row,
    int pad_;                    //     win_n the rows; their CPU lanes merged on this part's own pool
    float x[kMaxEmbd];           // the FFN input, written only when cpu_mask != 0
};
/// A ring sized from the model (fast_setup on HIP): each entry MoeRequest's fields + n_embd floats of x (rounded up to
/// its alignment), as many entries as a decode round routes (every layer once + a verify window's drafts) as a power of
/// two, kRingMinEntries (v1.1.56's 64) .. kRingSize.  The device writes every route's x to its entry: on the gfx1151
/// APU the full 256 x 8192-float ring (8.4 MB host-mapped) made GLM-5.3-Flash's drafting decode ~0.8 % slower than
/// its own 64 x 4096 floats (1 MB).
constexpr unsigned int kRingMinEntries = 64;
constexpr unsigned int ring_stride_for(int n_embd) {
    return (unsigned int) ((offsetof(MoeRequest, x) + (size_t) n_embd * sizeof(float) + alignof(MoeRequest) - 1) /
                           alignof(MoeRequest) * alignof(MoeRequest));
}
constexpr unsigned int ring_entries_for(int routes) {
    unsigned int n = kRingMinEntries;
    while (n < (unsigned int) routes && n < (unsigned int) kRingSize) n *= 2u;
    return n;
}
/// Ring entry `seq`: n entries (a power of two) of `stride` bytes (0, 0: kRingSize entries of sizeof(MoeRequest)).
__host__ __device__ inline MoeRequest* ring_entry(void* ring, unsigned int seq, unsigned int n, unsigned int stride) {
    if (n == 0u) n = (unsigned int) kRingSize;
    if (stride == 0u) stride = (unsigned int) sizeof(MoeRequest);
    return (MoeRequest*) ((char*) ring + (size_t) (seq & (n - 1u)) * stride);
}
/// The host's answer to a request with misses.
struct MoeResponse {
    volatile unsigned int seq;
    unsigned int ptr_mask;            // plan entries the host made resident: ptr[i] valid
    unsigned long long ptr[8];
    // MAYA_GLM_STREAM_FETCH=1: entries in stream_mask were posted before their disk reads finished; ready[i] gets
    // bit (part * s_chunks + chunk) as each chunk lands in ptr[i] (fast_read_part's layout: gate 0, up s_gu, down
    // 2 s_gu), and the fetch kernels copy each chunk once its bit is set (zephel01/Abyss d626d96, mw00 #570)
    unsigned int stream_mask;
    unsigned int ready[8];
    unsigned long long s_gu, s_dn;
    int s_chunks;
    int n_upd;                        // table updates: tab[key] = val
    int upd_key[64];
    unsigned long long upd_val[64];
    int cpu;                          // 1: cpu_part (host-mapped, below) holds the missed experts' weighted sum
    float cpu_part[kMaxEmbd];
};
/// moe_route's weight normalisation (norm_w on): w_i = p_i / D * w_scale over the k selected sigmoid scores.
/// kRouteNormClamp: D = max(sum, 2^-14) (Maya, llama.cpp's ggml_clamp); kRouteNormHf: D = sum + 1e-20 (HF
/// GLM-5.3 / DeepSeek-V3, glm-dsa glm-dsa B5.1).  The two differ only when sum < 6.1e-5 (all k sigmoids tiny).
constexpr int kRouteNormClamp = 0, kRouteNormHf = 1;
/// Router (sigmoid + bias, stable top-k, normalised weights - router_sigmoid's arithmetic), the plan
/// lookup in `tab`, the request publish, and the shared expert's swiglu + q8_1.
void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w,
               int layer, const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up,
               float sh_limit, int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits = nullptr,
               const float* pred_bias = nullptr, int max_prefetch = 0, const float* ahead_logits = nullptr,
               const float* const* ahead_bias = nullptr, int n_ahead = 0, int skip_from = 8,
               unsigned long long cpu_plan = 0, int promote_min = 0, int pf_rank = 0, int near_n = 0, int win = 0,
               unsigned int* scr_used = nullptr, int win_n = 0, int disk_skip_from = 8, float disk_skip_w = 0.0f,
               float miss_skip_w = 0.0f, int win_rows = 0, bool resident_ok = true, int norm_mode = kRouteNormClamp);
/// A verify window's MoE layer with each distinct expert computed ONCE: the T tokens were routed one after another into
/// their own plan arrays (moe_route with win = t and a scr_used word, so no two of them land in one scratch slot - the
/// word's bit 31 says one did), then moe_win_experts groups the entries by expert slot, reads each expert's gate/up and
/// down rows once for all the tokens it serves, and combines each token's rows in plan order as moe_down does - every
/// token's output is moe_gate_up + moe_down's for it alone, bit for bit.  The CPU lane's answers stay moe_cpu_wait's.
struct MoeWin {
    const unsigned long long* plan_ptr = nullptr;   // [T][8] the tokens' plans (0: an entry the GPU does not compute)
    const float* plan_w = nullptr;                  // [T][8]
    const float* cpu_part = nullptr;                // [T][n_embd] the host's answers to disk misses, when flagged
    const int* cpu_flag = nullptr;                  // [T]
    const void* xq = nullptr;                       // the tokens' FFN inputs (q8_1), xq_stride bytes apart
    size_t xq_stride = 0;
    void* hq = nullptr;                             // scratch: [T * 8][n_ff / 32] q8_1
    float* part = nullptr;                          // scratch: [T * 8][n_embd]
    const float* sh_out = nullptr;                  // [T][n_embd] the shared expert's down rows, or null
    float* out = nullptr;                           // [T][n_embd], out_stride floats apart
    size_t out_stride = 0;
    unsigned long long* grp_ptr = nullptr;          // scratch: [T * 8]
    int* grp_start = nullptr;                       // scratch: [T * 8 + 1]
    int* grp_ent = nullptr;                         // scratch: [T * 8]
    int* n_groups = nullptr;                        // scratch: [1]
    int T = 0, k = 0, n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    size_t down_off = 0;
};
constexpr int kMaxWinTokens = kMaxMvCols;   // the window's tokens moe_win_experts groups (16 x 8 entries)
/// par_group: the grouping with a thread an entry (moe_win_group's par), the same arrays
bool moe_win_experts(int gu_type, int d_type, const MoeWin& w, cudaStream_t s, bool par_group = false);
/// moe_win_experts' grouping alone (w's plan_ptr, T, k -> grp_ptr / grp_start / grp_ent / n_groups), for kernels of
/// other formats (EXL3: exl3::win_experts)
/// par: a thread an entry instead of one thread for all (the same groups, order and members, value for value)
void moe_win_group(const MoeWin& w, cudaStream_t s, bool par = false);
/// The CPU LANE's split: of f RAM-tier experts in a route, cpu_take(plan, f) go to the host (4 bits per f, f = 0..8).
inline int cpu_take(unsigned long long plan, int f) { return (int) ((plan >> (4 * f)) & 15ull); }
/// Waits (on the device) for the host's CPU-lane answer of the last route when it had CPU experts, then adds it to
/// `out` (the layer's FFN output, which moe_down wrote first - its work runs while the CPU computes).
void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s);
/// ... a layer's late answer into the streams: R[j] += post[j] * part, j < hc (MAYA_GLM_DEFER_CPU)
void moe_cpu_wait_streams(const MoeDev& d, int n_embd, const float* post, float* R, int hc, cudaStream_t s);
/// The side stream's copy of the prefetch list (pf_src -> pf_dst, pf_n entries of blob_bytes each).
void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s);
/// MAYA_GLM_PF_CE: need = (this route prefetched anything) ? its seq : 0 - read after the route, on its stream
void moe_pf_need(const MoeDev& d, unsigned int* need, cudaStream_t s);
/// MAYA_GLM_PF_CE: hold the stream until flag reaches need (no-op for need 0; gives up after ~25 s)
void moe_pf_wait(const unsigned int* need, const unsigned int* flag, cudaStream_t s);
/// Waits (on the device) for the host's response when the last route had misses, then applies it.
void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s);
/// Copies every planned expert with a fetch source from the RAM tier (host-mapped) into its VRAM slot.
void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s);
/// moe_wait then moe_fetch as ONE launch (the same effects; MAYA_GLM_SKIP_EMPTY's batched rows): a route with no
/// disk miss and nothing to fetch costs one near-empty launch instead of two.
void moe_wait_fetch(const MoeDev& d, int n_embd, int k, size_t blob_bytes, cudaStream_t s);
/// moe_cpu_wait for the n rows of a batched step in one launch: row r waits for the answer *seq[r] names (0: none) in
/// the host-mapped CpuAnswer ans[r] and adds it to out[r].
struct CpuWaitRows {
    const unsigned int* seq[kMaxMvNt] = {};
    const void* ans[kMaxMvNt] = {};
    float* out[kMaxMvNt] = {};
};
void moe_cpu_wait_rows(const CpuWaitRows& w, int n, int n_embd, cudaStream_t s);
/// The routed gate/up rows of the planned (resident) experts, the clamped swiglu, the q8_1 of h; and (sh_down and
/// sh_out given) the shared expert's down rows over its h in sh_hq into sh_out, by extra blocks of the same launch.
void moe_gate_up(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                 const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s);
/// out[r] = (sum_i w_i * down_i(r) . h_i over the resident experts) [+ cpu_part[r]] + sh_out[r] (when given).
/// down_off = the byte offset of the down rows inside an expert blob ([gate | up | down]).
void moe_down(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
              const float* sh_out, float* out, cudaStream_t s);
/// moe_gate_up then moe_down for d.mt_T tokens in two launches (a verify window's tokens, each with its own plan and
/// rows at d's mt_ strides): every token's arithmetic is its moe_gate_up / moe_down's, the blocks of all the tokens for
/// a chunk of rows issued together - an expert two tokens route reads its rows once from DRAM.  AMD builds only (the
/// token path's RDNA3 kernels); false: not here, nothing launched (the caller runs the tokens one by one).
bool moe_experts_multi(int gu_type, int d_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit,
                       size_t down_off, const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq,
                       int n_ff_sh, float* sh_out, float* out, cudaStream_t s);

/// The prompt path's LIGHTLY routed experts (a few rows each - MMQ's 64-128-token tiles would be mostly empty):
/// light[3 i .. 3 i + 2] = {slot, r0, nr} (device): expert i is the blob at base + slot * stride, its rows are
/// [r0, r0 + nr) of the sorted rows (token row_tok[r]), all inside [rlo, rhi).  x: the chunk's n_tok token rows
/// (n_embd floats).  Gate/up then swiglu (limit) then down, the down rows into out rows (ld floats apart; out also
/// holds the gate/up rows in between).  Scratch: xq for n_tok x n_embd q8_1, hq for (rhi - rlo) x n_ff q8_1.
/// False: a type without the light kernels (the caller keeps MMQ for them).
bool rows_experts(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light, int n,
                  const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo, int rhi,
                  void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s);

/// tab[keys[i]] = vals[i] for i < n (the expert-table edits the host queues between tokens).
void tab_update(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n, cudaStream_t s);

/// The NextN block's input: cat = [rms(emb) * enorm, rms(h) * hnorm] (2 n values) as q8_1.  n <= kMaxEmbd, a
/// multiple of 32 (else refused: an error is counted).
void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1,
            cudaStream_t s);
/// h += add (add may be null), then x = rms(h) * w and its q8_1 (a plain pre-norm residual step).  n <= kMaxEmbd, a
/// multiple of 32 (else refused: an error is counted).
void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s);

/// A chained draft's embedding: out = row *tok (a device int) of token_embd on the device (n_rows rows of type 14 =
/// Q6_K or 8 = Q8_0), dequantized as ggml's to_float does (the same floats).  n_embd % 256 == 0.
void embed_tok(const void* table, int type, int n_rows, const int* tok, int n_embd, float* out, cudaStream_t s);
/// argmax over n floats -> *out (device int).  Ties: lowest index.  map: an index i >= map_from is written as
/// map[i - map_from] (a draft head's gathered rows -> their token ids).  No finite value among the n (#860): *out is
/// 0 - a valid id, map[] never read - and the current device's argmax_nonfinite() count goes up.
void argmax(const float* x, int n, int* out, cudaStream_t s, const int* map = nullptr, int map_from = 0);
/// How many argmax calls on the current device found no finite value since the process started (a synchronous read
/// of a device word: call it with the device idle, never per token).  0 on a read error.
unsigned argmax_nonfinite();
/// argmax over x[0..n) with its value: out[0] the index (0x7fffffff when nothing is finite), out[1] the value's bits.
void argmax_pair(const float* x, int n, int* out, cudaStream_t s);
/// The pick of a row split in two at c0 from both halves' argmax_pair (b's index relative to c0): argmax's for the
/// whole row, ties to the lower index.
void argmax_join(const int* a, const int* b, int c0, int* out, cudaStream_t s);

}  // namespace strata::kernels::glmf
