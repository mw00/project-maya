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
    /// proposals): five or more Q4_K / Q8_0 columns then go to the tensor cores (sm_80+; STRATA_GLM_MV_APPROX=0: never).
    bool approx = false;
};
constexpr int kMaxMvCols = 16;   // a verify window's tokens, a draft block's
constexpr int kMaxMvJobs = 8;
/// One launch for up to kMaxMvJobs independent GEMVs (they share nothing but the launch).
bool mv(const MvJob* jobs, int n, cudaStream_t s);
/// The most activations mv_nt() takes at once (the conversations of one batched decode step).
constexpr int kMaxMvNt = 8;
/// mv() for nt activations at once (nt <= kMaxMvNt; 1 = mv()), the jobs' ncol 1: each job's xq / xf hold nt activations
/// one after another (n_in values each) and its y nt outputs (n_out each).  Output t is bit for bit what mv() gives
/// for activation t alone, and each weight row is read from memory once for all of them.
bool mv_nt(const MvJob* jobs, int n, int nt, cudaStream_t s);
/// Whether mv() serves this weight type.
bool mv_supported(int type);
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
    unsigned int* counter = nullptr;  // zero-initialised; the kernel leaves it zero
};
void hc(const HcArgs& a, cudaStream_t s);
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
};
void dsa_prep(const DsaPrepArgs& a, cudaStream_t s);
/// score[p] = sum_h relu(iq_h . pooled_p) * iw_h for the n_pools completed pools.
void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools,
               float* score, cudaStream_t s);
/// The stable top-k of the pools (ties by lower index) expanded to cells, then the tail cells.
void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells,
                cudaStream_t s);
/// Absorbed MLA for one token, per head: q_abs = wk_b_h . q_h, scores over the selected latents,
/// softmax, ctx, out_h = wv_b_h . ctx; writes the q8_1 of the n_head * v_head output.
/// lat_q8: the cache holds INT8 latent records (see lat8_rec_bytes), else FP16 rows of kv_lora.
void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel,
         int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8 = false);
/// The same absorbed MLA in three steps for a verify window (each step mla()'s arithmetic, bit for bit): the head-wise
/// q_abs = wk_b . q of `ncol` tokens at once (q rows q_stride floats apart, q_abs rows qa_stride), one token's attention
/// over its selected cells into ctx (n_head x kv_lora; false: not available here - use mla()), and the head-wise
/// wv_b . ctx of ncol tokens at once into their q8_1 rows (out_stride_bytes apart).  The two GEMVs read wk_b / wv_b once
/// for all the tokens, where mla() per token read them T times.
void mla_q_abs(const float* q, int64_t q_stride, const uint16_t* wk_b, int n_head, int qk_nope, int kv_lora, int ncol,
               float* q_abs, int64_t qa_stride, cudaStream_t s);
bool mla_attend(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head, int qk_nope,
                int kv_lora, float* ctx, cudaStream_t s, bool lat_q8);
void mla_out(const float* ctx, int64_t ctx_stride, const uint16_t* wv_b, int n_head, int kv_lora, int v_head, int ncol,
             void* out_q8_1, int64_t out_stride_bytes, cudaStream_t s);
/// The INT8 latent cache (STRATA_GLM_KV_INT8): per position kv_lora int8 codes, then one FP16 scale per 32 values
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
    const int* skip = nullptr;                 // set_skip's flag: non-zero = the experts' kernels do nothing
};
/// The CPU LANE's answer: the weighted sum of the RAM-tier experts the host computed for route `seq`.
struct CpuAnswer {
    volatile unsigned int seq;
    unsigned int pad[3];
    float part[4096];
};
constexpr int kRingSize = 64;
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
    unsigned int skip_mask;      // DISK SKIP: disk-only experts left out of this token (STRATA_GLM_DISK_SKIP_W / _FROM)
    unsigned long long cpu_src[8];   // ... their RAM-tier blobs
    short near_ids[16];          // STRATA_GLM_ROUTE_LOG only: the route's next ranks after the top k (-1: none)
    int win;                     // ... and the answer slot they go to: a verify window's token (0: the lone answer)
    int win_n;                   // a batched verify window's tokens (0: not one): their CPU lanes may run at once
    float x[4096];               // the FFN input, written only when cpu_mask != 0
};
/// The host's answer to a request with misses.
struct MoeResponse {
    volatile unsigned int seq;
    unsigned int ptr_mask;            // plan entries the host made resident: ptr[i] valid
    unsigned long long ptr[8];
    int n_upd;                        // table updates: tab[key] = val
    int upd_key[64];
    unsigned long long upd_val[64];
    int cpu;                          // 1: cpu_part (host-mapped, below) holds the missed experts' weighted sum
    float cpu_part[4096];
};
/// Router (sigmoid + bias, stable top-k, normalised weights - router_sigmoid's arithmetic), the plan
/// lookup in `tab`, the request publish, and the shared expert's swiglu + q8_1.
void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w,
               int layer, const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up,
               float sh_limit, int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits = nullptr,
               const float* pred_bias = nullptr, int max_prefetch = 0, const float* ahead_logits = nullptr,
               const float* const* ahead_bias = nullptr, int n_ahead = 0, int skip_from = 8,
               unsigned long long cpu_plan = 0, int promote_min = 0, int pf_rank = 0, int near_n = 0, int win = 0,
               unsigned int* scr_used = nullptr, int win_n = 0, int disk_skip_from = 8, float disk_skip_w = 0.0f);
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
bool moe_win_experts(int gu_type, int d_type, const MoeWin& w, cudaStream_t s);
/// moe_win_experts' grouping alone (w's plan_ptr, T, k -> grp_ptr / grp_start / grp_ent / n_groups), for kernels of
/// other formats (EXL3: exl3::win_experts)
void moe_win_group(const MoeWin& w, cudaStream_t s);
/// The CPU LANE's split: of f RAM-tier experts in a route, cpu_take(plan, f) go to the host (4 bits per f, f = 0..8).
inline int cpu_take(unsigned long long plan, int f) { return (int) ((plan >> (4 * f)) & 15ull); }
/// Waits (on the device) for the host's CPU-lane answer of the last route when it had CPU experts, then adds it to
/// `out` (the layer's FFN output, which moe_down wrote first - its work runs while the CPU computes).
void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s);
/// The side stream's copy of the prefetch list (pf_src -> pf_dst, pf_n entries of blob_bytes each).
void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s);
/// Waits (on the device) for the host's response when the last route had misses, then applies it.
void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s);
/// Copies every planned expert with a fetch source from the RAM tier (host-mapped) into its VRAM slot.
/// blocks > 0: that many thread blocks (a fetch on a side stream beside the expert kernels needs only a few to keep the
/// PCIe link busy); 0: one a multiprocessor.
void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s, int blocks = 0);
/// FETCH OVERLAP: the route's plan split in two (device, 8 entries each) - res = the planned experts already in VRAM
/// (no fetch source), fet = the ones the fetch kernel pulls over PCIe.  A MoeDev whose plan_ptr is res (fet) makes
/// moe_gate_up / moe_down run only those, so the resident experts compute while the fetch is still on the link.
void moe_split_plan(const MoeDev& d, int k, unsigned long long* res, unsigned long long* fet, cudaStream_t s);
/// The routed gate/up rows of the planned (resident) experts, the clamped swiglu, the q8_1 of h; and (sh_down and
/// sh_out given) the shared expert's down rows over its h in sh_hq into sh_out, by extra blocks of the same launch.
void moe_gate_up(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                 const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s);
/// out[r] = (sum_i w_i * down_i(r) . h_i over the resident experts) [+ cpu_part[r]] + sh_out[r] (when given).
/// down_off = the byte offset of the down rows inside an expert blob ([gate | up | down]).
/// part_mode (FETCH OVERLAP, part_buf = 8 x n_embd floats): 1 = only store each plan entry's down row values in
/// part_buf (0 for an entry this launch does not compute) and write nothing to out; 2 = compute this launch's
/// entries, take the others' values from part_buf, then the combine as with 0 - the same per-entry values summed in
/// the same plan order, so two launches give bit for bit what one launch gives.  0: the one launch.
void moe_down(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
              const float* sh_out, float* out, cudaStream_t s, float* part_buf = nullptr, int part_mode = 0);

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

/// The NextN block's input: cat = [rms(emb) * enorm, rms(h) * hnorm] (2 n values) as q8_1.  n <= 4096.
void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1,
            cudaStream_t s);
/// h += add (add may be null), then x = rms(h) * w and its q8_1 (a plain pre-norm residual step).  n <= 4096.
void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s);

/// argmax over n floats -> *out (device int).  Ties: lowest index.
void argmax(const float* x, int n, int* out, cudaStream_t s);

}  // namespace strata::kernels::glmf
