// src/hy4/hy4_kernels.cuh - maya-hy4's own kernels (hy4_kernels.cu): what Hy4-flash adds to the GGUF matrices that
// src/ds4/ds4_iq runs.  Every activation is F32 [rows][width] unless said otherwise; a "window" is T tokens of one
// sequence at positions pos[0..T-1] (consecutive), T <= kMaxWin on the token path, any T on the prompt path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

namespace hy4::k {

constexpr int kMaxWin = 16;   // the token / window path's tokens at most (ds4_iq's GEMVs take 16 columns)

/// int8 rows with an F32 scale a 32 values (ds4_iq's GEMV / expert input): q[r][k], d[r][k / 32]; k % 32 == 0
void q8_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s);

/// ... rows of a [T][H][...] tensor head by head: q row h T + t from x + t ldt + h ldh (k values)
void q8_heads(const float* x, size_t ldt, size_t ldh, int T, int H, int k, int8_t* q, float* d, cudaStream_t s);
/// dst[r * ldd + i] = src[r * lds + i], i < n (rows r < rows)
void copy_cols(const float* src, size_t lds, float* dst, size_t ldd, int rows, int n, cudaStream_t s);
/// ring[pos[t] % R] = src[t] (rows of n floats)
/// (f16: the ring holds F16 values - --kv q8's indexer rows -, n / 2 floats a row)
void store_rows(const float* src, int T, int n, float* ring, int R, const int* pos, cudaStream_t s, bool f16 = false);
/// y[c * ldy + r] = sum_i W[r][i] x[c * ldx + i], W F32 or BF16 rows of k (ld elements apart),
/// c < ncol (any), r < n
void gemv_f32w(const float* W, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy,
               float alpha, cudaStream_t s);
void gemv_bf16w(const uint16_t* W, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy,
                float alpha, cudaStream_t s);

/// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) (* w), rows of n (x and y may be the same)
void rmsnorm(const float* x, size_t ldx, int rows, int n, const float* w, float eps, float* y, size_t ldy, cudaStream_t s);

/// The hyper-connections (iHC: Hy4's, without the comb term).  x [T][hc][E].
///   pre:  mixes = fn [2 hc][hc E] . rmsnorm(x flat) (no weight, eps);  pre = sigmoid(mix[0:hc] scale[0] + base[0:hc]) +
///         hc_eps;  post = sigmoid(mix[hc:] scale[1] + base[hc:]) mag + hc_eps;  cur[t] = sum_s pre[s] x[t][s]
///   head: the same with fn [hc][hc E], scale[0], base[hc] and no post
void hc_pre(const float* x, int T, int hc, int E, const float* fn, const float* scale, const float* base, float eps,
            float hc_eps, float mag, float* cur, float* post, cudaStream_t s, const float* norm_w = nullptr,
            float norm_eps = 0.0f, bool fn_by_element = false);
/// (fn_by_element, both: fn laid out by element - [hc E][its rows], the rows a multiple of 4 - not by row: an element's
///  weights side by side, read in 16-byte loads; the same sums)
void hc_head(const float* x, int T, int hc, int E, const float* fn, const float* scale, const float* base, float eps,
             float hc_eps, float* cur, cudaStream_t s, bool fn_by_element = false);
/// hc_pre's and hc_head's buffers for launches of up to T rows, made on the current device (at the load: a buffer grown
/// inside a forward is freed under a lock every GPU's thread takes)
void hc_reserve(int T);
/// x[t][s] += y[t] post[t][s]
void hc_post(float* x, int T, int hc, int E, const float* y, const float* post, cudaStream_t s);
/// ... with y's RMS norm (weight w, eps) first: the layer's post norm and the post-mix in one pass (yo: the normed y
/// written there too, null: not)
void hc_post_norm(float* x, int T, int hc, int E, const float* y, const float* post, const float* w, float eps,
                  cudaStream_t s, float* yo = nullptr);
/// (hc_pre with norm_w: cur is the RMS norm of the mix with that weight - the layer's attn_norm / ffn_norm folded in)

/// RoPE (ggml's NORM: consecutive pairs) in place over dims [off, off + n_rot) of every head: x[t][h][ld], h < nh,
/// theta = pos[t] base^(-i / n_rot)
void rope(float* x, int T, int nh, size_t ld, size_t ldt, int off, int n_rot, const int* pos, float base, cudaStream_t s);

/// The latent cache rows of a window: kv [T][kv_lora + rope] (the kv_a output) -> cache row slot(pos) as F16: the first
/// kv_lora normed (w, eps), the rest rotated at pos (base); slot = pos % ring (ring 0: pos).  q8: the row in 8 bits
/// instead - the kv_lora values as int8 in blocks of 32 (ggml's Q8_0 rounding), the blocks' F16 scales, then the
/// rotated part's values in F16 - kv_q8_row_bytes a row (512 + 64 values: 672 bytes, against F16's 1152)
size_t kv_q8_row_bytes(int kv_lora, int n_rot);
/// KV streaming (an indexed layer's rows beyond the VRAM): the VRAM cache a ring whose slot p % ring holds position
/// tag[slot] (-1: none), and every position's row in host memory as well (rows: pinned and mapped, a row a position in
/// the cache's own format).  A reader takes a row from VRAM where the tag is the position, else from the host - the
/// same bits either way.  tag nullptr: off.
struct KvHost {
    int* tag = nullptr;
    uint16_t* rows = nullptr;
    // KV spill (the rows past what the RAM holds on a drive): hidx [positions] (device) the host slot of each position
    // (-1: on the drive only, -2: claimed by kv_need), the newest hres positions in slots p % hres (htag [hres],
    // device: the position each holds), the slots past them the rows a read brought from the drive (kv_need, then the
    // host, then kv_hidx).  hidx nullptr: position p's row at p, as before
    int* hidx = nullptr;
    int* htag = nullptr;
    int hres = 0;
};
/// KV spill: the positions a read takes that are neither in the VRAM ring nor in host memory, each once (hidx -1 ->
/// -2), into miss [cap] (host memory, mapped) with their count in mcount (device; the caller zeroes it); the slots
/// past the newest tier the read takes get fgen [slot - hres] = gen (host memory, mapped: kept while the host makes
/// room).  The positions: keys (T rows of ldk, nk[t] each) when given, else list's pools (n of kpool positions, those
/// below kend), else the span [k0, k0 + n).
void kv_need(const int* keys, int ldk, const int* nk, int T, const int* list, int kpool, int kend, int k0, int n,
             int ring, const KvHost& hs, int* fgen, int gen, int* miss, int* mcount, int cap, cudaStream_t s);
/// ... then hidx[set[2i]] = set[2i + 1] (n_set rows the host put in slots) and hidx[drop[2i]] = -1 where it still is
/// drop[2i + 1] (the slots given back); set and drop in host memory (mapped)
void kv_hidx(const int* set, int n_set, const int* drop, int n_drop, int* hidx, cudaStream_t s);
void kv_store(const float* kv, int T, int kv_lora, int n_rot, const float* w, float eps, const int* pos, float base,
              uint16_t* cache, int ring, cudaStream_t s, bool q8 = false, const KvHost& hs = KvHost{});
/// The picked rows of a sub-batch side by side: dst[t * ldk + j] = the cache row (row_bytes bytes, F16 or 8-bit: as
/// stored) of position keys[t * ldk + j], j < nk[t] - the attention then reads dst as a cache of consecutive rows
/// (lo[t] = t * ldk), each row fetched once whichever heads read it.
void kv_pick(const uint16_t* cache, int ring, size_t row_bytes, const int* keys, int ldk, const int* nk, int T,
             uint16_t* dst, cudaStream_t s, const KvHost& hs = KvHost{});
/// ... each picked position's row once for the whole sub-batch (its tokens pick much the same rows): dst the rows of
/// the positions picked, in no order, and keys2[t * ldk + j] the row there of keys[t * ldk + j] - the attention reads
/// dst through keys2, the same rows in the same order.  slot [positions] is -1 everywhere before and after (the
/// caller's, set once); count [1] and list [T * ldk] are scratch.
void kv_pick_union(const uint16_t* cache, int ring, size_t row_bytes, const int* keys, int ldk, const int* nk, int T,
                   int* slot, int* count, int* list, int* keys2, uint16_t* dst, cudaStream_t s,
                   const KvHost& hs = KvHost{});

/// MLA attention of a window over the latent cache: q [T][H][kv_lora + rope] (the absorbed q_nope, then the rotated
/// q_pe), the keys of token t either the positions keys[t][0 .. nk[t]) (keys non-null, ldk apart) or the range
/// [lo[t], lo[t] + nk[t]); cache row of position p: p % ring (ring 0: p).  out [T][H][kv_lora] = softmax(scale q.k,
/// with the head's sink) . v (v the row's first kv_lora values).  ws: the partial results' scratch (attn_ws_floats).
size_t attn_ws_floats(int T, int H, int kv_lora, int max_keys);
/// (q8: the cache's rows are kv_store's Q8_0 rows, dequantized as they are read)
void attention(const float* q, int T, int H, int kv_lora, int n_rot, const uint16_t* cache, int ring, const int* keys,
               int ldk, const int* lo, const int* nk, int max_keys, float scale, const float* sinks, float* ws,
               float* out, cudaStream_t s, bool q8 = false);
/// The same over keys[t][0 .. nk[t]) on an RDNA GPU's matrix cores (gfx11; HY4_ATTN_WMMA=0 off, =1 also gfx12): a
/// prompt's tokens past the dense span - within rounding of attention's sums (F16 products of a split query and
/// split probabilities, in another order), so never a token window's.  false: not taken (another GPU, H not a
/// multiple of 16, no keys) - attention then.  attn_wmma_on: whether this device takes it.
bool attn_wmma_on();
bool attention_wmma(const float* q, int T, int H, int kv_lora, int n_rot, const uint16_t* cache, int ring,
                    const int* keys, int ldk, const int* nk, float scale, const float* sinks, float* out,
                    cudaStream_t s, bool q8 = false);

/// x *= sigmoid(g), n values
void sigmoid_mul(float* x, const float* g, size_t n, cudaStream_t s);
/// h[r][i] = silu(g[r][i]) u[r][i] (g, u: rows ldg apart; h ldh apart)
void swiglu(const float* g, const float* u, size_t ldg, int rows, int n, float* h, size_t ldh, cudaStream_t s);
/// y += x, n values
void add(float* y, const float* x, size_t n, cudaStream_t s);
/// y += x w (each value by its own weight), n values
void addmul(float* y, const float* x, const float* w, size_t n, cudaStream_t s);

/// The router: logits [T][n_exp] -> sigmoid, top n_used by sigmoid + bias (bias may be null), weights = the picks'
/// sigmoids (/ their sum when norm) * scale -> ids [T][n_used], w [T][n_used]
void router(const float* logits, int T, int n_exp, const float* bias, int n_used, bool norm, float scale, int* ids,
            float* w, cudaStream_t s);
/// A window's expert entries grouped by expert (ds4_iq's grouped Args): entry e = t * 16 + j (j < n_used) of ids;
/// grp_ptr[g] = base + expert * stride, grp_start[g .. g + 1] its entries in grp_ent, *n_groups
/// slot (null: none): each expert's slot among the VRAM ones (base + slot stride), -1 for the CPU lane's - those entries
/// are left out and their weight w[t][j] zeroed (the lane adds their part)
/// (eptr, the GPU fetch's: each expert's blob wherever it lies - a VRAM slot, a staging slot, pinned RAM - 0 for none
///  of this GPU's; who [T][n_used], when given: 1 for an entry the CPU lane computes - left out, its weight zeroed -
///  so one expert may run here for one token and on the lane for another; slot is not read then)
void group_experts(const int* ids, int T, int n_used, int n_exp, const uint8_t* base, size_t stride,
                   unsigned long long* grp_ptr, int* grp_start, int* grp_ent, int* n_groups, cudaStream_t s,
                   const int* slot = nullptr, float* w = nullptr, const unsigned long long* eptr = nullptr,
                   const int* who = nullptr);
/// ... a token's alone (ds4_iq's plan form): plan[j] = base + ids[j] * stride
void plan_experts(const int* ids, int n_used, const uint8_t* base, size_t stride, unsigned long long* plan,
                  cudaStream_t s, const int* slot = nullptr, float* w = nullptr,
                  const unsigned long long* eptr = nullptr, const int* who = nullptr);

/// The GPU fetch (a token window's routed experts that VRAM does not hold, computed by this GPU from a copy of the
/// layer's experts in pinned RAM): one layer's device state and a GPU's scratch for it
struct FetchLayer {
    int* slot = nullptr;         // [n_exp] each expert's VRAM slot (-1: none) - the layer's slot table, kept current
    int* owner = nullptr;        // [n_res] each slot's expert
    float* score = nullptr;      // [n_exp] each expert's decayed picks (the cache's: a miss takes the slot of the
                                 //  lowest scored expert not picked in the window when it scores higher)
    const unsigned long long* hptr = nullptr;   // [n_exp] each expert's blob in pinned RAM, as this GPU addresses it
    uint8_t* vram = nullptr;     // the layer's VRAM slots
    size_t blob = 0;
    int n_res = 0, n_exp = 0;
};
struct FetchScratch {
    unsigned long long* eptr = nullptr;   // [n_exp] (out) the window's experts' blobs where this GPU reads them
    int* who = nullptr;                   // [kMaxWin][16] (out) 1: the entry is the CPU lane's
    unsigned long long *src = nullptr, *dst = nullptr;   // [kMaxWin x 16] (out) the copies to make, *nf of them
    unsigned* nf = nullptr;
    uint8_t* stage = nullptr;             // n_stage staging slots (blob apart) for misses the cache does not take
    int n_stage = 0;
    unsigned long long* stats = nullptr;  // [8] (summed) picks, VRAM hits, copied, read in place, the lane's, admitted
};
/// The window's picks resolved: k_gpu < 0 - every entry this GPU's (a miss copied into the slot of a colder expert
/// when `cache`, else into a staging slot, else read in place); k_gpu >= 0 - the VRAM experts and each token's first
/// k_gpu others (in pick order) this GPU's, copied into staging slots, the rest the CPU lane's (who); the slots kept
/// (no cache).  decay: the scores' fading a token.  One block.
/// ahead: the picks are a guess at a later layer's (its router on this layer's input) - only the misses the cache
/// takes are listed (copied ahead of the layer), nothing else is written, the scores are left as they are
void fetch_resolve(const int* ids, int T, int n_used, const FetchLayer& fl, const FetchScratch& fs, int k_gpu,
                   bool cache, float decay, cudaStream_t s, bool ahead = false);
/// The copies fetch_resolve listed (pinned RAM -> VRAM, blob bytes each) by `blocks` blocks
void fetch_copy(const FetchScratch& fs, size_t blob, int blocks, cudaStream_t s);
/// out[t] = sum_j w[t][j] part[t * 16 + j] (part rows of n; the plan form: part[j], T 1) - entries of weight 0 skipped
void moe_sum(const float* part, int T, int n_used, int n, const float* w, float* out, cudaStream_t s);
/// Duplicate each row: dst[2 r] = dst[2 r + 1] = src[r] (int8 rows of k with their scales: the grouped form's input
/// rows, two a token: entry e reads row e >> 3)
void dup_rows_q8(const int8_t* q, const float* d, int rows, int k, int8_t* q2, float* d2, cudaStream_t s);

/// Engram.  rows: the window's looked-up rows, F8 E4M3 [T][cols][dim]; valid[t] 0 / 1 -> r [T][cols dim] = each row
/// RMS-normed (eps) times lookup_norm [cols][dim], times valid.  scale [T][cols] (or none): each row's own scale, its
/// values the codes times it (a table of per-row scaled codes with its scales beside it)
void engram_rows(const uint8_t* rows, const float* valid, int T, int cols, int dim, const float* lookup_norm, float eps,
                 float* r, cudaStream_t s, const float* scale = nullptr);
/// kv [T][(hc + 1) E] (the fused key / value projection, without its bias kv_b [(hc + 1) E]) -> x[t][s] += delta, for
/// the tokens with valid[t]: delta[s][d] = rms(value + b)[d] gamma[d] (gate[s] + hc_scale[s][d]),
/// gate[s] = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), dot = sum_d rms(x[s])[d] rms(key[s])[d] egate[s][d] / sqrt(E)
void engram_mix(float* x, const float* kv, const float* kv_b, int T, int hc, int E, const float* egate,
                const float* gamma, const float* hc_scale, const float* valid, float eps, cudaStream_t s);

/// The indexer's pooled keys.  ivg ring [R][2 dim] (each token's value | gate row at slot pos % R): the groups g of
/// [g0, g0 + n) -> pooled[g] = rope(rmsnorm(sum_k softmax_k(gate_k + ape[k]) value_k) w) at position kpool g
/// (f16: pooled and the ivg ring hold F16 values - --kv q8's -, their rows half as many floats apart)
void ix_pool(const float* ivg, int ring, int g0, int n, int kpool, int dim, const float* ape, const float* w, float eps,
             int n_rot, float base, float* pooled, cudaStream_t s, bool f16 = false);
/// Each token's scores over the pools it sees (g < nv[t] = pos[t] / kpool): score[t][g] = sum_h relu(q[t][h] . pk[g])
/// wt[t][h]; -inf beyond.  q [T][heads][dim], wt [T][heads]; score rows ld apart, n_pool columns
void ix_score(const float* q, const float* wt, int T, int heads, int dim, const float* pooled, const int* pos, int kpool,
              int n_pool, float* score, size_t ld, cudaStream_t s, bool f16 = false);
/// ... on an RDNA GPU's matrix cores (attn_wmma_on's devices; 16 heads of 128; HY4_IX_WMMA=0 off): the pools rounded
/// to F16, the products of a split query, in another order - the prompt path past the exact span alone.  false: not
/// taken (ix_score then)
bool ix_score_wmma(const float* q, const float* wt, int T, int heads, int dim, const float* pooled, const int* pos,
                   int kpool, int n_pool, float* score, size_t ld, cudaStream_t s, bool f16 = false);
/// ... from the heads' dot products of a GEMM (the prompt path): S[(t heads + h) ldS + j] = q[t][h] . pk[g0 + j],
/// j < n -> score[t][g0 + j] as ix_score's (the same sum over the heads; -inf from nv[t] on)
void ix_reduce(const float* S, size_t ldS, const float* wt, int T, int heads, const int* pos, int kpool, int g0, int n,
               float* score, size_t ld, cudaStream_t s);
/// The attention keys of each token: the top min(nv, top) of its scores' pools (their kpool positions each), then its
/// own group's positions up to itself -> keys[t][0 .. nk[t]), ldk apart
void ix_select(const float* score, size_t ld, int T, const int* pos, int kpool, int top, int* keys, int ldk, int* nk,
               cudaStream_t s);
/// Tensor parallel's exchange of those picks: ix_pack writes the min(nv, top) picked pools of T tokens as 16-bit
/// numbers into wire (mapped host memory, rows ldw apart - a multiple of 8) and fences them for the host; ix_unpack
/// makes keys[t][0 .. nk[t]) again from such rows (read past every cache) - ix_select's own keys, the tail its own
void ix_pack(const int* keys, int ldk, int T, const int* pos, int kpool, int top, uint16_t* wire, int ldw,
             cudaStream_t s);
void ix_unpack(const uint16_t* wire, int ldw, int T, const int* pos, int kpool, int top, int* keys, int ldk, int* nk,
               cudaStream_t s);

/// The prompt path's attention as two GEMMs (the model's cuBLAS calls) and this between them: S [T H][nkr] (rows ldS
/// apart) the scores of tokens t (positions pos[t]) and heads h against the keys at positions k0 .. k0 + nkr - 1 ->
/// P (F16, ldP) = softmax(scale S, the head's sink) over the keys the token sees: window > 0 the last `window`
/// positions up to its own, else mask[t][j] == 0 (the indexer's picks, dsa_mask); 0 elsewhere
/// (window < 0: the mask alone - the keys a list, not the positions from k0: dsa_mask_list's)
void attn_softmax(const float* S, size_t ldS, int T, int H, int nkr, int k0, const int* pos, int window,
                  const float* mask, size_t ldm, float scale, const float* sinks, uint16_t* P, size_t ldP,
                  cudaStream_t s);
/// mask[t][j] = 0 for every key position k0 + j among keys[t][0 .. nk[t]), -inf for the others
void dsa_mask(const int* keys, int ldk, const int* nk, int T, int k0, int nkr, float* mask, size_t ldm, cudaStream_t s);
/// dst[j] = the latent cache row of position k0 + j (row p % ring), j < n: a ring's rows in position order
/// (list: row j the position list[j / kpool] kpool + j % kpool instead - the rows of a list of pools, zeros for the
///  positions from kend on; kv_deq's too)
void kv_gather(const uint16_t* cache, int ring, int row, int k0, int n, uint16_t* dst, cudaStream_t s,
               const int* list = nullptr, int kpool = 1, int kend = 0, const KvHost& hs = KvHost{});
/// ... from a q8 cache (kv_store's 8-bit rows): dst[j] the F16 row of position k0 + j
void kv_deq(const uint16_t* cache, int ring, int kv_lora, int n_rot, int k0, int n, uint16_t* dst, cudaStream_t s,
            const int* list = nullptr, int kpool = 1, int kend = 0, const KvHost& hs = KvHost{});
/// The pools the keys of T tokens name (ix_select's keys: whole pools, kpool positions each, then the token's own
/// pool up to itself) among [0, n_pool): col[g] = pool g's rank among them in index order (-1: not named),
/// list[rank] = g, *n = how many
void ix_union(const int* keys, int ldk, const int* nk, int T, int kpool, int n_pool, int* col, int* list, int* n,
              cudaStream_t s);
/// dsa_mask over the columns of such a list (pool g's kpool positions at columns col[g] kpool ..): ncol columns
void dsa_mask_list(const int* keys, int ldk, const int* nk, int T, const int* col, int kpool, int ncol, float* mask,
                   size_t ldm, cudaStream_t s);
/// The prompt path.  dst[r][i] = F16(src[r * lds + i]), i < n
void to_f16(const float* src, size_t lds, int rows, int n, uint16_t* dst, size_t ldd, cudaStream_t s);
/// dst[i] = the F32 of BF16 src[i] (exact), n values
void bf16_to_f32(const uint16_t* src, float* dst, size_t n, cudaStream_t s);
/// A chunk's expert entries sorted by expert for ds4_iq's sets: entry e = t n_used + j of ids; bounds[x .. x + 1] the
/// sorted rows of expert x (n_exp + 1 values), tok[r] the token of sorted row r, row_of[e] the sorted row of entry e
/// (slot non-null: the bounds over the n_slots slots, an entry without one left out - row_of -1, its weight zeroed when
/// w is given)
void moe_sort(const int* ids, int T, int n_used, int n_exp, int* bounds, int* tok, int* row_of, cudaStream_t s,
              const int* slot = nullptr, float* w = nullptr, int n_slots = 0);
/// out[t] = sum_j w[t][j] y[row_of[t n_used + j]] (rows of n; row_of -1: skipped)
void moe_combine(const float* y, const int* row_of, const float* w, int T, int n_used, int n, float* out,
                 cudaStream_t s, bool accumulate = false);
/// Tensor parallel (hy4_tp.hpp): the members' partial sums through pinned host memory.  publish: n floats of src to
/// dst (mapped host memory), then *flag = v once every block's are written (counter: a device word of zero, the
/// blocks' tally); wait_pull: once *flag (mapped) reaches v (wrapping compare), n floats of src (mapped) into dst;
/// wait_flag: the wait alone; set_flag: *flag = v after the stream's work before it.  The waits give up after a minute
/// (an error: errors() counts it), so no device spins for good.
void publish(const float* src, float* dst, size_t n, unsigned* counter, unsigned* flag, unsigned v, cudaStream_t s);
/// AMD under Windows (HIP): the runtime holds queued launches and copies back until the host waits on the GPU - the
/// work queued on `s` submitted now (cudaStreamQuery, which does not wait), as maya-ds4's and maya-mm3's own.  Where
/// the host or another GPU waits for a flag this stream sets, or the host works beside the GPU, the work must not sit
/// in the queue.  Elsewhere (Linux, and NVIDIA on Windows) nothing.  HY4_WIN_SUBMIT=0: off (A/B tests)
void submit(cudaStream_t s);
void wait_pull(const unsigned* flag, unsigned v, float* dst, const float* src, size_t n, cudaStream_t s);
void wait_flag(const unsigned* flag, unsigned v, cudaStream_t s);
void set_flag(unsigned* flag, unsigned v, cudaStream_t s);
/// dst = a (na words) | b (nb) | c (nc), 4-byte words; split3 the other way
void pack3(const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc, void* dst, cudaStream_t s);
void split3(const void* src, void* a, size_t na, void* b, size_t nb, void* c, size_t nc, cudaStream_t s);
/// A prompt chunk's sums in F16 (tensor parallel): each row of E values to F16 times 2^-s (s >= 0, the smallest that
/// brings its largest under 2^15: no overflow), the factor 2^s in sc[row]; unpack16 back (h * sc)
void pack16_rows(const float* x, int rows, int E, uint16_t* h, float* sc, cudaStream_t s);
void unpack16_rows(const uint16_t* h, const float* sc, int rows, int E, float* y, cudaStream_t s);
/// x[t][s] = e[t] for every stream s < hc (the window's embeddings into the hyper-connections' streams)
void expand_streams(const float* e, int T, int hc, int E, float* x, cudaStream_t s);
/// The waits that gave up since the start (this device's)
unsigned wait_timeouts();
/// A row's top token and its probability: out[0] = the argmax (ties: the lower id, as an int's bits), out[1] =
/// softmax(row)[argmax]
void top1(const float* row, int n, float* out, cudaStream_t s);
/// ... of a part of a row (n values): out[0] its top value, out[1] that one's index (as bits), out[2] the sum of
/// exp(v - top) over the part, out[3] 0 - the row's top token and its probability are made of its parts' on the host
void top1_part(const float* row, int n, float* out, cudaStream_t s);
/// rows of n values, ld apart: each row's largest value's index (of equals the lowest, as a walk from the first) plus
/// i0 at out[2 r] (as bits) and the value at out[2 r + 1] - the index -1 where the row has a NaN or nothing above -inf
void row_top(const float* x, int rows, int n, size_t ld, int i0, float* out, cudaStream_t s);
/// (tests) hc_pre writes its mixes [T][2 hc] here too (null: not)
void set_mix_out(float* p);
/// A launch failure since the last call (printed); 0 none
int errors();
/// The kernel of the first launch that failed in this process (empty: none).  A kernel's fault (an illegal memory
/// access) is sticky and reported at a later launch: CUDA_LAUNCH_BLOCKING=1 names the kernel that faulted
std::string first_fault();

}  // namespace hy4::k
