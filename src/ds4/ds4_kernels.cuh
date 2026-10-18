// src/ds4/ds4_kernels.cuh - DeepSeek-V4.1's own device kernels for a window of T tokens (1 <= T <= kWin) at positions
// p0 .. p0 + T - 1 (the EXL3 matrices go through strata::kernels::exl3).  The math is DeepSeek's inference/model.py
// (MIT); tools/ds4/ref.py is the oracle the engine is checked against.  Every pointer is device memory, every activation
// F32, rows of a window back to back; the KV caches are BF16 (the values the reference keeps there are FP8 / FP4
// roundings, exact in BF16).  What differs from query to query (its window, how much of the compressed KV it sees, how
// many rows it selects) is known on the host and passed by value.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4 {

constexpr int kWin = 16;      // tokens a window takes at most (the EXL3 GEMV's activations per job)
constexpr int kPicks = 8;     // routed experts a token can take at most

/// h[t][s][e] = BF16 rows[t][e] for each of the hc streams.
void embed(const uint16_t* rows_bf16, int T, float* h, int dim, int hc, cudaStream_t s);

/// The per-vector finish: n_vec vectors of len values at x + v * stride, each in order
///   norm  (w != null): x = x * rsqrt(mean(x^2) + eps) * w
///   rope  (rd > 0): the last rd values as adjacent complex pairs rotated by pos_v * freqs[i] (inverse: the conjugate),
///         pos_v = pos + (v / vpp) * pos_step
///   quant 0 none, 1 FP8 E4M3 with a power-of-two scale per 32, 2 FP4 E2M1 with an E4M3 scale per 16, 3 FP4 with a
///         power-of-two scale per 32 (the reference kernels' act_quant / fp4_act_quant, inplace)
/// and written back to x, or as BF16 to dst + v * dst_stride when dst is given.
struct Finish {
    const float* w = nullptr;
    float eps = 1e-20f;
    int rd = 0, pos = 0, inverse = 0, vpp = 1, pos_step = 1;
    const float* freqs = nullptr;   // rd / 2 values
    int quant = 0;
    uint16_t* dst = nullptr;
    long long dst_stride = 0;
};
void finish(float* x, int n_vec, int len, long long stride, const Finish& f, cudaStream_t s);

/// Hyper-Connections, row t of each [T][..] array: mixes[t][m] = (h_flat . fn[m]) * rsqrt(mean(h_flat^2) + eps), then
/// pre / post / comb as hc_split_sinkhorn (comb[t][j * hc + k], j the source stream, k the destination).  Strides: mixes
/// 32, pre / post 8, comb 64 floats a row.
void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s);
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb, cudaStream_t s);
/// x[t][e] = sum_s mix[t][s] h[t][s][e], then x = rmsnorm(x) * w (w null: no norm).
void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 cudaStream_t s);
/// out[t][k][e] = post[t][k] y[t][e] + sum_j comb[t][j][k] res[t][j][e]  (out must not alias res).
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, cudaStream_t s);
/// hc_post with out = res (h), the same arithmetic: a chunk keeps one copy of its streams.
void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T,
                     cudaStream_t s);
/// A chunk's hc_mix: the dots of each row with the n_mix weights already in mixes (a GEMM's, rows 32 apart), times
/// rsqrt(mean(h_row^2) + eps) - hc_mix's value up to the dots' summation order.
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s);

/// y[t][r] = alpha * sum_k W[r][k] x[t][k], W F16 bits [n][k] (the router's and the indexer's weights_proj).
void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, cudaStream_t s);

/// The router, row t: scores = sqrt(softplus(logits[t])) (n experts, rows 1024 apart), the top `k` of scores + bias;
/// weights from the unbiased scores, normalized to sum 1 (+1e-20) and times route_scale -> idx[t][i], w[t][i] (rows
/// kPicks apart).  Rows of image tokens (bit t of vl) take bias_vl instead (training's noaux_tc_for_vl).
constexpr int kVlRows = 8192;
struct VlBits {
    uint32_t w[kVlRows / 32];
};
void route(const float* logits, const float* bias, int n, int k, int T, float route_scale, int* idx, float* w,
           cudaStream_t s, const float* bias_vl = nullptr, const VlBits* vl = nullptr);

/// h = silu(min(g, lim)) * clamp(u, -lim, lim)  (lim <= 0: no clamp); n values.
void swiglu(const float* g, const float* u, int n, float lim, float* h, cudaStream_t s);

/// The ratio-r compressor over a window: token t (position p0 + t) takes slot (p0 + t) % r of the state; each token that
/// completes a group writes the group's pooled latent (softmax over the slots, per value, then rmsnorm(w)) to the next
/// row of latent.
void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, cudaStream_t s);

/// Per-query sizes, by value - or (gen > 0, a chunk of any length) made on the device from the query's position: query
/// t at position p0 + t has nn = (p0 + t + 1) / R compressed positions and cb = min(ccap, ceil(nn / blk)) candidate
/// blocks;  gen 1: n = nn, k = min(kcap, nn)  (the indexer's scores and their top-k)
///          gen 2: n = nn, k = cb             (index_score over the candidates)
///          gen 3: n = cb * blk, k = min(kcap, nn)   (the top-k of the candidates' scores)
///          gen 4: n = ceil(nn / blk), k = kcap      (the candidate blocks' top-k)
struct Rows {
    int n[kWin];
    int k[kWin];
    int gen = 0, R = 1, kcap = 0, ccap = 0, blk = 1;
    long long p0 = 0;
};

/// The indexer's scores, row t: out[t][i] = sum_h relu(q[t][h] . k[p]) w[t][h] for i < n (the widest query's), p = i, or
/// with candidates p = cand[t][i / blk] * blk + i % blk for i < lim.k[t] * blk; positions >= lim.n[t] (or past the
/// query's candidates) -> -inf.  q rows heads * hd apart, w rows 64, cand rows cand_stride, out rows out_stride.
void index_score(const float* q, const float* w, const uint16_t* k_bf16, int heads, int hd, int T, int n,
                 const int* cand, int cand_stride, int blk, const Rows& lim, float* out, long long out_stride,
                 cudaStream_t s);
/// The top r.k[t] of vals[t][0, r.n[t]) (row t at vals + t * stride): their positions (unordered; with candidates
/// cand[t][i / blk] * blk + i % blk) into out[t][0, k).  r.k[t] >= r.n[t] takes them all.
void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, cudaStream_t s);
/// Level one of the candidate pre-filter, row t: block b's score = the max over its blk positions of the r.n[t] the
/// query sees; the block of its newest position is pinned; the top r.k[t] blocks -> out[t] (bm: scratch rows bm_stride).
void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, cudaStream_t s);

/// Sparse attention, query t: heads x d values of q[t] against win rows win0[t] .. win0[t] + nwin[t] - 1 of the window
/// buffer (d BF16 each) and the nsel[t] compressed rows ckv[sel[t][i]]; K = V, a learned sink logit per head.
struct AttnRows {
    int win0[kWin], nwin[kWin], nsel[kWin];
    // gen: query t is a chunk's row t0 + t at position p0 + t0 + t (the window buffer: the ring's win rows before p0,
    // then the chunk's), selecting min(kcap, (its position + 1) / R) compressed rows when R > 0
    int gen = 0, t0 = 0, win = 0, R = 0, kcap = 0;
    long long p0 = 0;
};
void sparse_attn(const float* q, const uint16_t* win, const uint16_t* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                 cudaStream_t s);
/// The window buffer: ext[r] = ring[(p0 - n + r) % n] for the n rows before p0 (zeros before position 0); afterwards
/// ring[(p0 + t) % n] = ext[n + t] for t < T.
void ring_to_ext(const uint16_t* ring, uint16_t* ext, int64_t p0, int n, int d, cudaStream_t s);
void ext_to_ring(const uint16_t* ext, uint16_t* ring, int64_t p0, int n, int T, int d, cudaStream_t s,
                 uint16_t* backup = nullptr);
/// A speculative window's rollback: ring[(p0 + t) % n] = backup[t] for keep <= t < T (the rows ext_to_ring replaced).
void ring_restore(uint16_t* ring, const uint16_t* backup, int64_t p0, int n, int keep, int T, int d, cudaStream_t s);

/// Engram.  rows: n FP8 E4M3 rows of hd values, scales n x (hd / 32) E8M0 -> out F32 [n * hd].
void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, cudaStream_t s);
/// The same over the host's per-token layout (a chunk): token t's cols rows, then their scales, `per` bytes a token;
/// T tokens -> out F32 [T * cols][hd].
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, cudaStream_t s);
/// h[t][k] += gate[t][k] * value[t] with the gate from the normalized dot of h[t][k] * w[k] * key[t][k]
/// (kv[t] = [key hc x dim | value dim]).
void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, cudaStream_t s);

/// out[r] = x[src[r]] for r < n (rows of cols floats).
struct Gather {
    int src[kWin * kPicks];
};
void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, cudaStream_t s);
/// The MoE's sum, token t: out[t] = sum_j w[t][j] ed[row[t][j]] (its VRAM experts, in pick order) + cpu[t] (when its
/// cpu flag is set) + sh[t].
struct Combine {
    int n[kWin];
    int row[kWin][kPicks];
    float w[kWin][kPicks];
    int cpu[kWin];
};
void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out,
             cudaStream_t s);

/// The mailboxes of a small window (decode, a draft's verification, the drafter's block): host-mapped memory, a ring of
/// kRing.  The device publishes a layer's routing and its FFN inputs for the window's T tokens (seq written last, after
/// a system fence); the host answers with the CPU lane's weighted part of each token that took a CPU-lane expert.
constexpr int kRing = 64;
constexpr int kMaxDim = 8192;
struct Req {
    unsigned seq;
    int T;
    int pad[2];
    int ids[kWin][kPicks];
    float w[kWin][kPicks];
    float x[kWin][kMaxDim];
};
struct Resp {
    unsigned seq;
    int pad[3];
    int cpu[kWin];   // 1: part[t] is the CPU lane's weighted sum for token t
    float part[kWin][kMaxDim];
};
/// route() for the T tokens of a window, plus: plan_ptr[t][i] = vram[idx[t][i]] (0: not on the device), each token's x
/// (dim floats) and picks into the request slot, then T and seq (req null: no request, the plan alone).
void route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl = nullptr,
                   unsigned vl = 0);
/// Waits (one thread, on the device) until *seq_ptr == seq.
void wait_seq(const unsigned* seq_ptr, unsigned seq, cudaStream_t s);
/// A chunk's routed experts as sorted rows: the picks (t, i < k) of T tokens (idx rows kPicks apart) by the virtual
/// slot vs[expert] (n_vs of them: the VRAM slots, then the experts streamed in), each slot's rows in token order:
/// bounds[n_vs + 1], row_tok[r], pos[t][i] (the pick's sorted row, rows kPicks apart).
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos,
              cudaStream_t s);
/// acc[t] += w[t][i] * y[pos[t][i] - r0] for each pick of t (in pick order) whose sorted row is in [r0, r1).
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim,
                 cudaStream_t s);
/// rows[t][0] = 1, the rest 0 (the first layer's pre-mix: stream 0).
void init_pre(float* pre, int T, cudaStream_t s);

/// out[t][e] = mean over the hc streams of h[t][s][e] (the DSpark drafter's taps), rows out_stride floats apart.
void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, cudaStream_t s);
/// out[t][i] = i for i < n, rows `stride` apart (the drafter's block rows as every query's selection).
void iota_rows(int* out, int T, int n, int stride, cudaStream_t s);
/// The DSpark markov head: logits[v] += E[*tok] . H[v] for v < V (E BF16 [V][R], H F16 [V][R]).
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, cudaStream_t s);
/// The DSpark confidence: out[t] = sigmoid(w . [x[t] | E[tok[t]]]) (x rows dim floats, w F32 [dim + R]).
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out,
                cudaStream_t s);
/// Speculative sampling: p[t] = softmax(logits[t] / temp)[draft[t]] for t < n (rows V apart).
void draft_probs(const float* logits, const int* draft, int n, int V, float temp, float* p, cudaStream_t s);

/// The pipeline's hand-offs and tensor parallelism, through host-mapped memory (these cards copy peer to peer only
/// through the driver's staging).  xfer_put copies up to 4 segments (16-byte multiples, 16-byte aligned) device ->
/// host-mapped, then (flag given) sets *flag = seq after a system fence (cnt: a device counter of the stream, 0
/// between uses); xfer_get waits on the device until *flag >= seq, then copies host-mapped -> device.
struct Xfer {
    const void* src[4] = {nullptr, nullptr, nullptr, nullptr};
    void* dst[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t bytes[4] = {0, 0, 0, 0};
    int n = 0;
};
void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, cudaStream_t s);
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, cudaStream_t s);
/// Waits (one thread) until flags[i * stride] >= seq for i < n (a pipelined chunk's readers have taken a buffer);
/// set_flag: *flag = v after a system fence (a reader's acknowledgement).
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, cudaStream_t s);
void set_flag(unsigned* flag, unsigned v, cudaStream_t s);
/// A tensor-parallel group's all-reduce: host-mapped rows data[member][slot] (stride floats), flags[member][slot]
/// (32 apart).  x (n floats, n % 4 == 0: this member's partial) -> the sum of the nm members' partials in member order,
/// the same bits on every member; seq counts the group's all-reduces (the same on each member), slot = seq % slots.
struct ArGroup {
    float* data = nullptr;
    unsigned* flags = nullptr;
    size_t stride = 0;
    int nm = 1, slots = 4;
};
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, cudaStream_t s);

}  // namespace strata::ds4
