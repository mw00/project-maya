// src/mm3/mm3_kernels.cuh - maya-mm3's own kernels around the EXL3 matrices (mm3_exl3.hpp): MiniMax-M3.1-Flash's
// norms, rotary, MiniMax Sparse Attention (the indexer's block selection and the attention over the kept blocks), the
// sigmoid router and swigluoai.  The arithmetic follows transformers' modeling_minimax_m3_vl.py, in F32; the KV cache
// and the indexer's keys are BF16 (vLLM's and transformers' model dtype).
//
// Layouts: rows of tokens; a token's heads back to back ([T][H][128]); the caches by position, [ctx][Hkv][128] for K
// and V and [ctx][128] for the index keys, of one layer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include <string>

namespace mm3::k {

/// y = x * rsqrt(mean(x^2) + eps) * w1 per row of D (w1 = 1 + the stored weight: Gemma's); y16 (if given) the same
/// rounded to FP16.  x and y may alias.
void rms(const float* x, const float* w1, int rows, int D, float eps, float* y, uint16_t* y16, cudaStream_t s);

/// The rotary's inverse frequencies theta^(-2i / n_rot), i < n_rot / 2 (host, F32 as transformers computes them).
void rope_inv_freq(float theta, int n_rot, float* inv /*[n_rot / 2]*/);

/// The K / V caches' row formats - a row is one position's one KV head, 128 values: its codes, then (another array)
/// its scales.
///   kKvBf16  256 bytes of BF16 values (the model's dtype: exact)
///   kKvI8    128 int8 round(v / s), an F32 scale s = max |v| / 127
///   kKvFp8   128 E4M3 codes, a UE8M0 power-of-two scale a group of 32 (the scale byte: its F32 exponent)
///   kKvFp4   64 bytes of E2M1 codes (value 2i in the low nibble), an E4M3 scale a group of 16 (NVFP4's)
constexpr int kKvBf16 = 0, kKvI8 = 1, kKvFp8 = 2, kKvFp4 = 3;
__host__ __device__ constexpr size_t kv_code_bytes(int f) { return f == kKvBf16 ? 256 : f == kKvFp4 ? 64 : 128; }
__host__ __device__ constexpr size_t kv_scale_bytes(int f) { return f == kKvBf16 ? 0 : f == kKvFp4 ? 8 : 4; }
/// A row's bytes in all (codes and scales)
__host__ __device__ constexpr size_t kv_row_bytes(int f) { return kv_code_bytes(f) + kv_scale_bytes(f); }
/// A layer's quantized K / V caches (--kv-cache): K's rows in format kf, V's in vf - for a format other than BF16 its
/// codes [rows][Hkv][kv_code_bytes] and scales [rows][Hkv][kv_scale_bytes] here (the BF16 caches kc / vc unused for
/// it); rot (bit 0 K, bit 1 V): the rows rotated by the 128-point Hadamard transform before they are rounded (outliers
/// spread over the row - the queries are rotated alike and the attention's output back, so q . k and the values'
/// sum keep their meaning).  All 0: the BF16 caches.
struct KvQ {
    int kf = kKvBf16, vf = kKvBf16;
    int rot = 0;
    uint8_t *kc = nullptr, *vc = nullptr;
    uint8_t *ks = nullptr, *vs = nullptr;
};

/// The bytes of a layer's K and V caches of `rows` rows (positions x KV heads) in formats kf / vf.
inline size_t kv_bytes(int kf, int vf, size_t rows) { return rows * (kv_row_bytes(kf) + kv_row_bytes(vf)); }
/// One block of kv_bytes(kf, vf, rows) carved into the caches: K's codes and scales (a BF16 K: kc16), then V's.
inline void kv_carve(uint8_t* b, size_t rows, int kf, int vf, int rot, uint16_t*& kc16, uint16_t*& vc16, KvQ& q) {
    q.kf = kf;
    q.vf = vf;
    q.rot = rot;
    const int f[2] = {kf, vf};
    for (int r = 0; r < 2; ++r) {
        if (f[r] == kKvBf16) {
            (r == 0 ? kc16 : vc16) = (uint16_t*) b;
        } else {
            (r == 0 ? q.kc : q.vc) = b;
            (r == 0 ? q.ks : q.vs) = b + rows * kv_code_bytes(f[r]);
        }
        b += rows * kv_row_bytes(f[r]);
    }
}
/// A format's name (bf16, int8, fp8, fp4) and back (-1: none of them).
inline const char* kv_name(int f) { return f == kKvI8 ? "int8" : f == kKvFp8 ? "fp8" : f == kKvFp4 ? "fp4" : "bf16"; }
inline int kv_parse(const char* s) {
    const std::string v = s;
    return v == "bf16" || v == "16" ? kKvBf16 : v == "int8" || v == "i8" ? kKvI8 : v == "fp8" || v == "8" ? kKvFp8
           : v == "fp4" || v == "4" ? kKvFp4 : -1;
}

/// KV streaming (--kv-streaming): the VRAM K / V caches a ring of `ring` rows - position p at row p % ring, tag[row] the
/// position the row holds (-1: none) - and every position in host memory as well (pinned and mapped: rows [max_ctx]
/// [Hkv] in the VRAM caches' formats, kc / vc BF16 or q's codes and scales).  Attention reads a key from VRAM when the
/// ring holds it, else from the host over PCIe.  ring 0: off (the VRAM caches a row a position).
struct KvHost {
    int ring = 0;
    int* tag = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    KvQ q;
};

/// q [T][Hq][hd] in place: the per-head RMSNorm (one [hd] weight w1 = 1 + w for every head) then NeoX RoPE on the
/// first n_rot dims at pos[t]; k [T][Hkv][hd] the same and v as it is, both written to the caches at row pos[t] as
/// BF16.  inv [n_rot / 2] on the device.
void qk_norm_rope(float* q, const float* k, const float* v, const float* qn1, const float* kn1, const int* pos, int T,
                  int Hq, int Hkv, int hd, int n_rot, const float* inv, float eps, uint16_t* kc, uint16_t* vc,
                  cudaStream_t s, int ring = 0, const KvQ& kq = KvQ{}, const KvHost& hs = KvHost{});
/// (ring > 0: the caches are rings of that many rows, position p at row p % ring - the DSpark draft's window.  Hq 0:
///  K and V alone)

/// The indexer's queries iq [T][Hi][D] in place (norm, RoPE, rounded through BF16 as the model dtype keeps them) and
/// its key ik [T][D] (norm, RoPE) into the index cache ic at row pos[t] (BF16).
void index_norm_rope(float* iq, const float* ik, const float* qn1, const float* kn1, const int* pos, int T, int Hi,
                     int D, int n_rot, const float* inv, float eps, uint16_t* ic, cudaStream_t s);

/// bs [Hi][T][nblk]: for query t (at pos[t]) and index head h, block b's score = max over the keys j of block b with
/// j <= pos[t] of iq[t][h] . ic[j] (F32); -inf for a block past the query.  D = 128.
void block_scores(const float* iq, const uint16_t* ic, const int* pos, int T, int Hi, int nblk, int blk, float* bs,
                  cudaStream_t s);

/// kept [Hi][T][topk]: the query's own block (pos / blk) and the topk - 1 best of the blocks before it (ties: the lower
/// block), -1 past the valid ones - MiniMaxM3VLIndexer's selection with sparse_local_block 1, as a set.
void select_blocks(const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept,
                   cudaStream_t s);

/// out [T][Hq][hd] (F32): each query head over its KV group's kept blocks (kept [Hkv][T][topk], -1 unused), the keys
/// j <= pos[t] of them, softmax(q . k * scale) . v - exact over the cached rows (online softmax in F32).  hd = 128,
/// Hq / Hkv = 16.  kq: the caches' formats (KvQ).  splits > 1 (with part: T x Hkv x splits x 16 x 130 floats): each
/// query's kept blocks in that many runs on blocks of their own (a decode window's few queries), combined in run order -
/// a query's sums depend on splits alone, not on T.
void sparse_attn(const float* q, const uint16_t* kc, const uint16_t* vc, const int* kept, const int* pos, int T,
                 int Hq, int Hkv, int topk, int blk, float scale, float* out, cudaStream_t s, const KvQ& kq = KvQ{},
                 const KvHost& hs = KvHost{}, int splits = 1, float* part = nullptr);
/// part's floats for sparse_attn's splits over T queries and Hkv groups.
inline size_t attn_part_floats(int T, int Hkv, int splits) { return (size_t) T * Hkv * splits * 16 * 130; }

/// The router over logits [T][E] (F32): s = sigmoid, chosen by s + bias (the K largest, ties to the lower expert),
/// weights s / sum over the chosen -> top [T][K], w [T][K].
void route(const float* logits, const float* bias, int T, int E, int K, int* top, float* w, cudaStream_t s);

/// The routed experts' plan on the device, for a layer whose experts are all in VRAM (no round trip through the host):
/// token t's pick i < K -> plan_ptr[t * 8 + i] = tab[top[t * K + i]] (tab: expert -> its blob's device address) and
/// plan_w[t * 8 + i] = w[t * K + i] * scale; picks i >= K: 0 and 0.  counts (given): counts[expert] += 1 a pick (the
/// expert profile's routings).  grp_ptr (given, T <= 16): the entries grouped by expert pointer in order of first
/// appearance, each group's entries e = t * 8 + i in entry order (exl3::WinArgs' grp_ptr / grp_start / grp_ent /
/// n_groups) - the host's plan and grouping value for value.  false: T * 8 > 128 or K outside 1..8 (nothing launched).
bool plan_dev(const int* top, const float* w, int T, int K, const unsigned long long* tab, float scale,
              unsigned long long* plan_ptr, float* plan_w, unsigned* counts, unsigned long long* grp_ptr, int* grp_start,
              int* grp_ent, int* n_groups, cudaStream_t s);
/// Windows + AMD submit queued GPU work only when the host waits: the stream's queued work started now, without a wait
/// (cudaStreamQuery).  Elsewhere nothing.  MAYA_MM3_WIN_SUBMIT=0: off.
void submit(cudaStream_t s);

/// h = swigluoai(g, u) = min(g, L) * sigmoid(alpha min(g, L)) * (clamp(u, -L, L) + 1), elementwise over n values; F32
/// into hf and / or FP16 into h16 (either may be null).
void swigluoai(const float* g, const float* u, size_t n, float alpha, float limit, float* hf, uint16_t* h16,
               cudaStream_t s);
/// The same over gate / up halves of rows: gu [rows][2 ff] (gate first) -> h16 [rows][ff].
void swigluoai_pairs(const float* gu, int rows, int ff, float alpha, float limit, uint16_t* h16, cudaStream_t s);

/// FP16 copies of rows (ld floats a row in, ldo halves out).
void to_f16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s);

/// out[t] = res[t] + scale * sum over slots i of w[t][i] * y[inv[t * K + i]] + (sh ? sh[t] : 0), rows of D (res null: 0).
void moe_combine(const float* res, const float* y, const int* inv, const float* w, const float* sh, int T, int K,
                 int D, float scale, float* out, cudaStream_t s);

/// p[i] = base + i, i < n (a prompt chunk's positions, without a host buffer).
void iota(int* p, int n, int base, cudaStream_t s);
/// *p = v (on the stream: no copy).
void set_int(int* p, int v, cudaStream_t s);
/// The DSpark draft's block: h row 0 = the anchor's BF16 row (anchor: device-readable, e.g. mapped), rows 1 .. B - 1 =
/// mask (F32 [D], device); pos[t] = P + t, lo[t] = max(0, P + t - window + 1).
void draft_block(const uint16_t* anchor, const float* mask, int D, int B, int P, int window, float* h, int* pos, int* lo,
                 cudaStream_t s);
/// x += a (n floats).
void add(float* x, const float* a, size_t n, cudaStream_t s);
/// (the tensor-parallel path) one thread spins until *flag (mapped host memory, a counter) reaches v, wrap-safe - for at
/// most 120 s (then it goes on and wait_timeouts() counts it: a host that never answers leaves no GPU spinning).
void wait_flag(const unsigned* flag, unsigned v, cudaStream_t s);
/// The current device's waits on the host that gave up (wait_flag), since the process started.
unsigned wait_timeouts();
/// (the tensor-parallel path) n floats (a multiple of 4) from src (mapped host memory) into dst (device) by a kernel - no
/// copy engine: a copy waiting behind a wait_flag on one stream holds up the device's host -> device copies of the others
void pull(float* dst, const float* src, size_t n, cudaStream_t s);
/// (the tensor-parallel path) wait_flag and pull in one kernel: once *flag reaches v, n floats (a multiple of 4) from
/// src (mapped host memory) into dst - no copy engine, and no launch between the wait and the read.
void wait_pull(const unsigned* flag, unsigned v, float* dst, const float* src, size_t n, cudaStream_t s);
/// (the tensor-parallel path) n floats (a multiple of 4) from src (device) to dst (mapped host memory), then *flag = v
/// once every block's writes are the system's: each block fences, the last to count (counter: a device word, 0
/// between calls - it is left 0) raises the flag.
void publish(const float* src, float* dst, size_t n, unsigned* counter, unsigned* flag, unsigned v, cudaStream_t s);
/// Y [T][N] = X [T][K] . W [N][K]^T in F32, one warp an output in a fixed order: a token's row is the same bits
/// whatever T (the token path's and a window's router and indexer agree).
void rows_dot(const float* W, int N, int K, const float* X, int T, float* Y, cudaStream_t s);
/// ... W as ggml's Q8_0 rows (34-byte blocks: an F16 scale, 32 int8; K % 32 == 0) - each value scale * q in F32, the
/// value an F32 upload of the matrix holds, so rows_dot's bits - for two matrices on the same X in one launch (Wb may be
/// null with Nb 0): Ya [T][Na], Yb [T][Nb].
void rows_dot_q8(const uint8_t* Wa, int Na, const uint8_t* Wb, int Nb, int K, const float* X, int T, float* Ya,
                 float* Yb, cudaStream_t s);
/// n values of Q8_0 blocks -> F32 (scale * q: an F32 upload's values), e.g. for a prompt's F32 GEMM.
void q8_0_f32(const uint8_t* W, size_t n, float* out, cudaStream_t s);

// ---- the DSpark draft's

/// out [N][K] BF16 = an MXFP8 matrix: w (E4M3 bytes [N][K]) times 2^(scale - 127), scale (E8M0 bytes [N][K / 32]).
void mxfp8_to_bf16(const uint8_t* w, const uint8_t* scale, int N, int K, uint16_t* out, cudaStream_t s);
/// y [rows][ldy] = beta y + x [rows][ldx] . W^T for an MXFP8 W [N][K] (E4M3, a UE8M0 scale every 32 inputs) read as
/// it is stored - each weight exactly mxfp8_to_bf16's value, the sums in F32 in a fixed order.  rows <= 16, N a multiple
/// of 32, K of 512.  part (rows x N x 16 floats, or null): a split of K for an N of few blocks, its partials added in
/// order.  (The DSpark draft's few rows: half the bytes of its BF16 weights.)  More than 16 rows: in runs of 16.
void mx_rows(const uint8_t* w, const uint8_t* scale, int N, int K, const float* x, int rows, int ldx, float* y, int ldy,
             float beta, float* part, cudaStream_t s);
/// BF16 copies of rows (ld floats a row in, ldo out).
void to_bf16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s);
/// n BF16 values as FP16 (nearest) into y, on s (waited for); false: one past FP16's range (y then not to be used).
bool bf16_to_f16(const uint16_t* x, size_t n, uint16_t* y, cudaStream_t s);
/// out [T][Hq][hd]: each query head over its KV group's keys at the positions lo[t] .. hi (rings of `ring` rows,
/// position p at row p % ring), not causal - softmax(q . k * scale) . v in F32.  hd = 128, Hq / Hkv = 16.  part (given,
/// attn_part_floats(T, Hkv, splits) floats): the keys in `splits` runs of blocks of their own (at most 32), combined in
/// run order - the GPU busy instead of T x Hkv blocks walking the whole window.
void window_attn(const float* q, const uint16_t* kc, const uint16_t* vc, int ring, const int* lo, int hi, int T,
                 int Hq, int Hkv, float scale, float* out, cudaStream_t s, float* part = nullptr, int splits = 1);
/// Candidates instead of a row of logits (the sampler's: main.cpp's dist() over them is its own over the row): the kc
/// largest and every one equal to the kc-th - n of them (-1: more than kCandCap), by logit desc then id asc, their
/// exact logits - and the logits of n_extra (<= kCandExtra) ids the caller names (the penalised ones).
constexpr int kCandCap = 512, kCandExtra = 128;
struct CandRow {
    int n;
    int id[kCandCap];
    float v[kCandCap];
    float ex[kCandExtra];
};
/// CandRow out[r] (mapped host memory) for each of rows rows of lg [rows][V] (device); extra: ids (device-readable).
void topk_rows(const float* lg, int rows, int V, int kc, const int* extra, int n_extra, CandRow* out, cudaStream_t s);
/// A sampled draft step's distribution: q[i] of id[i], n of them (the draft's kq largest biased logits, by dist()'s
/// steps), and the token drawn from it.
constexpr int kQMax = 64;
struct QRow {
    int n;
    int id[kQMax];
    float q[kQMax];
};
struct SampleArgs {
    float temperature = 1.0f, top_p = 1.0f, min_p = 0.0f, u = 0.0f;   // u: the draw, uniform in [0, 1)
    int top_k = kQMax;
};

/// A greedy Markov step whole (markov_pick's argmax and confidence, w2 [V][R] . w1[*prev] the bias - R 256): the bias
/// rows over the GPU (8 lanes a row), each block's argmax, then one block for theirs and the confidence -> *prev, *out.
/// part: markov_part_floats() floats of scratch; bias_out (if given) gets the bias.  false: R not 256 (nothing run).
bool markov_greedy(const float* logits, const uint16_t* w2, const uint16_t* w1, int V, int R, int* prev, int* out,
                   const float* w, const float* b, const float* hidden, int D, float* conf, float* part, float* bias_out,
                   cudaStream_t s);
size_t markov_part_floats();
/// One Markov step: emb (BF16 [R], for the GEMV) and embf (F32 [R]) = table row *prev.
void gather_row(const uint16_t* table, int R, const int* prev, uint16_t* emb, float* embf, cudaStream_t s);
/// ... then: the next token = argmax(logits + bias) (ties: the lower id) -> *prev and *out; biased (if given) = logits +
/// bias; conf (if given) = sigmoid(w . [hidden ; embf] + b) with w [D + R] F32, b F32.
void markov_pick(const float* logits, const float* bias, int V, int* prev, int* out, float* biased, const float* w,
                 const float* b, const float* hidden, int D, const float* embf, int R, float* conf, cudaStream_t s);
/// ... or sampled: the token drawn from the biased logits' distribution as sa says it (QRow *qo, mapped host memory,
/// gets that distribution) -> *prev and *out, the confidence as markov_pick's.
void markov_sample(const float* logits, const float* bias, int V, int* prev, int* out, const float* w, const float* b,
                   const float* hidden, int D, const float* embf, int R, float* conf, const SampleArgs& sa, QRow* qo,
                   cudaStream_t s);

}  // namespace mm3::k
