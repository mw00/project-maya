// src/ik2/ik2_kernels.cuh - maya-ik2's kernels: the ops of Inkling 2 (thinkingmachines/Inkling-2-Small) that Maya's
// other engines do not have - the short convolutions, the attention residuals, the attention's clips, relative bias and
// sink, the softcapped SwiGLU with the experts' scales - and the decodes of the weights an `inkling2` GGUF holds
// (NVFP4, Q5_K, Q6_K, Q8_0).  The arithmetic follows ref/inkling2.py (transformers' modeling_inkling2.py), in F32; the
// experts' GEMMs of the prompt path take FP16 rows (an NVFP4 weight is exact in FP16).  docs/INKLING2.md has the model.
//
// Layouts: rows of tokens; a token's heads back to back ([T][H][D]); the K / V caches by position, [ctx][Hkv][D] F32.
// CUDA, and AMD (HIP, Linux) through include/strata/hip_compat.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ik2::k {

/// The current GPU's class for the kernels that differ by architecture: 70 (Volta: its own attention and NVFP4
/// expert kernels on mma.m8n8k4), 75 (Turing: Ampere's kernels with mma.sync m16n8k8 and no cp.async - not the
/// prompt's fused attention), 80 (Ampere and newer).  IK2_ARCH=70 / 75 takes a lower class on a newer GPU (tests).
int arch_class();

/// y = x * rsqrt(mean(x^2) + eps) * w per row of D (w null: no weight).  x and y may alias.
void rms(const float* x, const float* w, int rows, int D, float eps, float* y, cudaStream_t s);
/// rms of n <= 3 vectors [D] in one launch, each with its own weights: y[i] = rms(x[i]) * w[i] (y may be x).
struct RmsJobs {
    const float* x[3];
    const float* w[3];
    float* y[3];
};
void rms_multi(const RmsJobs& jobs, int n, int D, float eps, cudaStream_t s);
/// Fused RMSNorm + FP16 output (and optional F32 output y32 if non-null; y32 may alias x).
void rms_f16(const float* x, const float* w, int rows, int D, float eps, uint16_t* y16, float* y32, cudaStream_t s);

/// The scaled L2 clip, a group at a time: x [rows][G][D]; v = x * scale[g] (scale [G][D], null: 1);
/// y = v * min(1, th / (|v| + 1e-6)) * tau, tau = 1 + alpha ln max(1, (pos[row] + 1) / n_floor) with pos given (a full
/// layer's log scaling of q and r), else 1.  x and y may alias.
void l2clip(const float* x, const float* scale, int rows, int G, int D, float th, const int* pos, float alpha,
            float n_floor, float* y, cudaStream_t s);

/// The folded short conv (docs/INKLING2.md): y[t][c] = sum_j w[c][j] xp[t + j][c], xp = hist's 3 rows then x's T
/// (hist: the conv's last 3 inputs, oldest first; zeros at a sequence's start), taps = 4; with dyn [T][heads * 4]
/// (a full layer's k / v) + sum_lag dyn[t][h(c) * 4 + lag] xp[t + 3 - lag][c], h(c) = c / (C / heads).
/// Then hist = the last 3 rows of xp.  y must not alias x.
void conv4(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C, float* y,
           cudaStream_t s);
/// Residual-fused short conv: same as conv4, plus residual[t][c] added to y[t][c] when residual != nullptr.
void conv4_res(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C,
               const float* residual, float* y, cudaStream_t s);

/// The attention residual: the n states st[i] ([T][C] each, n <= kResMax; the block states then the stream) each
/// RMS-normed without a weight (eps), p = softmax_i(q . s_i / sqrt(C)) for each of the nq queries (q [nq][C]),
/// out [nq][T][C] = sum_i p_i s_i.
constexpr int kResMax = 12;
struct ResStates {
    const float* st[kResMax];
};
void attn_res(const ResStates& states, int n, const float* q, int nq, int T, int C, float eps, float* out,
              cudaStream_t s);
/// One token's attn_res for jobs x nq queries in one launch (a block a job): job j's queries are q + j nq C, its
/// output out + j nq C.
void attn_res_jobs(const ResStates& states, int n, const float* q, int nq, int jobs, int C, float eps, float* out,
                   cudaStream_t s);
struct RmsOuts {
    const float* w[3] = {};
    uint16_t* y16[3] = {};
    float* y32[3] = {};
};
/// Fused attn_res + per-query RMSNorm (res_eps for attn_res, rms_eps for output RMSNorm); writes FP16 outs.y16[qi]
/// and optional F32 outs.y32[qi] (outs.w[qi] may be null).  nq <= 3.
void attn_res_rms_f16(const ResStates& states, int n, const float* q, int nq, int T, int C, float res_eps,
                      float rms_eps, const RmsOuts& outs, cudaStream_t s);

/// One head's attention weights in place over lg [Tq][nk] = q . k (raw dots): query i at position qpos0 + i, key j at
/// kpos0 + j, dist = their difference;  logit = lg * qk_scale + (0 <= dist < extent ? r[i] . proj[dist] : 0), masked
/// unless 0 <= dist < window;  p = softmax over the keys and a sink logit sink[i] (whose value is zero: the weights sum
/// to less than 1).  r: this head's relative states, row i at r + i * r_ld (d_rel values); proj [extent][d_rel];
/// sink: row i at sink + i * sink_ld.
void attn_probs(float* lg, int Tq, int nk, int qpos0, int kpos0, float qk_scale, const float* r, int r_ld, int d_rel,
                const float* proj, int extent, int window, const float* sink, int sink_ld, cudaStream_t s);
/// Batched FP16-output variant across all H heads: lg [H][Tq][nk] F32 (read-only), r [Tq][H][d_rel], sink [Tq][H],
/// out16 [H][Tq][nk] FP16 in one launch.
/// attn_probs_f16 with the relative bias precomputed: bias [Tq * H][extent], row i * H + h, column the distance.
void attn_probs_bias_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0,
                         float qk_scale, const float* bias, int extent, int window, const float* sink, cudaStream_t s);
void attn_probs_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0, float qk_scale,
                    const float* r, int d_rel, const float* proj, int extent, int window, const float* sink,
                    cudaStream_t s);
/// Prefill attention fused (FP16 KV, D = 128): q16 [Tq] rows of q_ld (head h at h * D) at positions qpos0 ..;
/// kc / vc the cache [pos or pos % kv_cap][KV * D]; keys [max(0, qp - window + 1), qp] of each query; the relative
/// bias from the table bias [Tq * H][extent] (as attn_probs_bias_f16), sink [Tq][H] -> out [Tq] rows of o_ld (F32).
/// The keys split n_split ways (scratch: n_split * Tq * H * (D + 2) floats; fewer splits if it holds fewer).
void attn_flash_f16(const uint16_t* q16, int q_ld, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window,
                    const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_split,
                    cudaStream_t s);
/// attn_flash_f16 over an FP8 (E4M3) cache: its tiles turned into FP16 as they are loaded; qk_scale with K's scale
/// folded in, the output times v_inv_scale (as attn_decode_fp8).
void attn_flash_fp8(const uint16_t* q16, int q_ld, const uint8_t* kc, const uint8_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, float v_inv_scale, const float* bias, int extent,
                    int window, const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats,
                    int n_split, cudaStream_t s);
/// a split count for attn_flash_f16: Tq queries x H heads against up to nk keys on n_sm SMs
int attn_flash_splits(int Tq, int H, int nk, int n_sm);
/// Volta (sm_70): a prompt chunk's attention on a full layer (no window, no ring) on mma.m8n8k4 - q16 [Tq] rows of
/// q_ld at positions qpos0 .., each query's keys 0 .. its position; the bias in the kernel from proj [extent][d_rel]
/// and r [Tq] rows of r_ld (d_rel <= 16); sink [Tq][H] -> out [Tq] rows of o_ld (F32).  Its own splits (scratch as
/// attn_flash's).  False: not for these shapes (the caller takes its other path).
bool attn_prompt_v70_f16(const uint16_t* q16, int q_ld, const uint16_t* kc, const uint16_t* vc, int H, int Hkv, int Tq,
                         int qpos0, float qk_scale, const float* r, int r_ld, const float* proj, int d_rel, int extent,
                         const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_sm,
                         cudaStream_t s);
bool attn_prompt_v70_fp8(const uint16_t* q16, int q_ld, const uint8_t* kc, const uint8_t* vc, int H, int Hkv, int Tq,
                         int qpos0, float qk_scale, float v_inv_scale, const float* r, int r_ld, const float* proj,
                         int d_rel, int extent, const float* sink, float* out, int o_ld, float* scratch,
                         size_t scratch_floats, int n_sm, cudaStream_t s);
/// can this build run attn_flash_* on the current GPU (CUDA: always; AMD: the RDNA matrix-core kernel's archs)
bool attn_flash_available();

#ifdef STRATA_USE_HIP
/// AMD, RDNA 3 / 3.5 / 4 (ik2_rdna.inc): the prompt's routed experts of a layer in three launches from their blocks.
/// Rows: a token's pick each, grouped by expert in expert order; tiles: int4 {expert, first row, rows <= 32, 0} over
/// them; eptr [E]: an expert's blocks (gate, then up at off_up / down at off_down, as a cache slot holds them).
/// Can the current GPU take it for these expert types (all NVFP4 or all Q8_0) and widths (IK2_RDNA_MOE=0: no)?
bool experts_rdna_available(int ty_gate, int ty_up, int ty_down, int C, int F);
/// h16 [rows][F] = SwiGLU(x16 . gate^T * sg[row], x16 . up^T * su[row]) (softcaps gcap / ucap), x16 [rows][C]
void experts_rdna_gu(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_up,
                     const uint16_t* x16, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                     uint16_t* h16, cudaStream_t s);
/// y32 [rows][C] = h16 . down^T
void experts_rdna_down(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_down,
                       const uint16_t* h16, int C, int F, float* y32, cudaStream_t s);
/// out [T][C] = sum over token t's K picks (inv [T * K]: their rows) of wtr[row] y32[row], in row order
void moe_rows_combine(const float* y32, const int* inv, const float* wtr, int T, int K, int C, float* out,
                      cudaStream_t s);
/// AMD: y [T][ldy] = x [T][K] . W [N][K]^T, BF16 in, F32 sums, T <= 8 (the drafter's block): the weights read once,
/// a warp's rows sharing each x load.  False (nothing launched): not this shape (K % 8, alignment, T) - the BLAS then.
bool gemm_bf16_cols(const uint16_t* W, int N, int K, const uint16_t* x, int T, float* y, int ldy, cudaStream_t s);
#endif

/// o *= sigmoid(g), n values.
void gate_mul(float* o, const float* g, size_t n, cudaStream_t s);
/// o16[i] = f2h(o[i] * sigmoid(g[i])), and writes o[i] when write_f32 is true.
void gate_mul_f16(float* o, const float* g, size_t n, uint16_t* o16, bool write_f32, cudaStream_t s);

/// The router over logits [T][E]: sc = sigmoid; the K largest of sc + bias (ties: the lower expert) -> top [T][K] in
/// ascending expert order, w [T][K] = sc / (sum of the chosen + 1e-20) * scale.
void route(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
           cudaStream_t s);

/// FP16 rows gathered from F32 rows: y [n][C] = x[idx[i]][C].
void gather_f16(const float* x, const int* idx, int n, int C, uint16_t* y, cudaStream_t s);

/// The experts' activation over gate | up rows: gu [n][2 F] FP16 (gate first), row i's scales sg[i], su[i] (the
/// expert's NVFP4 scales; 1 without) -> h [n][F] FP16 = gcap tanh(g / gcap) sigmoid(g) ucap tanh(u / ucap),
/// g = gate * sg, u = up * su.
void swiglu_cap(const uint16_t* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                uint16_t* h, cudaStream_t s);

/// out [T][C] (+)= sum over the K slots i of wt[t * K + i] * y[t * K + i ... the row inv[t * K + i]] (y [rows][C] FP16;
/// wt already holds the route weight times the expert's down scale).  add: onto out, else out is overwritten.
void moe_combine(const uint16_t* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                 cudaStream_t s);

/// The same two with F32 inputs: gu [n][2 F] F32 (the experts' raw dot products can pass FP16's range: an NVFP4
/// expert's values are its true ones over its scale) -> h FP16; y [rows][C] F32.
void swiglu_cap_f32(const float* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                    uint16_t* h, cudaStream_t s);
void moe_combine_f32(const float* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                     cudaStream_t s);
/// One token's moe_combine_f32 and the conv4_res after it in one launch: out [C] = conv4(sum_k wt[k] y[inv[k]])
/// + res (hist [3][C], w [C][4]; res null: nothing added).
void moe_combine_conv(const float* y, const int* inv, const float* wt, int K, int C, float* hist, const float* w,
                      const float* res, float* out, cudaStream_t s);

/// Two GPUs (maya-ik2's split): GPU 0's table holds kHeldElsewhere for an expert GPU 1 runs, GPU 1's a null for one it
/// does not hold; the token path's expert kernels leave such a pick's row alone.  two_take_rows: the rows of GPU 0's
/// picks marked kHeldElsewhere (blob [S]) taken from GPU 1's rows (other [S][C], copied over) into y [S][C].
constexpr uintptr_t kHeldElsewhere = 1;
void two_take_rows(const uint8_t* const* blob, const float* other, int S, int C, float* y, cudaStream_t s);

/// y = a + b, n values (y may alias either).
void add(const float* a, const float* b, size_t n, float* y, cudaStream_t s);
/// FP16 copies of n values.
void to_f16(const float* x, size_t n, uint16_t* y, cudaStream_t s);

/// x = cap tanh(x / cap), n values (the head's logits).
void softcap(float* x, size_t n, float cap, cudaStream_t s);

/// The message gates of a looped model (Inkling-2-Mini: a core layer's inputs in the core's later passes), a row a
/// token: g = D [rms(x); rms(m)] (D [R][2 C]: x's columns first; the norms without a weight, eps),
/// y = (1 - tanh(F g)) x + tanh(G g) m (F, G by rank: [R][C]).  x: the input, m: the message from the pass before,
/// [T][C]; R <= 128; y may be x.  n <= 3 gates (a layer's q, k and v inputs) in two launches.
/// sc: scratch, [n][T][2 R + 2] floats.
struct LoopGate {
    const float *x, *m, *D, *F, *G;
    float* y;
};
void loop_gates(const LoopGate* gates, int n, int T, int C, int R, float eps, float* sc, cudaStream_t s);
void loop_gate(const float* x, const float* m, const float* D, const float* F, const float* G, int T, int C, int R,
               float eps, float* sc, float* y, cudaStream_t s);
/// Sampling's candidates found on the device: of x [n] (its values within [lo, hi]: softcapped logits) every value
/// of the highest bins that together hold K values at least (1024 bins over [lo, hi]) - K and a few more - gathered
/// in no order, so that the host sorts a thousand or two values and not the vocabulary.  out [2 + 2 cap] ints:
/// [0] their count (past cap: only cap of them are there), [1] the first bin taken, then idx [cap] and val [cap]
/// (F32).  scratch: cand_scratch(n) ints.
size_t cand_scratch(int n);
void cand_topk(const float* x, int n, float lo, float hi, int K, int cap, int* scratch, int* out, cudaStream_t s);
/// The same gate over a chunk of tokens with its products taken as matrix products by the caller (loop_gates'
/// kernels are a token's): loop_gate_mix gives g [T][R] = a1 / rms(x) + a2 / rms(m) of a1 = x D_x^T, a2 = m D_m^T
/// ([T][R]); loop_gate_blend y = (1 - tanh f) x + tanh(gn) m of f = g F, gn = g G ([T][C]; n = T C; y may be x).
void loop_gate_mix(const float* x, const float* m, const float* a1, const float* a2, int T, int C, int R, float eps,
                   float* g, cudaStream_t s);
void loop_gate_blend(const float* x, const float* m, const float* f, const float* gn, size_t n, float* y,
                     cudaStream_t s);

// ---- the token path: the experts straight from their blocks, the attention of one query

/// The routed experts of a few tokens as matrix-vector products over the blocks as the GGUF holds them (NVFP4, Q8_0,
/// Q5_K, Q6_K), one launch a phase.  Slot s (S of them): its expert's blob at blob[s] (gate | up | down: eb0, eb1 bytes
/// in), its token's row tok[s] of x.
///   gate_up: gu [S][2 F] = (gate, up) . x[tok[s]] (x [*][C] F32), raw sums (the scales applied by swiglu_cap_f32)
///   down:    y [S][C] = down . h[s] (h [S][F] FP16), raw sums
void experts_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                     const float* x, int C, int F, float* gu, cudaStream_t s);
/// One token's NVFP4 experts through the tensor cores (slot s: blob[s]; gate | up | down as above): the input in
/// FP16 (gate / up: x as two limbs, x16 [2][C] = hi, lo with x = hi + lo - to_f16_limbs, or route_token's x_limbs;
/// down: h [S][F]), a code times its scale exact in FP16, the sum FP32.  C and F multiples of 64.
void to_f16_limbs(const float* x, size_t n, uint16_t* y, cudaStream_t s);
bool experts_mma_fits(int ty_gate, int ty_up, int ty_down, int C, int F);
void experts_gate_up_swiglu_mma(const uint8_t* const* blob, int S, size_t eb0, const uint16_t* x16, int C, int F,
                                const float* sg, const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s);
void experts_down_mma(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                      cudaStream_t s);
void experts_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F,
                  float* y, cudaStream_t s);

/// y [N] = W [N][K] . x [K] for a dense matrix as the GGUF holds it (rows of `type` blocks: Q8_0, NVFP4, Q5_K, Q6_K),
/// a warp a row in a fixed order - one token's products, half the bytes of an FP16 copy for Q8_0.  K <= 8192.
void mv_blocks(int type, const uint8_t* W, int N, int K, const float* x, float* y, cudaStream_t s);

/// The same grouped: group g's expert at blob[g], picked by n[g] <= 8 tokens - x rows tok[g * 8 + j], output (and, for
/// down, input h) rows row[g * 8 + j] - its blocks read once for all of them.
constexpr int kGroupMax = 8;
void experts_gate_up_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int ty_gate,
                       int ty_up, size_t eb0, const float* x, int C, int F, float* gu, cudaStream_t s);
void experts_down_g(const uint8_t* const* blob, const int* n, const int* row, int G, int ty_down, size_t off_down,
                    const uint16_t* h, int C, int F, float* y, cudaStream_t s);
/// Fused gate+up+softcapped-SwiGLU straight to h [S][F] FP16 (or h[row[g*8+t]][F] in _g, where sg/su are also
/// indexed by row[g*8+t]).  sg, su may be null (scale 1).
void experts_gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                            const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                            uint16_t* h, cudaStream_t s);
/// experts_gate_up_swiglu with slot s's F outputs written at h + orow[s] * h_ld (orow null: s; h_ld 0: F) - a tensor-
/// parallel GPU's half of every row of a wider h
void experts_gate_up_swiglu_rows(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                 size_t eb0, const float* x, int C, int F, const float* sg, const float* su, float gcap,
                                 float ucap, uint16_t* h, const int* orow, int h_ld, cudaStream_t s);
void experts_gate_up_swiglu_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                              int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F, const float* sg,
                              const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s);

/// Fused attention-input prep: l2clip(q) in place, l2clip(k_in) -> FP16 kc[slot], rms(v_in) -> FP16 vc[slot],
/// l2clip(r) in place, where slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t).  pos may be null (sliding layers).
void prep_qkv_r_f16(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    cudaStream_t s);
/// attn_decode for nq queries at positions qpos0 .. qpos0 + nq - 1 (q [nq][H][128], r [nq][H][d_rel], sink [nq][H],
/// out [nq][H][128]), each over the keys k0 .. k0 + nk - 1 it may see.  scratch: nq x H x (nk + 128 x 130) floats.
void attn_multi(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int nq, int H,
                int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent, int window,
                float* scratch, float* out, cudaStream_t s);

/// One query's attention (a token being generated), every head in one launch: q [H][D] and r [H][d_rel] as the
/// clips left them, sink [H]; the keys at positions k0 .. k0 + nk - 1 of the caches kc / vc ([ctx][Hkv][D] F32), the
/// query at qpos; logits q . k * qk_scale + the relative bias (0 <= dist < extent), masked past the window, softmax
/// with the sink; out [H][D].  scratch: H x nk + H x 128 x 130 floats (nk >= 256: the keys in runs a block each,
/// combined after).  D = 128.
void attn_decode(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int H, int Hkv,
                 int D, int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                 float* scratch, float* out, cudaStream_t s);

/// FP16 K / V cache variants of attn_multi / attn_decode with optional ring-buffer wraparound (when kv_cap > 0, key
/// k0 + j sits at slot (k0 + j) % kv_cap; when kv_cap <= 0, linear slot k0 + j).  D = 128.
void attn_multi_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                    int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s);
void attn_decode_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale, const float* proj,
                     int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s);

/// FP8 (E4M3FN) K / V cache variants: scale K by k_scale and V by v_scale on write; fold 1/k_scale into qk_scale
/// and pass v_inv_scale = 1/v_scale on read.  slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t).  D = 128.
void prep_qkv_r_fp8(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    float k_scale, float v_scale, cudaStream_t s);
void attn_multi_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, cudaStream_t s);
void attn_decode_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                     float* out, cudaStream_t s);
/// Unwrap nk KV rows starting at k0 from FP8 ring buffer into contiguous FP16 [nk, KD] scaled by k_inv_scale / v_inv_scale.
void unpack_kv_fp8_to_f16(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int KD,
                          float k_inv_scale, float v_inv_scale, uint16_t* k_out16, uint16_t* v_out16,
                          cudaStream_t s);

// ---- the resident token path: every expert in VRAM (found on the device), the position read on the device, so a
// token needs nothing from the host between its embedding and its logits and its launches replay as one CUDA graph

/// y [N] = W [N][K] . x [K] for a Q8_0 matrix kept planar: q [N][K] its int8 values, d [N][K / 32] its blocks' FP16
/// scales (the GGUF's blocks taken apart, the values unchanged) - a warp reads 128 values an instruction.  K a
/// multiple of 128; q aligned to 4 bytes, x to 16.  decode_q8p_f16: the same matrix as FP16 (n values).
void mv_q8p(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, cudaStream_t s);
/// ... with a short conv (conv4's, one token) as the product's epilogue: the rows [r0, r0 + n) go through it - hist
/// [3][n], w [n][4], dyn [heads * 4] or null, res [n] added or null - to y [n] instead of to the product's y, and
/// hist takes the row.  A launch less a conv: every launch of a token costs 2 us and more, whatever it does.
struct MvConv {
    int r0 = 0, n = 0;
    float* hist = nullptr;
    const float* w = nullptr;
    const float* dyn = nullptr;
    int heads = 0;
    const float* res = nullptr;
    float* y = nullptr;
};
/// ... and with rows that take another input: the rows [r0, r0 + n) of a0 and of a1 are products with a0.x and
/// a1.x (a layer's fused q | k | v | r | gate matrix whose q, k and v each have an input of their own: one launch).
struct MvAlt {
    const float* x = nullptr;
    int r0 = 0, n = 0;
};
/// y [T][ldy] = W . x for a planar Q8_0 matrix (q8, d as mv_q8p's) and T <= a few tokens x [T][K] FP16, on the
/// tensor cores (a value times its block scale rounded to FP16, as the prompt path's decode_q8p_f16).
void q8_tile(const int8_t* q, const uint16_t* d, int N, int K, const uint16_t* x, int T, float* y, int ldy,
             cudaStream_t s);
void mv_q8p_conv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                 const MvConv& c1, cudaStream_t s, const MvAlt& a0 = MvAlt{}, const MvAlt& a1 = MvAlt{});
void decode_q8p_f16(const int8_t* q, const uint16_t* d, size_t n, uint16_t* out, cudaStream_t s);
/// y [N] = W [N][K] . x [K] for a small dense matrix kept in FP16 (the sinks, a full layer's dynamic taps, the
/// router): F32 sums, a warp a row.
void mv_f16(const uint16_t* W, int N, int K, const float* x, float* y, cudaStream_t s);
/// The routed slots' tables from the router's picks, all on the device.  top, w [S]: route()'s; tab [E]: where each
/// expert's blocks are (gate | up | down); sg_t, su_t, sd_t [E]: the experts' scales (null: 1).  Slot i (token
/// i / K): blob[i] = tab[top[i]], tok[i], inv[i] = i, sg[i], su[i], wt[i] = w[i] * down's scale - what
/// experts_gate_up_swiglu, experts_down and moe_combine_f32 take.
void route_gather(const int* top, const float* w, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                  const float* sd_t, int S, int K, const uint8_t** blob, int* tok, int* inv, float* sg, float* su,
                  float* wt, cudaStream_t s);
/// One token's route() and route_gather() in one launch (the same values to the bit): top, w [K] and the K slots'
/// tables.
void route_token(const float* logits, const float* bias, int E, int K, float scale, const uint8_t* const* tab,
                 const float* sg_t, const float* su_t, const float* sd_t, int* top, float* w, const uint8_t** blob,
                 int* tok, int* inv, float* sg, float* su, float* wt, cudaStream_t s, const float* x = nullptr,
                 int width = 0, uint16_t* x_limbs = nullptr);
/// prep_qkv_r_f16 / _fp8 of one token whose position is at pos (device): the cache slot from it, and with log_scale
/// (a full layer) q's and r's tau.  q16 (given): an FP16 copy of the clipped q, [H][D] (attn_decode_mma's).
/// sink_w (given): the token's sinks in the same launch, sink_y [H] = sink_w [H][sink_k] (FP16) . sink_x (mv_f16's).
void prep_qkv_r_f16_at(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       uint16_t* q16, cudaStream_t s, const uint16_t* sink_w = nullptr, const float* sink_x = nullptr,
                       int sink_k = 0, float* sink_y = nullptr);
void prep_qkv_r_fp8_at(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       float k_scale, float v_scale, uint16_t* q16, cudaStream_t s,
                       const uint16_t* sink_w = nullptr, const float* sink_x = nullptr, int sink_k = 0,
                       float* sink_y = nullptr);
/// attn_decode_f16 / _fp8 of the token at position *pos (device): the keys it may see follow from the position and
/// the window (1 << 30: every one), split P ways whatever their number (P fixed, so the launches do not change from
/// a token to the next).  scratch: H x P x 130 floats.
void attn_decode_f16_at(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, const float* proj,
                        int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s);
void attn_decode_fp8_at(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, float v_inv_scale,
                        const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                        cudaStream_t s);

/// One token's attention on the tensor cores - for long contexts, where the scalar attn_decode is bound by its
/// arithmetic; this one reads at the memory's speed, an FP8 cache too.  q16 [H][128]: the clipped q in FP16;
/// r [H][d_rel], sink [H] as attn_decode's.  The keys the token may see follow from its position (*pos on the device
/// if given, else qpos) and the window (1 << 30: every one); kv_cap > 0: the cache a ring of that many rows (>= 16).
/// A key's K and V rows are read once for the H / Hkv <= 8 heads that share them.  n_split: blocks a K / V head
/// (fixed, so the launches do not change with the position) - attn_decode_mma_blocks(Hkv) of them fill the device
/// once, a window of w keys needs no more than w / 64.  scratch: attn_decode_mma_scratch(H, n_split) floats.
/// gate [H][128] (or null): the output times its sigmoid (gate_mul's); out16 (or null): an FP16 copy of the output.
int attn_decode_mma_blocks(int Hkv);
size_t attn_decode_mma_scratch(int H, int n_split);
void attn_decode_mma_f16(const uint16_t* q16, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                         const float* gate, uint16_t* out16, cudaStream_t s, int blk0 = 0, int nblk = -1,
                         int phase = 0);
void attn_decode_mma_fp8(const uint16_t* q16, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                         float* out, const float* gate, uint16_t* out16, cudaStream_t s, int blk0 = 0, int nblk = -1,
                         int phase = 0);
/// (KV streaming) a part of the above: phase 1 the blocks blk0 .. blk0 + nblk - 1 of the n_split only (their
/// partials, the same as the whole launch's), phase 2 the combine only.  attn_decode_mma_keys: the keys [lo, hi) the
/// blocks b0 .. b1 - 1 read for a token at qpos (a layer without a window).
void attn_decode_mma_keys(int qpos, int n_split, int b0, int b1, int& lo, int& hi);

// ---- speculative decoding: the draft model's (Inkling 2's DSpark drafter, mtp.safetensors) and the rollback's

/// hist = the last 3 rows of [old (3 rows of C); x (T rows of C)] - a short conv's history after T of the inputs it
/// was given (the accepted part of a verify window).  hist may alias old.
void conv_hist_after(const float* old, const float* x, int T, int C, float* hist, cudaStream_t s);
/// BF16 copies of n values (nearest even).
void to_bf16(const float* x, size_t n, uint16_t* y, cudaStream_t s);
/// out [rows][C] = rms(sum_k alpha[k] x[k][row]) * w: x's taps n_tap apart (tap_stride floats), alpha (device) [n_tap].
void mix_rms(const float* x, int n_tap, size_t tap_stride, int rows, int C, const float* alpha, const float* w,
             float eps, float* out, cudaStream_t s);
/// NeoX rotary in place on x [rows][heads][128] at positions pos[row] (device), base theta (inverse frequencies in F32
/// as torch computes them).
void rope_neox(float* x, int rows, int heads, const int* pos, float theta, cudaStream_t s);
/// The drafter's attention: nq queries q [nq][H][128] over the context keys at positions k_lo .. k_hi - 1 of kc / vc
/// ([ctx][Hkv][128]) and the block's own nq keys kb / vb ([nq][Hkv][128], all visible to every query); softmax(q . k *
/// scale) . v -> out [nq][H][128].  scratch: H x nq x (k_hi - k_lo + nq) floats.
void draft_attn(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, const float* kb, const float* vb,
                int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s, int cap = 0);
/// ... over a context of FP16 rows.
void draft_attn_f16(const float* q, const uint16_t* kc, const uint16_t* vc, int k_lo, int k_hi, const float* kb,
                    const float* vb, int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s,
                    int cap);
/// out [T][F] BF16 = silu(gu[t][f]) * gu[t][F + f] (gu [T][2 F] F32: gate's then up's).
void silu_mul_bf16(const float* gu, int T, int F, uint16_t* out, cudaStream_t s);
/// The Markov head's input [state (R); emb[prev] (R, BF16 table); hidden (C); f (C)] as BF16 into out [2 R + 2 C].
void rnn_inputs(const float* state, const uint16_t* emb, const int* prev, const float* hidden, const float* f, int R,
                int C, uint16_t* out, cudaStream_t s);
/// The GRU-like step over proj [3 R] (+ bias [3 R]): gate = sigmoid, cand = tanh, out = tanh; state = gate state +
/// (1 - gate) cand (in place), latent = out.
void rnn_gates(const float* proj, const float* bias, int R, float* state, float* latent, cudaStream_t s);
/// *out = argmax over v < n of a[v] + b[v] + c[v] (b, c may be null; ties: the lower index), *out written on the device.
void argmax_sum(const float* a, const float* b, const float* c, int n, int* out, cudaStream_t s);
/// out[r] = argmax over c < n_cols of x[r * ld + c] for 0 <= r < n_rows (ties: the lower index), written on the device.
void argmax_rows(const float* x, int ld, int n_rows, int n_cols, int* out, cudaStream_t s);
/// *out = argmax over v < n of (base[v] + bias_f16[v] + sum_{r<R} W[r * ld + v] * x[r]), where W is [R][ld] FP16,
/// x is [R] F32, base (optional) is [n] F32, bias_f16 (optional) is [n] FP16.  Ties: the lower index.
void argmax_gemv_f16(const uint16_t* W, int ld, const float* x, const float* base, const uint16_t* bias_f16,
                     int R, int n, int* out, cudaStream_t s);

// ---- the weights' decodes (ggml's block types; values exactly ggml's dequantize_row_*, then rounded to the output)

/// ggml type ids of the blocks these decode.
constexpr int kQ8_0 = 8, kQ5_K = 13, kQ6_K = 14, kIQ3_XXS = 18, kIQ4_XS = 23, kNVFP4 = 40;
/// values a block / bytes a block (0: not one of the above).
int block_values(int type);
size_t block_bytes(int type);
/// n values (a multiple of the block's) of `type` blocks -> FP16 (nearest; NVFP4: exact) or F32.
void decode_f16(int type, const uint8_t* blocks, size_t n, uint16_t* out, cudaStream_t s);
void decode_f32(int type, const uint8_t* blocks, size_t n, float* out, cudaStream_t s);
/// ... on the host (the reference of the tests, and the embedding's rows).
void decode_host(int type, const uint8_t* blocks, size_t n, float* out);

// ---- the block path: a few tokens at once (2 .. 8, a speculative step's verify window) on the resident path, every
// position read from the device (pos [T]: the tokens' positions, consecutive) so that a step's launches replay

/// attn_decode_mma for the T tokens of a window (T * H / Hkv <= 24): q16 [T][H][D]; r: a token's [H][d_rel] r_ld
/// floats apart; sink [T][H]; a token's keys its own (causal, its window); gate rows gate_ld apart; out, out16
/// [T][H][D].  scratch: attn_block_scratch floats.
void attn_block_f16(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint16_t* kc,
                    const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int n_split, float qk_scale,
                    const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                    const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s);
void attn_block_fp8(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint8_t* kc,
                    const uint8_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int n_split, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s);
size_t attn_block_scratch(int H, int T, int n_split);
/// conv4_res for T tokens with row strides (x, res, y); with rec_old [3][C] and rec_x [T][C] given, the history
/// before and the inputs kept for rollback_block.
void conv_block(const float* x, int x_ld, float* hist, const float* w, const float* dyn, int d_ld, int heads, int T,
                int C, const float* res, int res_ld, float* y, int y_ld, float* rec_old, float* rec_x,
                cudaStream_t s);
/// A conv's record of a window (conv_block's rec_old, rec_x) and its history.
struct ConvRec {
    const float* old;
    const float* x;
    float* hist;
    int C;
};
/// Every recorded conv's history as after the first n of the window's tokens (recs: device [n_recs]).
void rollback_block(const ConvRec* recs, int n_recs, int n, cudaStream_t s);
/// y [T][N] (rows y_ld apart) = W [N][K] (FP16) . x [T][K] (rows x_ld apart).
void mv_f16_rows(const uint16_t* W, int N, int K, const float* x, int x_ld, int T, float* y, int y_ld,
                 cudaStream_t s);
/// x [T][C] as two FP16 limbs a token: out [T][2][C] (hi, lo).
void f16_limbs_rows(const float* x, int T, int C, uint16_t* out, cudaStream_t s);
/// route()'s picks of a window (S = T K <= 64 slots) grouped by expert on the device: blob [S], n [S] (0 past the
/// last group), tok / row [S][kGroupMax], and a slot's sg, su, wt (down's scale folded) and inv (the slot).
void group_route(const int* top, const float* w, int S, int K, const uint8_t* const* tab, const float* sg_t,
                 const float* su_t, const float* sd_t, const uint8_t** blob, int* n, int* tok, int* row, float* sg,
                 float* su, float* wt, int* inv, cudaStream_t s);
/// group_route's groups of NVFP4 experts on the tensor cores: gate | up (the input as f16_limbs_rows' limbs) to
/// h [slot][F] through the capped SwiGLU; down from h to y [slot][C].  G: the launch's groups (those past the last
/// return at once); T: the window's tokens (a group's members).
void experts_block_gu(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int T,
                      size_t off_up, int C, int F, const uint16_t* limbs, const float* sg, const float* su, float gcap,
                      float ucap, uint16_t* h, cudaStream_t s);
void experts_block_down(const uint8_t* const* blob, const int* n, const int* row, int G, size_t off_down, int C,
                        int F, const uint16_t* h, float* y, cudaStream_t s);
/// prep_qkv_r_*_at for the T tokens of a window: q and r rows q_ld and r_ld floats apart (q16 [T][H][D]), the
/// positions pos [T] on the device; sinks for each token (sink_x [T][sink_k], sink_y [T][H]).
void prep_qkv_r_fp8_block(float* q, int q_ld, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in,
                          uint8_t* vc, float* r, int r_ld, const float* r_clip, int T, int H, int KV, int D, int d_rel,
                          float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                          float n_floor, int kv_cap, float k_scale, float v_scale, uint16_t* q16, cudaStream_t s,
                          const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y);
void prep_qkv_r_f16_block(float* q, int q_ld, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                          uint16_t* vc, float* r, int r_ld, const float* r_clip, int T, int H, int KV, int D, int d_rel,
                          float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                          float n_floor, int kv_cap, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w,
                          const float* sink_x, int sink_k, float* sink_y);
/// route() for a window, a block a token (the same picks and weights to the bit).
void route_rows(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
                cudaStream_t s);
/// experts_block_gu / _down over Q8_0 experts (a block's sum scaled in FP32: the int8 values exact).
void experts_block_gu_q8(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int T,
                         size_t off_up, int C, int F, const uint16_t* limbs, const float* sg, const float* su,
                         float gcap, float ucap, uint16_t* h, cudaStream_t s);
void experts_block_down_q8(const uint8_t* const* blob, const int* n, const int* row, int G, size_t off_down, int C,
                           int F, const uint16_t* h, float* y, cudaStream_t s);
/// attn_res_jobs for T tokens: job j's queries q + j nq C, its output out + j nq T C ([nq][T][C]).
void attn_res_jobs_block(const ResStates& states, int n, const float* q, int nq, int jobs, int T, int C, float eps,
                         float* out, cudaStream_t s);

// ---- the exact window: the one-token path's arithmetic for each of a window's tokens (drafting on = off)
/// y [T][N] (rows y_ld apart) = W . x [T][K] (rows x_ld apart, F32) for a planar Q8_0 matrix: each token's row sums
/// mv_q8p's to the bit (T <= 8), the matrix read once.
void mv_q8p_cols(const int8_t* q, const uint16_t* d, int N, int K, const float* x, int x_ld, int T, float* y, int y_ld,
                 cudaStream_t s);
/// experts_gate_up_swiglu / experts_down for group_route's groups: each member's h / y the one-token kernels' to the
/// bit (x [T][C] F32; h [slot][F]; y [slot][C]), an expert's rows read once from memory for its members.
void experts_gate_up_swiglu_mem(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                                int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F, const float* sg,
                                const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s);
void experts_down_mem(const uint8_t* const* blob, const int* n, const int* row, int G, int ty_down, size_t off_down,
                      const uint16_t* h, int C, int F, float* y, cudaStream_t s);
/// experts_block_gu / _down give experts_gate_up_swiglu_mma / experts_down_mma's values to the bit (their warp
/// splits the same: IK2_EXPERT_MMA_SPLIT 8, _DOWN 4, the defaults).
bool experts_block_exact();

}  // namespace ik2::k
