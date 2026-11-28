// src/ik2/ik2_draft.hpp - maya-ik2's drafter: Inkling 2's DSpark draft model (the official checkpoint's
// mtp.safetensors, BF16, or a GGUF of it: unsloth's dspark-Inkling-2-Mini-Q8_0.gguf) for speculative decoding.
//
//   * The target leaves 8 hidden states a position: its stream at the start of layers 5, 11, .., 41 and the final
//     normed hidden (taps).  For each committed position, each draft layer mixes them (softmax(fuse_weight[layer])),
//     RMS-norms (its dflare_norm), projects K / V (k_proj_ctx, v_proj_ctx; K's per-head norm and RoPE) into its own
//     context cache.  The last committed position's mix (fuse_weight[-1], the Markov head's dflare_norm) is f_rnn.
//   * A block of 8: the anchor (the last token the target chose, embedded and normed as the target embeds) then 7
//     mask_embed rows, through 5 dense layers (Qwen3-like: per-head q / k norm, NeoX RoPE theta 1e6, SiLU MLP), the
//     block's rows seeing each other and the context (layers 0, 1, 3, 4: the last 1024 positions); rows 1 .. 7 of
//     the final norm, through the target's output head, are the base logits.
//   * The Markov head drafts the 7 tokens one by one: a GRU-like cell over [state; markov_w1[prev]; hidden; f_rnn]
//     gives a latent whose bias on the logits is lm_head . (hidden_out . latent + b) - folded into a [V][256] matrix
//     once - and the token is the argmax of base + bias (the softcap keeps the order).
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ik2::draft {

struct Drafter;

/// The target's layers whose stream is a tap (at their start), the final normed hidden being tap 7.
const std::vector<int>& tap_layers();
constexpr int kTaps = 8, kBlock = 8;

/// Loads the drafter (path) onto the current device: mtp.safetensors (BF16), or a GGUF of it (llama.cpp's dflash
/// arch, e.g. unsloth's dspark-Inkling-2-Mini-Q8_0.gguf - its layers' matrices Q8_0, read on the tensor cores from
/// their blocks; its metadata checked against this drafter).  ctx: positions the context caches hold.  C: the
/// target's width (2304, 4608).  kv8: the layer without a window's context rows in FP8 (2 KiB a position instead of
/// FP16's 4 - the drafts only, the target checks every token).  Throws (std::runtime_error) on a file it does not
/// understand.
Drafter* load(const std::string& path, int C, int ctx, cublasHandle_t blas, cudaStream_t s, bool kv8 = false);
/// Frees everything load() and fold() took on the device (a drafter that does not fit after all).
void destroy(Drafter* d);
bool matrices_q8(const Drafter* d);
bool context_fp8(const Drafter* d);
/// The rows of the context cache of the layer without a window (IK2_DRAFT_CTX, default ctx).
int context_rows(int ctx);
/// The engine's plan for them where the whole context's do not fit (the last `rows` positions; 0: the context).
void plan_context_rows(int rows);
/// The Markov head's bias folded through the target's output head (FP16 [n_rows][C], on the device).
void fold(Drafter* d, const uint16_t* lm_f16, int n_rows);
/// ... through the head kept as planar Q8_0 (set_head_q8's form), read a chunk at a time: the same fold as through
/// its FP16 copy, without the copy.
void fold_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d, int n_rows);
/// The context caches for positions pos0 .. pos0 + m - 1 from tap rows r0 .. r0 + m - 1 (taps [8][ld][C] F32).
void write_context(Drafter* d, const float* taps, int ld, int r0, int m, int pos0);
/// f_rnn from tap row `row`.
void set_f_rnn(Drafter* d, const float* taps, int ld, int row);
/// Drafts 7 tokens (out, host) after the anchor at position P; anchor_h: its embedding, normed (device [C] F32).
/// lm_f16: the target's output head; n_vocab: the real rows (the argmax's range).
/// noise (a sampled request, or null): the drafts are then drawn - draft j the argmax of its softcapped logits over
/// the temperature plus k::gumbel_noise(salt, P + 1 + j, token), the noise the model's own token of that position
/// is drawn with - instead of the drafter's best tokens (a GGUF drafter's int8 fold; else its best tokens).
struct Noise {
    uint64_t salt = 0;
    float inv_temp = 0.0f, cap = 30.0f;
};
void propose(Drafter* d, int anchor, const float* anchor_h, int P, const uint16_t* lm_f16, int n_rows, int n_vocab,
             int* out, int proposals = kBlock - 1, const Noise* noise = nullptr);
/// The base logits through the target's head kept as planar Q8_0 (int8 [n_rows][C], FP16 scales [n_rows][C / 32]):
/// propose's lm_f16 is not read then (the FP16 copy is fold()'s only).
void set_head_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d);
/// propose's mean host time to queue its launches and its wait for them after, ms (a profile).
void propose_times(double* submit_ms, double* wait_ms);
/// VRAM it holds, bytes.
size_t bytes(const Drafter* d);
/// The context caches' rows of the positions before `pos` that their rings hold (each layer's last min(pos, cap)),
/// copied to the host (out, resized) / back from it - a session put aside and taken up again (--serve's session
/// cache).  Synchronous.
void save_context(const Drafter* d, int pos, std::vector<uint8_t>& out);
void load_context(Drafter* d, int pos, const std::vector<uint8_t>& in);
int mask_token_id(const Drafter* d);
void set_mask(Drafter* d, const float* mask_h);

}  // namespace ik2::draft
