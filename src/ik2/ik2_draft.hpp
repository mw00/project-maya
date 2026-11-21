// src/ik2/ik2_draft.hpp - maya-ik2's drafter: Inkling 2's DSpark draft model (the official checkpoint's
// mtp.safetensors, BF16) for speculative decoding, after SGLang's models/inkling2_mtp.py, dflash.py and dspark.py.
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

/// Loads mtp.safetensors (path) onto the current device; ctx: positions the context caches hold.  C: the target's
/// width (4608).  Throws (std::runtime_error) on a file it does not understand.
Drafter* load(const std::string& path, int C, int ctx, cublasHandle_t blas, cudaStream_t s);
/// The Markov head's bias folded through the target's output head (FP16 [n_rows][C], on the device).
void fold(Drafter* d, const uint16_t* lm_f16, int n_rows);
/// The context caches for positions pos0 .. pos0 + m - 1 from tap rows r0 .. r0 + m - 1 (taps [8][ld][C] F32).
void write_context(Drafter* d, const float* taps, int ld, int r0, int m, int pos0);
/// f_rnn from tap row `row`.
void set_f_rnn(Drafter* d, const float* taps, int ld, int row);
/// Drafts 7 tokens (out, host) after the anchor at position P; anchor_h: its embedding, normed (device [C] F32).
/// lm_f16: the target's output head; n_vocab: the real rows (the argmax's range).
void propose(Drafter* d, int anchor, const float* anchor_h, int P, const uint16_t* lm_f16, int n_rows, int n_vocab,
             int* out, int proposals = kBlock - 1);
/// The base logits through the target's head kept as planar Q8_0 (int8 [n_rows][C], FP16 scales [n_rows][C / 32]):
/// propose's lm_f16 is not read then (the FP16 copy is fold()'s only).
void set_head_q8(Drafter* d, const int8_t* q8, const uint16_t* q8d);
/// VRAM it holds, bytes.
size_t bytes(const Drafter* d);

}  // namespace ik2::draft
