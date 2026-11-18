"""ref/kimi_k31.py - the Kimi K3.1 (kimi-k31) reference forward pass, one op at a time.

Phase 1 of a Kimi K3.1 port: the specification every engine kernel is parity-tested against, and
the model an independent implementation built from Moonshot's own modeling code is gated against.
K3.1 is Kimi K3 (Moonshot's `modeling_kimi_linear.py`, HF `moonshotai/Kimi-K3` @ f831ab6) with a
per-token "lightning" indexer (Gated DSA) on every MLA layer; Moonshot ships no K3.1 modeling code,
so the indexer follows the two public serving implementations (SGLang `kimi-k3.1-support`
@ accf045c, vLLM PR #65685 @ 9d6ac73f), which agree on every semantic point.

It reads a `kimi-k31` GGUF: tensor names and `kimi-k31.*` keys are what llama.cpp's kimi-k3
converter writes, plus five indexer tensors per MLA layer (`blk.N.indexer.{attn_q_b,attn_k,k_norm,
proj}`) and the keys `attention.indexer.{head_count,key_length,top_k}` and
`attention.layer_norm_epsilon`. `tools/kimi_k31_synth_gguf.py` writes a tiny one. Every key has the
released K3.1 value as its default, so a synthetic GGUF only has to write what it changes.

Conventions are ref/glm.py's (ggml's), so the two references and llama.cpp compare op by op:
  * activations are (features, tokens), features varying fastest like a ggml tensor's ne[0];
  * weights from ref/load.py's `tensor()` are (out, in), so `y = W @ x` IS `ggml_mul_mat`;
  * a (dim*n_head, T) projection splits into heads in ggml's reshape order (element (e, h) at
    row e + dim*h), which is also the checkpoint's `view(T, n_head, dim)` order;
  * float32 at every seam, float64 accumulation inside norms, softmaxes, dot products and the
    KDA recurrence (whose state is kept in float64); dtype="float64" carries float64 everywhere
    instead (on the tiny fixture prefill and decode then agree to ~1e-13, in float32 to ~1e-5
    relative: matrix-vector vs matrix-matrix rounding);
  * every intermediate is recorded by `cb()`, under llama.cpp's cb() name where the kimi-k3 /
    kimi-k31 graph has one, otherwise under a name in the same style (list below).

The model, in order (0-indexed layers; config layer lists are 1-indexed):
  * Block Attention Residuals replace the residual stream entirely. Two things cross layers: the
    stack of completed blocks [b_0 = embeddings, B_1, ...] and the current block's partial sum P.
    Before attention (skipped at layer 0) and before the FFN, the layer input is a softmax mix of
    [blocks..., P]: s_i = sum_d w_d v_id / sqrt(mean_d v_i^2 + eps), w = res_norm * res_proj (fused
    in the GGUF as *_res_score), and the mix sums the RAW candidates. On layers il % block == 0
    the incoming P is pushed as a block and P restarts from that layer's attention output. One
    last mix (output_res_score) feeds the final norm. Nothing here crosses tokens.
  * KDA (kimi delta attention) layers: q/k/v projections, depthwise causal short conv (4 taps,
    the last tap on the current token) + SiLU, q/k l2-normalized per head, q scaled by dk^-0.5,
    decay gate g = lb * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias)) per KEY channel
    (lb = gate_lower_bound = -5; NOT the softplus form; the GGUF's ssm_a holds -exp(A_log)
    sliced to the head count), beta = sigmoid(b_proj(x)) per head, then per head with state S
    (dk x dv, zero at sequence start):
        S <- diag(exp(g_t)) S;   S <- S + k_t (beta_t (v_t - S^T k_t))^T;   o_t = S^T q_t,
    then o = RMSNorm(o; o_norm, per head) * sigmoid(g_proj(x)) and o_proj.
  * MLA layers, NoPE: q = q_b(RMSNorm(q_a(x))) as 96 heads x (128 nope + 64 "rope"), the
    576-wide kv_a row = [512 latent | 64 key part shared by all heads]; the latent is
    RMS-normed, the 64-wide part is used raw and NEVER rotated (there is no RoPE anywhere in
    K3.1). Attention is computed absorbed, as engines do: score_h(t, s) = (k_b_h^T q_nope_h) .
    c_s + q_pe_h . k_pe_s, scale (nope+rope)^-0.5, V = v_b_h c (exactly the materialized
    kv_b form; tools/test_kimi_k31_ref.py checks it). Output = kqv * sigmoid(attn_gate(x)),
    then o_proj.
  * The lightning indexer on each MLA layer (every DSA layer runs its own; index_topk_freq 1,
    index_skip_topk_offset 1 are no-ops; no RoPE, no Hadamard, no always-kept tokens):
        iq = indexer.attn_q_b(qr)  (qr = the normed q latent)  -> (head_dim, n_head_idx)
        ik = LayerNorm(indexer.attn_k(x); k_norm weight + bias)
        iw = indexer.proj(x) * (n_head_idx * head_dim)^-0.5
        score[t, s] = sum_h iw[t, h] * ReLU(iq[t, h] . ik[s])
    Query t sees keys 0..t (its own included); future keys are -inf, NOT 0 (scores can be
    negative, so a 0-filled row picks future keys). The top_k highest visible keys are
    selected; MLA then attends to exactly those (a -inf mask on the rest). With <= top_k
    visible keys this is dense causal MLA.
  * Layer 0 (leading_dense_block_count) is a dense SiTU-GLU MLP; the rest are latent MoE:
    router logits on the full-width x, probs = sigmoid, top-k (16 released; > 8) by probs +
    e_score_correction_bias (noaux_tc: the bias picks, it never weights), weights = the
    unbiased probs renormalized to sum 1 (+1e-20) times routed_scaling_factor; experts run in
    the latent space z = routed_down(x) (3584 released); the weighted sum is RMS-normed
    (routed_norm, AFTER the sum) and up-projected; shared experts (SiTU-GLU, width
    n_ff_exp * n_shared) read the full-width x.
  * SiTU-GLU: situ(g, u) = beta tanh(g/beta) sigmoid(g) * lbeta tanh(u/lbeta) (4.0 / 25.0).
  * Final RMSNorm, untied lm_head, no softcap.

Epsilons (the `eps` argument; `eps_overrides` changes any single one):
    name          used by                                      "moonshot" (default)  "llama.cpp"
    rms           attn/ffn norms, attn-res scoring, KDA o_norm, routed_norm, output norm
                                                               GGUF rms eps (1e-5)   same
    mla_latent    q_a_layernorm, kv_a_layernorm                1e-6 (KimiRMSNorm     rms (1e-5)
                                                               class default)
    kda_l2        q/k l2 norm x / sqrt(sum x^2 + eps)          1e-6 (fla kernels)    rms (1e-5)
    indexer_ln    indexer k LayerNorm                          GGUF layer_norm_epsilon (1e-5) both;
                                                               engine consensus, not Moonshot code
    router_renorm MoE weight renorm w / (sum + eps)            1e-20 (Moonshot)      same
  "moonshot" follows Moonshot's code and the fla kernels it calls; "llama.cpp" follows the
  llama.cpp kimi-k3 graph, which uses the GGUF rms eps for both. vLLM and SGLang also use 1e-5
  for q_a/kv_a. The difference is small: a relative change of about (1e-5 - 1e-6) / (2 * mean
  square) on each normed vector.

Teacher forcing and state. `forward(tokens)` runs one teacher-forced pass over a fresh
sequence. `forward(tokens, state)` continues a sequence: the state carries, per KDA layer, the
last d_conv-1 projection inputs of each short conv and the float64 recurrent state, and per MLA
layer the normed latents, the raw 64-wide key parts and the indexer keys of every earlier
position. Prefill, chunked prefill and token-by-token decode therefore run the same ops;
tools/test_kimi_k31_ref.py checks they agree.

Seams (`self.dumps`, reset by every forward; il = layer, -1 = none; T = the tokens of this call,
n_kv = positions so far including them):
    inp_embd (n_embd, T)
    res_probs_attn-il (n_cand, T), res_mix_attn-il (n_embd, T)      not on layer 0
    attn_norm-il
    KDA: kda_q_conv, kda_k_conv, kda_v_conv (d_inner, T); kda_g1 (dk, n_head, T);
         kda_beta (n_head, T); kda_scan_out, kda_g2, kda_normed (dk, n_head, T); kda_out (n_embd, T)
    MLA: qr (q_lora, T); kv_cmpr (kv_lora, T); k_pe (rope, T); q_absorbed (kv_lora, n_head, T);
         indexer_q (idx_dim, idx_heads, T); indexer_k (idx_dim, T);
         indexer_weights (idx_heads, T) UNSCALED (llama.cpp's cb point; the scale is applied after);
         indexer_score (n_kv, T) with -inf on future keys; top_k (min(top_k, n_kv), T) int64:
         absolute key positions in descending score order (ties to the lower position), -1 pads
         a query with fewer visible keys - compare as SETS; kq_mask_top_k (n_kv, T) 0 / -inf;
         kqv_out (v_head*n_head, T); mla_gated; mla_out (n_embd, T)
    prefix_sum_attn-il, res_probs_ffn-il, res_mix_ffn-il, ffn_norm-il
    dense: ffn_out-il
    MoE: ffn_moe_logits, ffn_moe_probs, ffn_moe_probs_biased (n_expert, T); ffn_moe_topk (k, T)
         int64 in descending biased-score order; ffn_moe_weights (raw selected probs),
         ffn_moe_weights_norm, ffn_moe_weights_scaled (k, T); ffn_routed_in (latent, T);
         ffn_moe_out (latent, T) = the weighted expert sum, before routed_norm (llama.cpp's cb
         point); ffn_routed_normed (latent, T); ffn_routed_out, ffn_shexp, ffn_out (n_embd, T)
    l_out-il (the partial sum P after the layer)
    res_blocks-(N-1) (n_embd, n_blocks, T): the completed block stack, on a truncated run only
         (layers < the file's, or no head); not a llama.cpp cb() name
    res_probs, res_mix, result_norm (n_embd, T), result_output (n_vocab, T)
  llama.cpp's graph names both per-layer mixes "res_probs"; here they carry an _attn / _ffn
  suffix. `expert_ids(il)` and `indexer_sets(il)` return the selections; `topk_sets_agree()`
  compares two selections with tie tolerance (a key with q.k <= 0 on every head scores exactly
  0, so exact ties at the cut can happen, and implementations break them differently).

Ties. The engines do not specify how a top-k breaks ties: SGLang's and vLLM's indexer top-k and
their router top-k return sets. This reference keeps the lower key position at the indexer cut and
the lower expert id at the router cut (ggml_top_k's stable order, `top_k_desc`); that is a choice of
this file, not part of K3.1, so selections are compared as SETS with a tie tolerance. The Phase 1
gate against the independent model (tools/kimi_k31_gate.py) is built tie-free - no cut of its
golden lies near a tie - so its verdict does not depend on the rule.

Not covered, refused loudly: an indexer that shares selections across layers (index_topk_freq
!= 1, i.e. `attention.indexer.types` not all 1 on the MLA layers), a pooled indexer (`kpool` > 0),
any other `attention.indexer.*` key (the file format has no indexer RoPE, so a converter that
adds one must add a key), grouped expert routing, softmax gating, block size 0 (plain
residuals), an unsplit attn_kv_b, a general.architecture other than kimi-k31.

Real weights. A whole K3.1 file is 100+ GB, so the reference reads only what a call uses:
  * `layers=N` (--layers N) is the truncated model: layers 0..N-1, then (unless head=False,
    --no-head) the output mix, norm and lm_head exactly as a GGUF holding only those N layers
    would apply them. Only those layers' tensors are checked and read. The completed block stack
    at the cut is the seam res_blocks-(N-1) (n_embd, n_blocks, T); with l_out-(N-1) it is the
    whole state the next layer would read.
  * routed experts are read lazily: per chunk of `moe_token_chunk` tokens, each expert some token
    picked is read once and runs every (token, rank) that picked it; no other expert is read. The
    arithmetic is the token-by-token one (a matrix-vector product per token and expert, the
    weighted sum in rank order in float64), so the result does not depend on the chunking.
  * the embedding is read by rows, and without the weight cache the head in slabs of
    `head_chunk_rows` vocabulary rows; `reads` / `bytes_read` count every read from the file.
  * `seams` (--seams) keeps only the seams matching its globs. The library keeps them in `dumps`
    (a 13-layer run over thousands of tokens would hold tens of GB); the CLI hands each to a
    `sink` (`DumpWriter`) that writes it to --dump-dir / --dump-raw-dir as it is made, and keeps
    none. The last layer's output is `trunk_out` whichever seams are kept.
  * memory: the indexer scores `idx_block_bytes` (64 MiB) of queries at a time, so its float64
    temporaries stay bounded; what grows with the prompt is the (n_kv, T) score and mask (with
    their seams), the logits (n_vocab, T) and the (features, T) activations. `memory_estimate()`
    adds up those buffers.
  * --dry-run checks the tensors a run would read, prints the read map (what a forward of the
    given tokens reads, per layer; experts per chunk of moe_token_chunk tokens), probes a small
    slice of every tensor through the same accessors, reports the peak RSS and the forward's
    memory_estimate() for those tokens, and fails if either is over --max-rss-gb; it never reads
    a whole large tensor.
  * `act_q8=32` (--act-q8 32) emulates the engine's device path: the input of every product whose
    weight is a type iq::mv decodes (IQ_MV_TYPES: Q8_0, the K-quants, the i-quants) is rounded to
    int8 per 32 values with an F32 scale (q8_round) - the 2-D GEMVs (attention, indexer, dense and
    shared FFN, router, latent down / up, head) and the routed experts (the latent row into gate /
    up, the SiTU output into down). F32 / F16 / BF16 / MXFP4 weights keep their inputs (cuBLAS,
    exact-expert mode), as do the per-head attn_k_b / attn_v_b and the attention arithmetic. Off by
    default; `act_q8_types` replaces the type set (a test can round an F32 fixture). Only 32 is
    accepted: the CPU lane's Q8_K rounding (256-value blocks) is not modelled.

Usage:
    python ref/kimi_k31.py MODEL.gguf [--tokens 1,2,3 | --tokens-file F] [--layers N [--no-head]]
                           [--eps moonshot|llama.cpp] [--dtype float64] [--act-q8 32]
                           [--seams GLOB,...] [--dump-dir DIR] [--dump-raw-dir DIR] [--greedy N]
    python ref/kimi_k31.py MODEL.gguf --dry-run [--layers N] [--tokens-file F] [--max-rss-gb 20]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

from glm import join_heads, split_heads, top_k_desc  # noqa: E402  (ref/glm.py)
from load import GEOM, Gguf  # noqa: E402  (ref/load.py)

ARCH = "kimi-k31"
F32 = np.float32
F64 = np.float64
DTYPES = {"float32": F32, "float64": F64}

# act_q8: the weight types iq::mv decodes (src/ds4/ds4_iq.cu block_bytes). A product with one of these
# weights takes its input rounded to int8 per 32 values with an F32 scale; F32 / F16 / BF16 go through
# cuBLAS and MXFP4 experts are dequantized exactly on the host (exact-expert mode), so those inputs stay
# as they are. The CPU lane rounds its experts' inputs as Q8_K (256-value blocks), which is not modelled.
IQ_MV_TYPES = frozenset({"Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ1_S", "IQ1_M", "IQ2_XXS",
                         "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"})
ACT_Q8_BLOCK = 32

# Branch-hit coverage, per process (like ref/glm.py's COVER): a fixture that never fires a
# guarded case (indexer discards, ties at the cut, router near-ties, gate saturation) cannot
# prove anything about it.
COVER: Counter = Counter()

EPS_PRESETS = {
    # Moonshot's modeling_kimi_linear.py and the fla kernels it calls
    "moonshot": {"mla_latent": 1e-6, "kda_l2": 1e-6},
    # llama.cpp's kimi-k3 graph: both take attention.layer_norm_rms_epsilon
    "llama.cpp": {"mla_latent": "rms", "kda_l2": "rms"},
}
EPS_NAMES = ("rms", "mla_latent", "kda_l2", "indexer_ln", "router_renorm")

# the indexer keys this reference understands; anything else under attention.indexer. is refused
_INDEXER_KEYS = {"head_count", "key_length", "top_k", "kpool", "types"}


def released_mla_layers(n_layers: int = 93) -> list[int]:
    """0-indexed MLA (= DSA) layers of the released layout: every 4th layer plus the last
    (3, 7, ..., 91, 92 for 93 layers)."""
    return [i for i in range(n_layers) if (i + 1) % 4 == 0 or i == n_layers - 1]


# ================================ the config ================================

class KimiK31Config:
    """Every number the forward pass depends on, from the GGUF metadata (kimi-k31.* keys) with
    the released Kimi K3.1 values as defaults. Key names are llama.cpp's LLM_KV strings.

    eps: "moonshot" (default) or "llama.cpp" (see the module docstring); eps_overrides: a dict
    over EPS_NAMES. index_top_k overrides attention.indexer.top_k (tests use a huge value to
    make the indexer select every visible key)."""

    def __init__(self, meta: dict, *, eps: str = "moonshot", eps_overrides: dict | None = None,
                 index_top_k: int | None = None):
        arch = meta.get("general.architecture", ARCH)
        if arch != ARCH:
            raise ValueError(f"general.architecture is {arch!r}, this reference reads {ARCH!r} only")

        def u(key: str, default):
            v = meta.get(f"{ARCH}.{key}")
            return default if v is None else v

        self.n_vocab = int(u("vocab_size", 163840))
        self.n_embd = int(u("embedding_length", 7168))
        self.n_layers = int(u("block_count", 93))
        self.n_head = int(u("attention.head_count", 96))      # MLA heads and (as upstream) KDA heads

        hkv = u("attention.head_count_kv", None)
        if hkv is None:
            if self.n_layers != 93:
                raise ValueError("attention.head_count_kv is missing and block_count is not the "
                                 "released 93, so the KDA/MLA layout is unknown")
            mla = set(released_mla_layers(self.n_layers))
            hkv = [1 if i in mla else 0 for i in range(self.n_layers)]
        elif isinstance(hkv, (int, float)):
            hkv = [int(hkv)] * self.n_layers
        self.n_head_kv = [int(x) for x in hkv]               # 0 marks a KDA (recurrent) layer
        if len(self.n_head_kv) != self.n_layers:
            raise ValueError(f"attention.head_count_kv has {len(self.n_head_kv)} entries for "
                             f"{self.n_layers} layers")

        # MLA (NoPE). The deepseek2 cache contract: key_length = kv_lora + rope, value_length = kv_lora.
        self.q_lora_rank = int(u("attention.q_lora_rank", 1536))
        self.kv_lora_rank = int(u("attention.kv_lora_rank", 512))
        self.n_rot = int(u("rope.dimension_count", 64))       # the un-rotated 64-wide key part
        self.n_head_k_mla = int(u("attention.key_length_mla", 192))
        self.n_head_v_mla = int(u("attention.value_length_mla", 128))
        self.qk_nope = self.n_head_k_mla - self.n_rot
        kl = int(u("attention.key_length", self.kv_lora_rank + self.n_rot))
        vl = int(u("attention.value_length", self.kv_lora_rank))
        if kl != self.kv_lora_rank + self.n_rot:
            raise ValueError(f"attention.key_length {kl} != kv_lora_rank {self.kv_lora_rank} + "
                             f"rope {self.n_rot}")
        if vl != self.kv_lora_rank:
            raise ValueError(f"attention.value_length {vl} != kv_lora_rank {self.kv_lora_rank}")
        if self.qk_nope <= 0:
            raise ValueError(f"attention.key_length_mla {self.n_head_k_mla} leaves no nope part "
                             f"after rope {self.n_rot}")

        # KDA
        self.ssm_d_conv = int(u("ssm.conv_kernel", 4))
        self.kda_head_dim = int(u("kda.head_dim", 128))
        self.kda_gate_lower_bound = float(u("kda.gate_lower_bound", -5.0))
        if not (-np.inf < self.kda_gate_lower_bound < 0.0):
            raise ValueError(f"kda.gate_lower_bound {self.kda_gate_lower_bound} must be finite and < 0")

        # FFN / MoE
        self.n_ff = int(u("feed_forward_length", 33792))
        self.n_layer_dense_lead = int(u("leading_dense_block_count", 1))
        self.n_expert = int(u("expert_count", 896))
        self.n_expert_used = int(u("expert_used_count", 16))
        self.n_ff_exp = int(u("expert_feed_forward_length", 3072))
        self.n_expert_shared = int(u("expert_shared_count", 2))
        self.n_expert_latent = int(u("expert_latent_length", 3584))
        # a thin file: the routed experts read a narrower latent (routed_down's output) and have a narrower hidden
        # layer; their down projection still writes the full latent. The shared experts keep the full width.
        # Scalar (every MoE layer) or per layer, 0 = that layer is not thin (a mixed file keeps some layers wide).
        def thin(key: str, stock: int) -> list[int]:
            v = u(key, None)
            if v is None:
                return [stock] * self.n_layers
            vs = [int(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else [int(v)] * self.n_layers
            if len(vs) != self.n_layers:
                raise ValueError(f"{ARCH}.{key}: {len(vs)} values for {self.n_layers} layers")
            return [x if x > 0 else stock for x in vs]
        self.n_expert_latent_in = thin("expert_thin_latent_length", self.n_expert_latent)
        self.n_ff_exp_routed = thin("expert_thin_feed_forward_length", self.n_ff_exp)
        self.expert_weights_scale = float(u("expert_weights_scale", 1.0))
        self.expert_weights_norm = bool(u("expert_weights_norm", True))
        gating = int(u("expert_gating_func", 2))
        if gating != 2:
            raise NotImplementedError(f"expert_gating_func {gating}: only sigmoid (2) routing is implemented")
        for key in ("expert_group_count", "expert_group_used_count"):
            if int(u(key, 1)) > 1:
                raise NotImplementedError(f"{key} {u(key, 1)}: grouped expert routing is not implemented")
        if not 0 < self.n_expert_used <= self.n_expert:
            raise ValueError(f"expert_used_count {self.n_expert_used} vs expert_count {self.n_expert}")
        self.situ_beta = float(u("activation.situ_beta", 4.0))
        self.situ_linear_beta = float(u("activation.situ_linear_beta", 25.0))

        self.attn_res_block_size = int(u("attn_res.block_size", 12))
        if self.attn_res_block_size <= 0:
            raise NotImplementedError("attn_res.block_size 0 (plain residuals): K3.1 always uses "
                                      "block attention residuals; that branch is not implemented")

        # the lightning indexer
        for k in meta:
            pre = f"{ARCH}.attention.indexer."
            if k.startswith(pre) and k[len(pre):] not in _INDEXER_KEYS:
                raise NotImplementedError(f"{k}: unknown indexer setting (per-token indexer only)")
        self.idx_n_head = int(u("attention.indexer.head_count", 32))
        self.idx_key_dim = int(u("attention.indexer.key_length", 128))
        self.idx_top_k = int(u("attention.indexer.top_k", 2048)) if index_top_k is None else int(index_top_k)
        if self.idx_n_head <= 0 or self.idx_key_dim <= 0 or self.idx_top_k <= 0:
            raise ValueError("indexer head_count, key_length and top_k must all be > 0")
        kpool = u("attention.indexer.kpool", None)
        if kpool is not None and int(kpool) != 0:
            raise NotImplementedError(f"attention.indexer.kpool {kpool}: a pooled indexer is not K3.1")
        types = u("attention.indexer.types", None)
        if types is not None:
            types = [int(x) for x in types] if isinstance(types, list) else [int(types)] * self.n_layers
            if any(types[i] != 1 for i in self.mla_layers()):
                raise NotImplementedError("attention.indexer.types marks an MLA layer as sharing "
                                          "another layer's selection (index_topk_freq != 1)")

        # epsilons
        self.eps_rms = float(u("attention.layer_norm_rms_epsilon", 1e-5))
        self.eps_indexer_ln = float(u("attention.layer_norm_epsilon", 1e-5))
        if eps not in EPS_PRESETS:
            raise ValueError(f"eps preset {eps!r}: one of {sorted(EPS_PRESETS)}")
        self.eps_preset = eps
        preset = {k: (self.eps_rms if v == "rms" else float(v)) for k, v in EPS_PRESETS[eps].items()}
        self.eps_mla_latent = preset["mla_latent"]
        self.eps_kda_l2 = preset["kda_l2"]
        self.eps_router_renorm = 1e-20
        for k, v in (eps_overrides or {}).items():
            if k not in EPS_NAMES:
                raise ValueError(f"eps override {k!r}: one of {EPS_NAMES}")
            setattr(self, f"eps_{k}", float(v))

        self.d_inner = self.n_head * self.kda_head_dim       # KDA width

    def is_recr(self, il: int) -> bool:
        return self.n_head_kv[il] == 0

    def mla_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if not self.is_recr(i)]

    def eps(self) -> dict:
        return {k: getattr(self, f"eps_{k}") for k in EPS_NAMES}


# ================================ primitives ================================
# ref/glm.py's ggml primitives, with the seam dtype `dt` as a parameter (float32 = the repo's
# convention; float64 makes every op exact enough for tight self-consistency checks).

def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float, dt=F32) -> np.ndarray:
    """ggml_rms_norm over the feature axis (ne[0]); the weight broadcasts over trailing axes."""
    ms = np.mean(x.astype(F64) ** 2, axis=0, keepdims=True)
    y = (x / np.sqrt(ms + eps)).astype(dt)
    return y if w is None else (y * w.reshape((-1,) + (1,) * (x.ndim - 1))).astype(dt)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float, dt=F32) -> np.ndarray:
    """ggml_norm (LLM_NORM): mean and variance over the feature axis, then weight and bias."""
    xf = x.astype(F64)
    mu = xf.mean(axis=0, keepdims=True)
    y = ((xf - mu) / np.sqrt(((xf - mu) ** 2).mean(axis=0, keepdims=True) + eps)).astype(dt)
    shape = (-1,) + (1,) * (x.ndim - 1)
    return ((y * w.reshape(shape)).astype(dt) + b.reshape(shape)).astype(dt)


def sigmoid(x: np.ndarray, dt=F32) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(F64)))).astype(dt)


def silu(x: np.ndarray, dt=F32) -> np.ndarray:
    return (x * sigmoid(x, dt)).astype(dt)


def l2_norm(x: np.ndarray, eps: float, dt=F32) -> np.ndarray:
    """x / sqrt(sum x^2 + eps) over the feature axis (fla's l2norm, llama.cpp's build_gdn_l2_norm)."""
    ss = np.sum(x.astype(F64) ** 2, axis=0, keepdims=True)
    return (x / np.sqrt(ss + eps)).astype(dt)


SITU_BLOCK = 1 << 22      # situ() works on blocks of rows of at most this many values (32 MiB of float64)


def situ(gate: np.ndarray, up: np.ndarray, beta: float, linear_beta: float, dt=F32) -> np.ndarray:
    """SiTU-GLU: beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta), in
    float64. linear_beta <= 0 leaves the up branch linear (llama.cpp's convention). Elementwise, so
    a large input is taken SITU_BLOCK values of rows at a time: the same values, bounded float64
    temporaries (a dense FFN over thousands of tokens would otherwise hold several GB of them)."""
    if gate.ndim >= 2 and gate.size > SITU_BLOCK:
        out = np.empty(gate.shape, dtype=dt)
        rows = max(1, SITU_BLOCK // (gate.size // gate.shape[0]))
        for r0 in range(0, gate.shape[0], rows):
            out[r0:r0 + rows] = _situ(gate[r0:r0 + rows], up[r0:r0 + rows], beta, linear_beta, dt)
        return out
    return _situ(gate, up, beta, linear_beta, dt)


def _situ(gate: np.ndarray, up: np.ndarray, beta: float, linear_beta: float, dt) -> np.ndarray:
    g = gate.astype(F64)
    a = beta * np.tanh(g / beta) / (1.0 + np.exp(-g))
    u = up.astype(F64)
    if linear_beta > 0:
        u = linear_beta * np.tanh(u / linear_beta)
    return (a * u).astype(dt)


def q8_round(x: np.ndarray, block: int, dt=F32) -> np.ndarray:
    """A GEMV input as the engine's int8 path sees it (src/ds4/ds4_gguf.cu q8_quant_block, which
    iq::mv's light and token kernels repeat): the float32 values in blocks of `block` along the
    feature axis, sc = max|x| / 127 in float32, q = x / sc (float32 division) rounded half to even
    (__float2int_rn), and the value q * sc; a block of zeros stays 0. Float64 input is first
    rounded to float32, the engine's activation type."""
    v = np.asarray(x, dtype=F32)
    n = v.shape[0]
    if block <= 0 or n % block:
        raise ValueError(f"q8 blocks of {block}: the input width {n} is not a multiple")
    vb = v.reshape((n // block, block) + v.shape[1:])
    am = np.abs(vb).max(axis=1, keepdims=True)
    sc = (am / F32(127.0)).astype(F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(am > 0, np.rint(vb / sc), F32(0.0))
    return (q.astype(F64) * sc.astype(F64)).reshape(v.shape).astype(dt)


def short_conv(x_proj: np.ndarray, w: np.ndarray, hist: np.ndarray, dt=F32) -> tuple[np.ndarray, np.ndarray]:
    """Depthwise causal conv + SiLU (fla ShortConvolution, ggml_ssm_conv). x_proj is (d_inner, T),
    w is (d_conv, d_inner) with taps oldest first (w[d_conv-1] multiplies the current token),
    hist is (d_inner, d_conv-1): the inputs at the d_conv-1 previous positions, oldest first
    (zeros at a sequence start). Returns (silu(conv), new hist)."""
    d_conv = w.shape[0]
    T = x_proj.shape[1]
    seq = np.concatenate([hist, x_proj], axis=1).astype(F64)          # (d_inner, d_conv-1+T)
    out = np.zeros(x_proj.shape, dtype=F64)
    for j in range(d_conv):
        out += w[j].astype(F64)[:, None] * seq[:, j:j + T]           # tap j reads position t-(d_conv-1)+j
    hist_new = np.ascontiguousarray(seq[:, T:], dtype=dt) if d_conv > 1 else hist
    return silu(out.astype(dt), dt), hist_new


def kda_scan(q: np.ndarray, k: np.ndarray, v: np.ndarray, g: np.ndarray, beta: np.ndarray,
             S: np.ndarray, scale: float, dt=F32) -> tuple[np.ndarray, np.ndarray]:
    """The KDA recurrence (gated delta rule, decay per key channel), all heads at once.
    q, k, g: (dk, H, T); v: (dv, H, T); beta: (H, T); S: (H, dk, dv) float64, the state before
    the first token. Per token t and head h:
        S <- diag(exp(g_t)) S ;  S <- S + k_t (beta_t (v_t - S^T k_t))^T ;  o_t = scale * S^T q_t
    Returns o (dv, H, T) in dt and the final state (float64)."""
    S = np.array(S, dtype=F64, copy=True)
    T = q.shape[2]
    o = np.empty((v.shape[0], v.shape[1], T), dtype=dt)
    for t in range(T):
        qt, kt, vt = (a[:, :, t].T.astype(F64) for a in (q, k, v))  # (H, dk) / (H, dv)
        S *= np.exp(g[:, :, t].T.astype(F64))[:, :, None]
        delta = (vt - np.einsum("hkv,hk->hv", S, kt)) * beta[:, t].astype(F64)[:, None]
        S += np.einsum("hk,hv->hkv", kt, delta)
        o[:, :, t] = (np.einsum("hkv,hk->hv", S, qt) * scale).T.astype(dt)
    return o, S


def softmax_cols(s: np.ndarray, axis: int) -> np.ndarray:
    """float64 softmax along `axis` (-inf entries get exactly 0)."""
    z = s - s.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def topk_sets_agree(scores, sel_a, sel_b, tol: float = 0.0) -> bool:
    """Do two top-k selections of one query agree up to ties? `scores` holds that query's
    scores indexed by key position (any length covering both selections). True when the sets
    are equal, or have the same size and every key in their symmetric difference scores within
    `tol` of the cut (the smallest score either selection kept)."""
    a = {int(i) for i in np.asarray(sel_a).ravel() if i >= 0}
    b = {int(i) for i in np.asarray(sel_b).ravel() if i >= 0}
    if a == b:
        return True
    if len(a) != len(b):
        return False
    sc = np.asarray(scores, dtype=F64)
    if not all(np.isfinite(sc[i]) for i in a | b):
        return False                                  # a selection holds a masked (future) key
    cut = max(min(sc[i] for i in a), min(sc[i] for i in b))   # the k-th score, if either is exact
    return all(abs(sc[i] - cut) <= tol for i in a ^ b)


# ================================ the state ================================

class KimiK31State:
    """What one sequence carries from one forward call to the next."""

    def __init__(self, layers: int | None = None):
        self.n_past = 0
        self.layers = layers                          # the model depth it belongs to (set on first use)
        self.conv: dict[int, list[np.ndarray]] = {}   # KDA il -> [q, k, v] (d_inner, d_conv-1) inputs
        self.ssm: dict[int, np.ndarray] = {}          # KDA il -> (H, dk, dv) float64 recurrent state
        self.kv: dict[int, np.ndarray] = {}           # MLA il -> (kv_lora, n_past) normed latents
        self.k_pe: dict[int, np.ndarray] = {}         # MLA il -> (rope, n_past) raw key parts
        self.idx_k: dict[int, np.ndarray] = {}        # MLA il -> (idx_dim, n_past) indexer keys


# ================================ the model ================================

class KimiK31:
    """The reference model over one kimi-k31 GGUF.

    indexer=False runs every MLA layer dense-causal (for equivalence tests; not a K3.1 mode).
    dtype "float32" (default) keeps float32 at every seam like ref/glm.py; "float64" carries
    float64 throughout (weights upcast exactly), for tight self-consistency and gate checks.
    cache_weights keeps dequantized tensors in memory; None (default) caches when the GGUF is
    under 4 GiB, so the tiny fixture is fast and a real model is read on demand.

    layers: run layers 0..layers-1 only (None: all), the truncated model of the module docstring;
    head=False stops after the last of them (forward returns None; trunk_out holds the output).
    act_q8: 0 (off) or 32, the engine's int8 rounding of the inputs of products with weights of a
    type in act_q8_types (None: IQ_MV_TYPES). seams: fnmatch globs over seam names ("l_out-12",
    "res_blocks-*"); only matching seams are kept (None: all; expert_ids / indexer_sets read the
    ffn_moe_topk / top_k seams), in dumps or, with sink set, handed to the sink.
    moe_token_chunk: tokens per pass of the lazy expert loop. head_chunk_rows: vocabulary rows
    per head slab; None reads the head whole when the weights are cached and in slabs of 8192
    rows otherwise. idx_block_bytes: the indexer's float64 working set per query block."""

    def __init__(self, path, *, eps: str = "moonshot", eps_overrides: dict | None = None,
                 index_top_k: int | None = None, indexer: bool = True,
                 dtype: str = "float32", cache_weights: bool | None = None,
                 layers: int | None = None, head: bool = True, act_q8: int = 0,
                 seams: list[str] | None = None, moe_token_chunk: int = 2048,
                 head_chunk_rows: int | None = None, act_q8_types=None,
                 idx_block_bytes: int = 64 << 20):
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r}: one of {sorted(DTYPES)}")
        self.dt = DTYPES[dtype]
        self.g = Gguf(str(path))
        self.cfg = KimiK31Config(self.g.g.metadata, eps=eps, eps_overrides=eps_overrides,
                                 index_top_k=index_top_k)
        c = self.cfg
        self.n_run = c.n_layers if layers is None else int(layers)
        if not 1 <= self.n_run <= c.n_layers:
            raise ValueError(f"layers {layers}: the file has {c.n_layers} layers (1..{c.n_layers})")
        self.head = bool(head)
        self.truncated = self.n_run < c.n_layers or not self.head      # the run stops before the file's end
        self.trunk_out: np.ndarray | None = None
        self.act_q8 = int(act_q8)
        if self.act_q8 not in (0, ACT_Q8_BLOCK):
            raise ValueError(f"act_q8 {act_q8}: 0 (off) or {ACT_Q8_BLOCK} (iq::mv's int8 blocks); the CPU "
                             "lane's Q8_K rounding (256-value blocks) is not modelled")
        self.act_q8_types = IQ_MV_TYPES if act_q8_types is None else frozenset(act_q8_types)
        self.seams = None if seams is None else [str(s) for s in seams]
        if moe_token_chunk < 1 or (head_chunk_rows is not None and head_chunk_rows < 1):
            raise ValueError("moe_token_chunk and head_chunk_rows must be >= 1")
        self.moe_token_chunk = int(moe_token_chunk)
        self.head_chunk_rows = head_chunk_rows
        if idx_block_bytes < 1:
            raise ValueError("idx_block_bytes must be >= 1")
        self.idx_block_bytes = int(idx_block_bytes)
        self.use_indexer = indexer
        if cache_weights is None:
            cache_weights = sum(p.stat().st_size for p in self.g.paths) < (4 << 30)
        self._cache: dict | None = {} if cache_weights else None
        self.dumps: dict[str, np.ndarray] = {}
        # sink(name, array): when set, cb() hands every kept seam to it (in the seam dtype, possibly a
        # view: use it at once) instead of storing it in dumps; the CLI writes seams to disk this way
        self.sink = None
        # every read from the file: (tensor, None) whole, (tensor, e) expert e, (tensor, "rows") rows,
        # (tensor, "slice") a dry-run probe's first outer slice
        self.reads: Counter = Counter()
        self.bytes_read = 0
        self._check_tensors()
        if self.act_q8:
            for name, shape in self.expected_tensors().items():
                if self.q8_rounds(name) and shape[-1] % self.act_q8:
                    raise ValueError(f"act_q8 {self.act_q8}: {name} has {shape[-1]} inputs, not a multiple")

    def close(self) -> None:
        self.g.close()

    # ---- weights; tensor() is (out, in), so every use below is y = W @ x
    def _nbytes(self, name: str, elements: int) -> int:
        el, by = GEOM[self.g.info(name).type_name]
        return elements // el * by

    def _count(self, name: str, what, n: int, elements: int) -> None:
        self.reads[(name, what)] += n
        self.bytes_read += self._nbytes(name, elements)

    def t(self, name: str) -> np.ndarray:
        if self._cache is not None and name in self._cache:
            return self._cache[name]
        w = self.g.tensor(name).astype(self.dt)
        self._count(name, None, 1, w.size)
        if self._cache is not None:
            self._cache[name] = w
        return w

    def layer_t(self, il: int, base: str, suffix: str = "weight") -> np.ndarray:
        return self.t(f"blk.{il}.{base}.{suffix}")

    def expert_w(self, il: int, base: str, e: int) -> np.ndarray:
        """One routed expert's (out, in) matrix, read lazily (ref/load.py's expert())."""
        key = (il, base, e)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        name = f"blk.{il}.{base}.weight"
        w = self.g.expert(name, e).astype(self.dt)
        self._count(name, int(e), 1, w.size)
        if self._cache is not None:
            self._cache[key] = w
        return w

    def embed(self, tokens: list[int]) -> np.ndarray:
        """(n_embd, T) embedding rows of `tokens`, each distinct row read once (never the table)."""
        name = "token_embd.weight"
        uniq, inv = np.unique(np.asarray(tokens, dtype=np.int64), return_inverse=True)
        rows = self.g.rows(name, uniq)
        self._count(name, "rows", len(uniq), rows.size)
        return rows[inv].T.astype(self.dt)

    def head_logits(self, cur: np.ndarray) -> np.ndarray:
        """lm_head: one product when the weights are cached (unless head_chunk_rows is set),
        otherwise slabs of vocabulary rows, so the head is never resident whole."""
        name = "output.weight"
        cur = self.q8_in(name, cur)
        if self.head_chunk_rows is None and self._cache is not None:
            return self.t(name) @ cur
        R, V = self.head_chunk_rows or 8192, self.cfg.n_vocab
        out = np.empty((V, cur.shape[1]), dtype=self.dt)
        for r0 in range(0, V, R):
            w = self.g.row_range(name, r0, min(V, r0 + R))
            self._count(name, "rows", w.shape[0], w.size)
            out[r0:r0 + w.shape[0]] = w.astype(self.dt) @ cur
        return out

    def q8_rounds(self, name: str) -> bool:
        """act_q8 is on and `name` is a matrix whose products take int8 inputs: a 2-D GEMV weight or a
        routed-expert tensor of a type in act_q8_types (IQ_MV_TYPES). Not the embedding (read by
        rows), the per-head attn_k_b / attn_v_b (the MLA kernel's own arithmetic), vectors or convs."""
        if (not self.act_q8 or name == "token_embd.weight" or ".ssm_conv1d_" in name
                or name.endswith(("attn_k_b.weight", "attn_v_b.weight"))):
            return False
        t = self.g.info(name)
        return len(t.shape) >= 2 and t.type_name in self.act_q8_types

    def q8_in(self, name: str, x: np.ndarray) -> np.ndarray:
        """x as a product with weight `name` takes it: rounded as iq::mv rounds it (q8_round) when
        q8_rounds(name), otherwise x itself."""
        return q8_round(x, self.act_q8, self.dt) if self.q8_rounds(name) else x

    def mv(self, name: str, x: np.ndarray) -> np.ndarray:
        """W @ x for the 2-D weight `name` (y = W @ x IS ggml_mul_mat), x taken as q8_in takes it."""
        return self.t(name) @ self.q8_in(name, x)

    def layer_mv(self, il: int, base: str, x: np.ndarray) -> np.ndarray:
        return self.mv(f"blk.{il}.{base}.weight", x)

    def wants(self, name: str, il: int = -1) -> bool:
        if self.seams is None:
            return True
        key = f"{name}-{il}" if il >= 0 else name
        return any(fnmatch.fnmatchcase(key, s) for s in self.seams)

    def cb(self, name: str, x: np.ndarray, il: int = -1) -> None:
        if not self.wants(name, il):
            return
        a, key = np.asarray(x), (f"{name}-{il}" if il >= 0 else name)
        if self.sink is not None:
            self.sink(key, a if a.dtype == np.int64 else np.asarray(a, dtype=self.dt))
            return
        self.dumps[key] = a.copy() if a.dtype == np.int64 else np.array(a, dtype=self.dt, copy=True)

    def _check_tensors(self) -> None:
        """Every tensor the forward pass reads, with its (out, in) shape, before any compute."""
        c = self.cfg
        for name, shape in self.expected_tensors().items():
            if name not in self.g._by_name:
                raise ValueError(f"{self.g.path.name}: missing tensor {name}")
            got = tuple(reversed(self.g.info(name).shape))
            if shape is None:
                if int(np.prod(got)) != c.d_inner * c.ssm_d_conv:
                    raise ValueError(f"{name}: shape {got}, expected d_inner*d_conv elements")
            elif got != shape:
                raise ValueError(f"{name}: shape {got}, expected {shape}")

    def expected_tensors(self) -> dict[str, tuple | None]:
        """name -> (out, in) shape of every tensor a forward of this (possibly truncated) model
        reads; None: a conv weight checked by element count."""
        c = self.cfg
        D, H, dk, di = c.n_embd, c.n_head, c.kda_head_dim, c.d_inner
        exp: dict[str, tuple] = {"token_embd.weight": (c.n_vocab, D)}
        if self.head:
            exp.update({"output_norm.weight": (D,), "output_res_score.weight": (D,),
                        "output.weight": (c.n_vocab, D)})
        for il in range(self.n_run):
            b = f"blk.{il}."
            for n in ("attn_norm", "attn_res_score", "ffn_res_score", "ffn_norm"):
                exp[b + n + ".weight"] = (D,)
            if c.is_recr(il):
                for s in "qkv":
                    exp[f"{b}attn_{s}.weight"] = (di, D)
                    exp[f"{b}ssm_conv1d_{s}.weight"] = None      # 3-D or 4-D, d_inner * d_conv elements
                exp.update({b + "ssm_f_a.weight": (dk, D), b + "ssm_f_b.weight": (di, dk),
                            b + "ssm_beta.weight": (H, D), b + "ssm_a": (H,), b + "ssm_dt.bias": (di,),
                            b + "ssm_g.weight": (di, D), b + "ssm_norm.weight": (dk,),
                            b + "attn_output.weight": (D, di)})
            else:
                if f"{b}attn_kv_b.weight" in self.g._by_name:
                    raise NotImplementedError(f"layer {il}: unsplit attn_kv_b; the kimi-k31 file "
                                              "carries attn_k_b / attn_v_b")
                vh = c.n_head_v_mla
                exp.update({b + "attn_q_a.weight": (c.q_lora_rank, D),
                            b + "attn_q_a_norm.weight": (c.q_lora_rank,),
                            b + "attn_q_b.weight": (H * c.n_head_k_mla, c.q_lora_rank),
                            b + "attn_kv_a_mqa.weight": (c.kv_lora_rank + c.n_rot, D),
                            b + "attn_kv_a_norm.weight": (c.kv_lora_rank,),
                            b + "attn_k_b.weight": (H, c.kv_lora_rank, c.qk_nope),
                            b + "attn_v_b.weight": (H, vh, c.kv_lora_rank),
                            b + "attn_gate.weight": (H * vh, D), b + "attn_output.weight": (D, H * vh),
                            b + "indexer.attn_q_b.weight": (c.idx_n_head * c.idx_key_dim, c.q_lora_rank),
                            b + "indexer.attn_k.weight": (c.idx_key_dim, D),
                            b + "indexer.k_norm.weight": (c.idx_key_dim,),
                            b + "indexer.k_norm.bias": (c.idx_key_dim,),
                            b + "indexer.proj.weight": (c.idx_n_head, D)})
            if il < c.n_layer_dense_lead:
                exp.update({b + "ffn_gate.weight": (c.n_ff, D), b + "ffn_up.weight": (c.n_ff, D),
                            b + "ffn_down.weight": (D, c.n_ff)})
            else:
                E, F, L, S = c.n_expert, c.n_ff_exp, c.n_expert_latent, c.n_expert_shared
                Li, Fr = c.n_expert_latent_in[il], c.n_ff_exp_routed[il]   # (L, F unless this layer is thin)
                exp.update({b + "ffn_gate_inp.weight": (E, D), b + "exp_probs_b.bias": (E,),
                            b + "ffn_routed_down.weight": (Li, D), b + "ffn_routed_up.weight": (D, L),
                            b + "ffn_routed_norm.weight": (L,),
                            b + "ffn_gate_exps.weight": (E, Fr, Li), b + "ffn_up_exps.weight": (E, Fr, Li),
                            b + "ffn_down_exps.weight": (E, L, Fr),
                            b + "ffn_gate_shexp.weight": (F * S, D), b + "ffn_up_shexp.weight": (F * S, D),
                            b + "ffn_down_shexp.weight": (D, F * S)})
        return exp

    # ================================ block attention residuals ================================

    def res_mix(self, blocks: list[np.ndarray], cur: np.ndarray, score_w: np.ndarray,
                il: int, site: str) -> np.ndarray:
        """Softmax mix of [blocks..., cur], scored on the RMS-normalized candidates against the
        fused res_score vector, summed over the RAW candidates. No blocks: cur unchanged."""
        if not blocks:
            return cur
        V = np.stack(blocks + [cur], axis=1).astype(F64)                # (n_embd, n_cand, T)
        ms = np.mean(V ** 2, axis=0)                                    # (n_cand, T)
        s = np.einsum("d,dnt->nt", score_w.astype(F64), V) / np.sqrt(ms + self.cfg.eps_rms)
        p = softmax_cols(s, axis=0)
        out = np.einsum("dnt,nt->dt", V, p).astype(self.dt)
        tag = "" if site == "out" else f"_{site}"
        self.cb(f"res_probs{tag}", p, il)
        self.cb(f"res_mix{tag}", out, il)
        return out

    # ================================ KDA ================================

    def kda_layer(self, x: np.ndarray, il: int, st: KimiK31State) -> np.ndarray:
        c, dt = self.cfg, self.dt
        dk, H = c.kda_head_dim, c.n_head
        hist = st.conv.get(il) or [np.zeros((c.d_inner, c.ssm_d_conv - 1), dtype=dt) for _ in range(3)]
        conv, new_hist = [], []
        for j, s in enumerate("qkv"):
            w = self.layer_t(il, f"ssm_conv1d_{s}").reshape(c.d_inner, c.ssm_d_conv).T   # (d_conv, d_inner)
            y, h = short_conv(self.layer_mv(il, f"attn_{s}", x), w, hist[j], dt)
            conv.append(y)
            new_hist.append(h)
            self.cb(f"kda_{s}_conv", y, il)
        st.conv[il] = new_hist

        # decay gate: lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)); ssm_a = -exp(A_log)
        pre = self.layer_mv(il, "ssm_f_b", self.layer_mv(il, "ssm_f_a", x))
        pre = pre + self.layer_t(il, "ssm_dt", "bias")[:, None]
        exp_a = -self.t(f"blk.{il}.ssm_a").astype(F64)                  # (H,) = exp(A_log)
        arg = (split_heads(pre, dk, H).astype(F64) * exp_a[None, :, None]).astype(dt)
        g1 = (c.kda_gate_lower_bound * sigmoid(arg, dt).astype(F64)).astype(dt)
        COVER["kda_gate_saturated"] += int((arg > 8.0).sum())          # gate sits on the lower bound
        self.cb("kda_g1", g1, il)

        beta = sigmoid(self.layer_mv(il, "ssm_beta", x), dt)            # (H, T)
        self.cb("kda_beta", beta, il)

        q = l2_norm(split_heads(conv[0], dk, H), c.eps_kda_l2, dt)
        k = l2_norm(split_heads(conv[1], dk, H), c.eps_kda_l2, dt)
        v = split_heads(conv[2], dk, H)
        S0 = st.ssm.get(il)
        if S0 is None:
            S0 = np.zeros((H, dk, dk), dtype=F64)
        o, st.ssm[il] = kda_scan(q, k, v, g1, beta, S0, dk ** -0.5, dt)
        self.cb("kda_scan_out", o, il)

        g2 = split_heads(self.layer_mv(il, "ssm_g", x), dk, H)
        normed = rms_norm(o, self.layer_t(il, "ssm_norm"), c.eps_rms, dt)
        self.cb("kda_g2", g2, il)
        self.cb("kda_normed", normed, il)
        out = self.layer_mv(il, "attn_output", join_heads(normed * sigmoid(g2, dt)))
        self.cb("kda_out", out, il)
        return out

    # ================================ the lightning indexer ================================

    def indexer(self, x: np.ndarray, qr: np.ndarray, il: int, st: KimiK31State,
                pos: np.ndarray) -> list[np.ndarray]:
        """Per query, the selected key positions (descending score, ties to the lower position)."""
        c = self.cfg
        Hi, Di = c.idx_n_head, c.idx_key_dim
        iq = split_heads(self.layer_mv(il, "indexer.attn_q_b", qr), Di, Hi)          # (Di, Hi, T)
        ik = layer_norm(self.layer_mv(il, "indexer.attn_k", x), self.layer_t(il, "indexer.k_norm"),
                        self.layer_t(il, "indexer.k_norm", "bias"), c.eps_indexer_ln, self.dt)  # (Di, T)
        iw = self.layer_mv(il, "indexer.proj", x)                                     # (Hi, T)
        self.cb("indexer_q", iq, il)
        self.cb("indexer_k", ik, il)
        self.cb("indexer_weights", iw, il)

        prev = st.idx_k.get(il)
        ik_all = ik if prev is None else np.concatenate([prev, ik], axis=1)          # (Di, n_kv)
        st.idx_k[il] = ik_all
        n_kv = ik_all.shape[1]
        w = iw.astype(F64) * (Hi * Di) ** -0.5
        iq64, ik64 = iq.astype(F64), ik_all.astype(F64)
        score = np.empty((n_kv, len(pos)), dtype=F64)
        # queries a block, (n_kv, Hi, tb) float64 temporaries each: the same sums as one block of all T,
        # bit for bit, as long as no block holds a lone query when T > 1 (numpy sums a lone column's
        # heads pairwise, several columns' in head order), so a last block of one joins the one before
        tb = max(2, self.idx_block_bytes // (n_kv * Hi * 8))
        cuts = list(range(0, len(pos), tb)) + [len(pos)]
        if len(cuts) > 2 and cuts[-1] - cuts[-2] == 1:
            del cuts[-2]
        for t0, t1 in zip(cuts, cuts[1:]):
            kq = np.einsum("dht,ds->sht", iq64[:, :, t0:t1], ik64)                   # (n_kv, Hi, tb)
            score[:, t0:t1] = (np.maximum(kq, 0.0) * w[None, :, t0:t1]).sum(axis=1)
            del kq
        future = np.arange(n_kv)[:, None] > pos[None, :]
        score[future] = -np.inf                       # -inf, not 0: a visible score can be negative
        self.cb("indexer_score", score, il)

        k = c.idx_top_k
        sel, top = [], np.full((min(k, n_kv), len(pos)), -1, dtype=np.int64)
        for t, p in enumerate(pos):
            vis = score[:p + 1, t]
            idx = top_k_desc(vis, k)
            sel.append(idx)
            top[:len(idx), t] = idx
            COVER["idx_negative_visible_scores"] += int((vis < 0).sum())
            if len(vis) > k:
                COVER["idx_queries_over_top_k"] += 1
                ordered = np.sort(vis)[::-1]
                if ordered[k - 1] == ordered[k]:
                    COVER["idx_ties_at_cut"] += 1
                    COVER["idx_zero_ties_at_cut"] += int(ordered[k] == 0.0)
                # a 0-filled row (instead of -inf) would rank a future key above this cut
                COVER["idx_zero_fill_would_pick_future"] += int(ordered[k - 1] < 0.0)
        self.cb("top_k", top, il)
        return sel

    # ================================ MLA (NoPE, absorbed) ================================

    def mla_layer(self, x: np.ndarray, il: int, st: KimiK31State, pos: np.ndarray) -> np.ndarray:
        c, dt = self.cfg, self.dt
        H, T = c.n_head, x.shape[1]
        qr = rms_norm(self.layer_mv(il, "attn_q_a", x), self.layer_t(il, "attn_q_a_norm"), c.eps_mla_latent, dt)
        self.cb("qr", qr, il)
        q = split_heads(self.layer_mv(il, "attn_q_b", qr), c.n_head_k_mla, H)       # (192, H, T)
        q_nope, q_pe = q[:c.qk_nope], q[c.qk_nope:]

        kv_pe = self.layer_mv(il, "attn_kv_a_mqa", x)                              # (kv_lora + rope, T)
        kv_cmpr = rms_norm(kv_pe[:c.kv_lora_rank], self.layer_t(il, "attn_kv_a_norm"), c.eps_mla_latent, dt)
        k_pe = np.ascontiguousarray(kv_pe[c.kv_lora_rank:])                       # raw, never rotated
        self.cb("kv_cmpr", kv_cmpr, il)
        self.cb("k_pe", k_pe, il)

        wk_b = self.layer_t(il, "attn_k_b")                                        # (H, kv_lora, nope)
        q_abs = np.empty((c.kv_lora_rank, H, T), dtype=dt)                        # (kv_lora, H, T)
        for h in range(H):
            q_abs[:, h, :] = wk_b[h] @ q_nope[:, h, :]
        self.cb("q_absorbed", q_abs, il)

        lat_all = kv_cmpr if il not in st.kv else np.concatenate([st.kv[il], kv_cmpr], axis=1)
        pe_all = k_pe if il not in st.k_pe else np.concatenate([st.k_pe[il], k_pe], axis=1)
        st.kv[il], st.k_pe[il] = lat_all, pe_all
        n_kv = lat_all.shape[1]

        if self.use_indexer:
            sel = self.indexer(x, qr, il, st, pos)
        else:
            sel = [np.arange(p + 1) for p in pos]
        mask = np.full((n_kv, T), -np.inf, dtype=dt)
        for t in range(T):
            mask[sel[t], t] = 0.0
        self.cb("kq_mask_top_k", mask, il)

        wv_b = self.layer_t(il, "attn_v_b").astype(F64)                            # (H, v_head, kv_lora)
        scale = c.n_head_k_mla ** -0.5
        out = np.empty((c.n_head_v_mla * H, T), dtype=dt)
        for t in range(T):
            cells = np.sort(sel[t])                       # a set: attend in position order
            L = lat_all[:, cells].astype(F64)
            P = pe_all[:, cells].astype(F64)
            s = (q_abs[:, :, t].T.astype(F64) @ L + q_pe[:, :, t].T.astype(F64) @ P) * scale   # (H, n_sel)
            p = softmax_cols(s, axis=1)
            ctx = L @ p.T                                                           # (kv_lora, H)
            oh = np.einsum("hvc,ch->hv", wv_b, ctx)                                 # (H, v_head)
            out[:, t] = oh.reshape(-1).astype(dt)                                   # row e + v_head*h
        self.cb("kqv_out", out, il)

        gated = out * sigmoid(self.layer_mv(il, "attn_gate", x), dt)
        self.cb("mla_gated", gated, il)
        attn = self.layer_mv(il, "attn_output", gated)
        self.cb("mla_out", attn, il)
        return attn

    # ================================ the FFNs ================================

    def dense_ffn(self, x: np.ndarray, il: int) -> np.ndarray:
        c = self.cfg
        h = situ(self.layer_mv(il, "ffn_gate", x), self.layer_mv(il, "ffn_up", x), c.situ_beta, c.situ_linear_beta,
                 self.dt)
        out = self.layer_mv(il, "ffn_down", h)
        self.cb("ffn_out", out, il)
        return out

    def moe_ffn(self, x: np.ndarray, il: int) -> np.ndarray:
        c, dt = self.cfg, self.dt
        K, T = c.n_expert_used, x.shape[1]
        logits = self.layer_mv(il, "ffn_gate_inp", x)                               # full-width x
        probs = sigmoid(logits, dt)                                                 # unbiased
        biased = (probs + self.layer_t(il, "exp_probs_b", "bias")[:, None]).astype(dt)
        self.cb("ffn_moe_logits", logits, il)
        self.cb("ffn_moe_probs", probs, il)
        self.cb("ffn_moe_probs_biased", biased, il)

        z = self.layer_mv(il, "ffn_routed_down", x)                                 # (latent, T)
        self.cb("ffn_routed_in", z, il)
        ids = np.empty((K, T), dtype=np.int64)
        w_raw = np.empty((K, T), dtype=dt)
        w_norm = np.empty((K, T), dtype=dt)
        w_scaled = np.empty((K, T), dtype=dt)
        w_sum = np.empty((K, T), dtype=F64)                 # the float64 weights the sum uses
        for t in range(T):
            sel = top_k_desc(biased[:, t], K)
            ids[:, t] = sel
            w = probs[sel, t].astype(F64)
            w_raw[:, t] = w
            if c.expert_weights_norm:
                w = w / (w.sum() + c.eps_router_renorm)
            w_norm[:, t] = w
            w = w * c.expert_weights_scale
            w_scaled[:, t] = w
            w_sum[:, t] = w
            ordered = np.sort(biased[:, t])[::-1]
            if len(ordered) > K:
                COVER["moe_selection_near_ties"] += int(ordered[K - 1] - ordered[K] < 1e-4)
            COVER["moe_bias_changed_selection"] += int(set(sel) != set(top_k_desc(probs[:, t], K)))
        moe = self.routed_experts(il, z, ids, w_sum)
        self.cb("ffn_moe_topk", ids, il)
        self.cb("ffn_moe_weights", w_raw, il)
        self.cb("ffn_moe_weights_norm", w_norm, il)
        self.cb("ffn_moe_weights_scaled", w_scaled, il)
        self.cb("ffn_moe_out", moe, il)

        r = rms_norm(moe, self.layer_t(il, "ffn_routed_norm"), c.eps_rms, dt)      # AFTER the weighted sum
        self.cb("ffn_routed_normed", r, il)
        r = self.layer_mv(il, "ffn_routed_up", r)
        self.cb("ffn_routed_out", r, il)

        sh = situ(self.layer_mv(il, "ffn_gate_shexp", x), self.layer_mv(il, "ffn_up_shexp", x),
                  c.situ_beta, c.situ_linear_beta, dt)
        sh = self.layer_mv(il, "ffn_down_shexp", sh)
        self.cb("ffn_shexp", sh, il)
        out = (r + sh).astype(dt)
        self.cb("ffn_out", out, il)
        return out

    def routed_experts(self, il: int, z: np.ndarray, ids: np.ndarray, w: np.ndarray) -> np.ndarray:
        """The weighted routed-expert sum (latent, T), before routed_norm: token t's experts
        ids[:, t] with the float64 weights w[:, t], summed in rank order in float64, one
        matrix-vector product per token and expert (the token-by-token arithmetic). Lazy: per
        chunk of moe_token_chunk tokens each expert some token picked is read once and runs every
        (token, rank) that picked it; no other expert is read."""
        c, dt = self.cfg, self.dt
        K, T = ids.shape
        names = [f"blk.{il}.{b}.weight" for b in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")]
        zg = self.q8_in(names[0], z)                        # the gate / up inputs (act_q8), rounded once
        zu = zg if self.q8_rounds(names[1]) == self.q8_rounds(names[0]) else self.q8_in(names[1], z)
        moe = np.empty((c.n_expert_latent, T), dtype=dt)
        for c0 in range(0, T, self.moe_token_chunk):
            c1 = min(T, c0 + self.moe_token_chunk)
            users: dict[int, list[tuple[int, int]]] = {}
            for t in range(c0, c1):
                for i in range(K):
                    users.setdefault(int(ids[i, t]), []).append((i, t))
            part = np.empty((K, c.n_expert_latent, c1 - c0), dtype=dt)
            for e in sorted(users):                         # file order: one pass over the layer's experts
                wg, wu, wd = (self.expert_w(il, b, e) for b in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"))
                for i, t in users[e]:
                    h = situ(wg @ zg[:, t], wu @ zu[:, t], c.situ_beta, c.situ_linear_beta, dt)
                    part[i, :, t - c0] = wd @ self.q8_in(names[2], h)
                del wg, wu, wd
            for t in range(c0, c1):
                acc = np.zeros(c.n_expert_latent, dtype=F64)
                for i in range(K):
                    acc += w[i, t] * part[i, :, t - c0].astype(F64)
                moe[:, t] = acc.astype(dt)
        return moe

    # ================================ the trunk ================================

    def new_state(self) -> KimiK31State:
        return KimiK31State(self.n_run)

    def forward(self, tokens, state: KimiK31State | None = None) -> np.ndarray | None:
        """Run `tokens` (a fresh sequence, or the continuation of `state`); returns the
        (n_vocab, T) logits of every position, or None when the model has no head (head=False);
        trunk_out is the last layer's output l_out-(n_run-1) either way. `state` is updated in place."""
        c, dt = self.cfg, self.dt
        st = state if state is not None else self.new_state()
        if st.layers is None:
            st.layers = self.n_run
        elif st.layers != self.n_run:
            raise ValueError(f"the state belongs to a {st.layers}-layer run, this model runs {self.n_run}")
        tokens = [int(x) for x in tokens]
        if not tokens:
            raise ValueError("no tokens")
        if min(tokens) < 0 or max(tokens) >= c.n_vocab:
            raise ValueError(f"token id out of range 0..{c.n_vocab - 1}")
        self.dumps = {}
        T = len(tokens)
        pos = np.arange(st.n_past, st.n_past + T)
        x = self.embed(tokens)                                                       # (n_embd, T)
        self.cb("inp_embd", x)

        blocks: list[np.ndarray] = []                     # completed blocks; per token, nothing cached
        prefix = x                                        # the current block's partial sum
        for il in range(self.n_run):
            cur = self.res_mix(blocks, prefix, self.layer_t(il, "attn_res_score"), il, "attn")
            banked = il % c.attn_res_block_size == 0
            if banked:                                    # push the RAW incoming partial sum
                blocks.append(prefix)
                COVER["attn_res_block_pushes"] += 1
            cur = rms_norm(cur, self.layer_t(il, "attn_norm"), c.eps_rms, dt)
            self.cb("attn_norm", cur, il)
            a = self.kda_layer(cur, il, st) if c.is_recr(il) else self.mla_layer(cur, il, st, pos)
            prefix = a if banked else (prefix + a).astype(dt)
            self.cb("prefix_sum_attn", prefix, il)

            cur = self.res_mix(blocks, prefix, self.layer_t(il, "ffn_res_score"), il, "ffn")
            cur = rms_norm(cur, self.layer_t(il, "ffn_norm"), c.eps_rms, dt)
            self.cb("ffn_norm", cur, il)
            f = self.dense_ffn(cur, il) if il < c.n_layer_dense_lead else self.moe_ffn(cur, il)
            prefix = (prefix + f).astype(dt)
            self.cb("l_out", prefix, il)

        self.trunk_out = prefix                           # l_out-(n_run-1), whichever seams are kept
        if self.truncated and self.wants("res_blocks", self.n_run - 1):
            self.cb("res_blocks", np.stack(blocks, axis=1), self.n_run - 1)   # with l_out: the next layer's input
        st.n_past += T
        if not self.head:
            return None
        cur = self.res_mix(blocks, prefix, self.t("output_res_score.weight"), -1, "out")
        cur = rms_norm(cur, self.t("output_norm.weight"), c.eps_rms, dt)
        self.cb("result_norm", cur)
        logits = self.head_logits(cur)
        self.cb("result_output", logits)
        return logits

    # ---- selections of the last forward call
    def expert_ids(self, il: int) -> np.ndarray:
        """(k, T) routed expert ids of MoE layer il, descending biased score."""
        return self.dumps[f"ffn_moe_topk-{il}"]

    def indexer_sets(self, il: int) -> list[frozenset]:
        """Per query, the set of absolute key positions the indexer of MLA layer il selected."""
        top = self.dumps[f"top_k-{il}"]
        return [frozenset(int(i) for i in top[:, t] if i >= 0) for t in range(top.shape[1])]

    # ---- the dry run: what a forward would read, and a small slice of every tensor
    def expert_reads(self, n_tokens: int) -> int:
        """Expert slices of one routed tensor a forward of n_tokens reads, at most: each chunk of
        moe_token_chunk tokens reads every expert one of its tokens picked, min(E, k x chunk)."""
        c, ch = self.cfg, self.moe_token_chunk
        return sum(min(c.n_expert, c.n_expert_used * min(ch, n_tokens - t0)) for t0 in range(0, n_tokens, ch))

    def memory_estimate(self, n_tokens: int) -> dict[str, int]:
        """Bytes of a forward's largest buffers over n_tokens fresh tokens, as the CLI runs it (seams
        handed to a sink one at a time, none kept): the activations every layer carries ("trunk"),
        the MLA caches, the largest weight decoded whole, with the weight cache on (a file under
        4 GiB) every tensor the run reads decoded, and per layer kind its working buffers (float64
        temporaries counted as such); "peak" = trunk + caches + weight + weight cache + the largest
        kind. An upper estimate from the code's own buffers, not a measurement (the interpreter and
        the allocator come on top); the library's default (every seam kept in dumps) adds the seams."""
        c, T = self.cfg, n_tokens
        it, D, H, di = np.dtype(self.dt).itemsize, c.n_embd, c.n_head, c.d_inner
        run = range(self.n_run)
        mla = [il for il in run if not c.is_recr(il)]
        n_blocks = len(range(0, self.n_run, c.attn_res_block_size))
        Hi, Di = c.idx_n_head, c.idx_key_dim
        est = {"trunk": (n_blocks + 3) * D * T * it,            # blocks, prefix, the layer input, its output
               "mla caches": len(mla) * (c.kv_lora_rank + c.n_rot + Di) * T * it}
        whole = [n for n in self.expected_tensors() if n != "output.weight" and not n.endswith("_exps.weight")
                 and n != "token_embd.weight"]
        est["weight"] = max(self.g.info(n).elements for n in whole) * (it + 12) if whole else 0  # raw + F32 decode + copy
        est["weight cache"] = (sum(self.g.info(n).elements for n in self.expected_tensors()) * it
                               if self._cache is not None else 0)
        kinds = {"attn-res mix": D * (n_blocks + 1) * T * 16 + D * T * (8 + it)}
        if any(c.is_recr(il) for il in run):                    # 3 conv outputs + short_conv's float64 buffers
            kinds["KDA"] = di * T * (4 * it + 24)
        if mla:                                                 # q, q_abs, mask, kqv, indexer (block-bounded)
            tb = min(T, max(2, self.idx_block_bytes // (T * Hi * 8)) + 1)   # the indexer's largest query block
            idx_block = 3 * tb * T * Hi * 8 + T * Hi * 8
            kinds["MLA + indexer"] = (H * (c.n_head_k_mla + c.kv_lora_rank + c.n_head_v_mla) * T * it
                                      + T * T * (8 + 2 * it) + Di * Hi * T * (8 + it) + idx_block)
        situ_tmp = 4 * 8 * min(SITU_BLOCK, c.n_ff * T)         # situ()'s float64 temporaries, a block of rows
        if self.n_run > 0 and c.n_layer_dense_lead > 0:
            kinds["dense FFN"] = c.n_ff * T * 3 * it + situ_tmp
        if self.n_run > c.n_layer_dense_lead:
            F, S, L, K = c.n_ff_exp, c.n_expert_shared, c.n_expert_latent, c.n_expert_used
            kinds["MoE"] = (3 * c.n_expert + 3 * L) * T * it + K * L * min(T, self.moe_token_chunk) * it \
                + F * S * T * 3 * it + situ_tmp
        if self.head:
            R = self.head_chunk_rows or 8192
            kinds["head"] = c.n_vocab * T * it + R * D * (it + 8) + R * T * it
        est.update(kinds)
        est["peak"] = est["trunk"] + est["mla caches"] + est["weight"] + est["weight cache"] + max(kinds.values())
        return est


    def read_map(self, n_tokens: int) -> list[dict]:
        """Per tensor a forward of n_tokens fresh tokens reads, reading nothing: name, layer (-1:
        none), type, kind ("rows": the embedding, "whole", "experts", "head": whole, in slabs),
        elements, bytes in the file, and the bytes that forward reads (rows: one per distinct
        token at most; experts: at most min(E, k*T) of the E expert slices)."""
        c = self.cfg
        out = []
        for name in self.expected_tensors():
            t = self.g.info(name)
            nb = self._nbytes(name, t.elements)
            if name == "token_embd.weight":
                kind, reads = "rows", self.g.row_bytes(t) * min(n_tokens, c.n_vocab)
            elif name == "output.weight":
                kind, reads = "head", nb
            elif name.endswith("_exps.weight"):
                kind, reads = "experts", nb // c.n_expert * self.expert_reads(n_tokens)
            else:
                kind, reads = "whole", nb
            out.append({"name": name, "layer": int(name.split(".")[1]) if name.startswith("blk.") else -1,
                        "type": t.type_name, "kind": kind, "elements": t.elements, "bytes": nb, "reads": reads})
        return out

    def probe(self, entry: dict, tokens) -> np.ndarray:
        """A small slice of one read_map() tensor through the forward's own accessors: the tokens'
        embedding rows, one expert of a routed tensor, the head's first row, a tensor of at most
        1 MiB whole, otherwise its first row (2-D) or its first outer slice (3-D)."""
        name, kind = entry["name"], entry["kind"]
        if kind == "rows":
            return self.embed(tokens)
        if kind == "experts":
            return self.expert_w(entry["layer"], name.split(".")[2], 0)
        if kind == "whole" and entry["bytes"] <= (1 << 20):
            return self.t(name)
        if len(self.g.info(name).shape) == 2:
            w = self.g.row_range(name, 0, 1)
            self._count(name, "rows", 1, w.size)
            return w
        w = self.g.expert(name, 0)
        self._count(name, "slice", 1, w.size)
        return w


# ================================ CLI ================================

def write_dumps(dumps: dict, d: pathlib.Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in dumps.items():
        np.save(d / f"{name}.npy", arr)


def write_raw_dumps(dumps: dict, d: pathlib.Path) -> None:
    """ref/glm.py's raw layout: axes fully reversed (ggml order, features fastest), int64 ->
    int32, plus manifest.txt lines `name d0 d1 ...` in ggml order."""
    w = DumpWriter(raw_dir=d)
    for name, arr in dumps.items():
        w(name, arr)
    w.close()


class DumpWriter:
    """A KimiK31.sink that writes each seam as it is made (.npy to npy_dir, write_raw_dumps' layout
    to raw_dir), so a long run never holds its seams; close() writes the raw manifest."""

    def __init__(self, npy_dir=None, raw_dir=None):
        self.dirs = [pathlib.Path(x) if x else None for x in (npy_dir, raw_dir)]
        for x in self.dirs:
            if x:
                x.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        self.n = 0

    def __call__(self, name: str, arr: np.ndarray) -> None:
        npy, raw = self.dirs
        if npy:
            np.save(npy / f"{name}.npy", arr)
        if raw:
            rev = np.ascontiguousarray(arr.transpose(tuple(range(arr.ndim - 1, -1, -1))))
            rev = rev.astype(np.int32) if arr.dtype == np.int64 else rev.astype(F32)
            rev.tofile(raw / f"{name}.f32")
            self.lines.append(f"{name} {' '.join(str(x) for x in reversed(arr.shape))}")
        self.n += 1

    def close(self) -> None:
        if self.dirs[1]:
            (self.dirs[1] / "manifest.txt").write_text("\n".join(self.lines) + "\n")


def check_output(m: KimiK31, logits: np.ndarray | None) -> int:
    """The CLI's verdict on a forward: the logits, or without the head the last layer's output
    (m.trunk_out, whichever seams are kept), all finite; prints a summary line."""
    if logits is None:
        out, what = m.trunk_out, f"l_out-{m.n_run - 1}"
        if out is None or not np.isfinite(out).all():
            print(f"kimi_k31.py: NON-FINITE OUTPUT ({what})")
            return 1
        print(f"{what} {out.shape}: mean {out.mean():.6f} std {out.std():.6f}")
        return 0
    if not np.isfinite(logits).all():
        print("kimi_k31.py: NON-FINITE LOGITS")
        return 1
    print(f"logits {logits.shape}: mean {logits.mean():.6f} std {logits.std():.6f} "
          f"argmax per token: {[int(i) for i in logits.argmax(axis=0)]}")
    return 0


def load_tokens(path) -> list[int]:
    """Token ids from a file: .npy (an integer array), .json (a list, or an object with a "tokens"
    list), anything else text: ids separated by commas and / or whitespace, optionally in [ ]
    (llama-tokenize --ids), '#' starting a comment."""
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        a = np.load(p, allow_pickle=False)
        if a.dtype.kind not in "iu":
            raise ValueError(f"{p}: token array of dtype {a.dtype}, not integers")
        vals = [int(x) for x in a.ravel()]
    elif p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        vals = d.get("tokens") if isinstance(d, dict) else d
        if not isinstance(vals, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in vals):
            raise ValueError(f"{p}: expected a list of integer token ids (or an object with one under \"tokens\")")
    else:
        text = "\n".join(line.split("#", 1)[0] for line in p.read_text(encoding="utf-8").splitlines())
        words = text.replace("[", " ").replace("]", " ").replace(",", " ").split()
        try:
            vals = [int(w) for w in words]
        except ValueError as e:
            raise ValueError(f"{p}: not a list of integer token ids ({e})") from None
    if not vals:
        raise ValueError(f"{p}: no token ids")
    return vals


def _gb(n: float) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.1f} MiB"


def dry_run(m: KimiK31, tokens: list[int], max_rss_gb: float) -> int:
    """Check, map and probe (KimiK31.read_map / probe); never reads a tensor over 1 MiB whole. Fails
    on a probe over 1 MiB read whole, a non-finite probe, a measured peak RSS over the limit, or a
    forward whose memory_estimate() peak is over it."""
    c, T = m.cfg, len(tokens)
    rmap = m.read_map(T)
    file_bytes = sum(p.stat().st_size for p in m.g.paths)
    layers: dict[int, Counter] = {}
    bad = []
    for e in rmap:
        before = m.bytes_read
        a = m.probe(e, tokens)
        if not np.isfinite(a).all():
            bad.append(e["name"])
        del a
        s = layers.setdefault(e["layer"], Counter())
        s["n"] += 1
        s["bytes"] += e["bytes"]
        s[e["kind"]] += e["reads"]
        s["probed"] += m.bytes_read - before
    whole = [n for (n, what) in m.reads if what is None and m._nbytes(n, m.g.info(n).elements) > (1 << 20)]
    itemsize = np.dtype(m.dt).itemsize
    big = max((e for e in rmap if e["kind"] == "whole"), key=lambda e: e["elements"])
    total, reads = sum(e["bytes"] for e in rmap), sum(e["reads"] for e in rmap)
    print(f"read map: layers 0..{m.n_run - 1} of {c.n_layers}{' + head' if m.head else ''}, {T} tokens: "
          f"{len(rmap)} tensors, {_gb(total)} of the {_gb(file_bytes)} file; a forward reads {_gb(reads)}")
    for il in sorted(layers):
        s = layers[il]
        what = ("embedding + head" if il < 0 else
                f"{'KDA' if c.is_recr(il) else 'MLA'} {'dense' if il < c.n_layer_dense_lead else 'MoE'}")
        n_ch = -(-T // m.moe_token_chunk)
        exp = (f", experts {_gb(s['experts'])} (<= {m.expert_reads(T)} reads of {c.n_expert} slices x 3"
               f"{f' over {n_ch} chunks of {m.moe_token_chunk} tokens' if n_ch > 1 else ''})"
               if s["experts"] else "")
        print(f"  layer {il:3d} {what:16s} {s['n']:3d} tensors {_gb(s['bytes']):>10s}; reads whole "
              f"{_gb(s['whole'] + s['head']):>10s}{exp}{', rows ' + _gb(s['rows']) if s['rows'] else ''}; "
              f"probed {_gb(s['probed'])}")
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    except ImportError:                                 # Windows: not measured
        rss = None
    print(f"largest tensor read whole: {big['name']} ({big['type']}), {_gb(big['elements'] * itemsize)} as "
          f"{np.dtype(m.dt).name}")
    print(f"dry run: probes read {_gb(m.bytes_read)} in {sum(m.reads.values())} reads; tensors over 1 MiB read "
          f"whole: {len(whole)}; non-finite probes: {len(bad)}; peak RSS "
          f"{'not measured here' if rss is None else _gb(rss)} (limit {max_rss_gb:g} GiB)")
    for n in whole + bad:
        print(f"  {'read whole' if n in whole else 'non-finite'}: {n}")
    est = m.memory_estimate(T)
    print(f"forward memory estimate ({T} tokens, {np.dtype(m.dt).name}, seams streamed): peak "
          f"{_gb(est['peak'])} (limit {max_rss_gb:g} GiB) = trunk {_gb(est['trunk'])} + MLA caches "
          f"{_gb(est['mla caches'])} + weight {_gb(est['weight'])} + weight cache {_gb(est['weight cache'])} + "
          "the largest of " + ", ".join(f"{k} {_gb(v)}" for k, v in est.items()
                                        if k not in ("trunk", "mla caches", "weight", "weight cache", "peak")))
    ok = (not whole and not bad and (rss is None or rss <= max_rss_gb * 2**30)
          and est["peak"] <= max_rss_gb * 2**30)
    print("dry run: OK" if ok else "dry run: FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="kimi-k31 reference forward pass")
    ap.add_argument("model", help="a kimi-k31 GGUF (real or tools/kimi_k31_synth_gguf.py's tiny.gguf)")
    tk = ap.add_mutually_exclusive_group()
    tk.add_argument("--tokens", default=None, help="comma-separated token ids (default 1,2,...,8)")
    tk.add_argument("--tokens-file", default=None,
                    help="token ids from a file: .npy, .json (a list or {\"tokens\": [...]}), or text")
    ap.add_argument("--layers", type=int, default=None,
                    help="run layers 0..N-1 only (the first N layers, then the head as an N-layer file has it)")
    ap.add_argument("--no-head", action="store_true", help="stop after the last layer (no logits)")
    ap.add_argument("--eps", default="moonshot", choices=sorted(EPS_PRESETS))
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES), help="seam dtype")
    ap.add_argument("--act-q8", type=int, default=0, choices=(0, ACT_Q8_BLOCK),
                    help="32: round the input of every product whose weight is an iq::mv type (IQ_MV_TYPES) "
                         "to int8 per 32 values, as the engine's device path does")
    ap.add_argument("--seams", default=None,
                    help="comma-separated globs: keep (and dump) only these seams, e.g. 'l_out-12,res_blocks-12'")
    ap.add_argument("--dump-dir", default=None, help="write every kept cb() seam here as .npy, as it is made")
    ap.add_argument("--dump-raw-dir", default=None, help="write every kept cb() seam as raw ggml-order data")
    ap.add_argument("--greedy", type=int, default=0, help="then decode N tokens greedily (with the state)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check and map what the run would read, probe a small slice of each tensor; no forward")
    ap.add_argument("--max-rss-gb", type=float, default=20.0, help="--dry-run fails above this peak RSS")
    args = ap.parse_args()
    if args.no_head and args.greedy:
        ap.error("--greedy needs the head (drop --no-head)")

    m = KimiK31(args.model, eps=args.eps, dtype=args.dtype, layers=args.layers, head=not args.no_head,
                act_q8=args.act_q8, seams=None if args.seams is None else args.seams.split(","))
    c = m.cfg
    print(f"kimi-k31: n_embd {c.n_embd}, layers {c.n_layers} ({sum(c.is_recr(i) for i in range(c.n_layers))} KDA"
          f" / {len(c.mla_layers())} MLA+indexer), attn-res block {c.attn_res_block_size}, "
          f"experts {c.n_expert} x top{c.n_expert_used} latent {c.n_expert_latent}, "
          f"kda {c.n_head}x{c.kda_head_dim}, mla {c.qk_nope}+{c.n_rot}/kv {c.kv_lora_rank}, "
          f"indexer {c.idx_n_head}x{c.idx_key_dim} top {c.idx_top_k}, eps {c.eps()}")
    if m.n_run != c.n_layers or not m.head or m.act_q8:
        n_q8 = sum(m.q8_rounds(n) for n in m.expected_tensors())
        print(f"run: layers 0..{m.n_run - 1}{' + head' if m.head else ', no head'}"
              f"{f', int8 inputs per {m.act_q8} on {n_q8} iq::mv-type tensors' if m.act_q8 else ''}")
    if args.tokens_file:
        tokens = load_tokens(args.tokens_file)
    else:
        tokens = [int(x) for x in (args.tokens or "1,2,3,4,5,6,7,8").split(",")]
    if min(tokens) < 0 or max(tokens) >= c.n_vocab:
        raise SystemExit(f"kimi_k31.py: token id out of range 0..{c.n_vocab - 1}")
    if args.dry_run:
        return dry_run(m, tokens, args.max_rss_gb)
    st = m.new_state()
    writer = DumpWriter(args.dump_dir, args.dump_raw_dir)
    m.sink = writer                                     # seams go to disk as they are made (or nowhere)
    logits = m.forward(tokens, st)
    writer.close()
    m.sink = lambda name, arr: None                     # decode steps keep and write no seams
    rc = check_output(m, logits)
    for d, what in ((args.dump_dir, "dumps"), (args.dump_raw_dir, "raw dumps")):
        if d:
            print(f"{what}: {writer.n} tensors -> {d}")
    if rc:
        return rc
    if args.greedy:
        out, nxt = [], int(logits[:, -1].argmax())
        for _ in range(args.greedy):
            out.append(nxt)
            nxt = int(m.forward([nxt], st)[:, -1].argmax())
        print("greedy continuation:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
