"""ref/glm_dsa.py - the glm-dsa (GLM-5.3 744B-A40B) reference forward pass, numpy float64.

glm-dsa BUILD-PLAN B3.1: the specification every glm-dsa engine kernel is parity-tested against.
Two readings of the same GGUF are implemented:

  --mode hf     (default) the HF/vLLM semantics (transformers modular_glm_moe_dsa.py /
                modeling_deepseek_v32.py, vLLM deepseek_v32), which glm-dsa's engine implements:
                indexer key LayerNorm eps 1e-6; no Hadamard; MoE weights s / (sum s + 1e-20) x 2.5;
                the MTP block runs its own full indexer (layer 78 is a "full" layer).
  --mode llama  llama.cpp aa94f208 src/models/glm-dsa.cpp, for the second-oracle cross-check:
                indexer LayerNorm eps 0 (glm-dsa never loads attention.layer_norm_epsilon, so
                f_norm_eps stays 0); the 128x128 normalized Hadamard on indexer q and k; MoE weight
                sum clamped below at 6.103515625e-5 (llama-graph.cpp build_moe_ffn); dense MLA in
                the MTP block (graph_mtp has no indexer); ggml's cumulative-float RoPE angles.

Both share everything else (GLM-5.3-SPEC.md "Per-token dataflow"; glm-dsa.cpp graph::graph):

  h = RMS(x; attn_norm, eps 1e-5)               qr = RMS(W_q_a h; q_a_norm)
  q = W_q_b qr, per head [nope 192 | rope 64]   kv = W_kv_a_mqa h = [c_kv 512 | k_pe 64]
  c_kv = RMS(c_kv; kv_a_norm)                   q_pe, k_pe: NORM RoPE (adjacent pairs), theta 8e6
  q_abs_h = W_k_b,h q_nope_h  (192 -> 512)      logit_h(s) = (q_abs_h . c_kv(s) + q_pe_h . k_pe(s)) * kq_scale
  kq_scale = 1/sqrt(key_length_mla) = 1/sqrt(192 + 64) = 1/16.  Resolved from both sources:
      llama.cpp glm-dsa.cpp: kq_scale = mscale^2 / sqrt(n_embd_head_k_mla) with mscale = 1 when
      freq_scale = 1 (no YaRN; GLM-5.3 rope_type "default"); HF: scaling = qk_head_dim^-0.5 with
      qk_head_dim = qk_nope + qk_rope = 256.  NOT 1/sqrt(192) (the nope width) and not 1/sqrt(576).
  softmax over the selected cells (top-k of the indexer, or all cells when n <= top_k)
  o_h = W_v_b,h (sum_s p_s c_kv(s))  (512 -> 256)    attn = W_o concat_h(o_h)
  x += attn;  h2 = RMS(x; ffn_norm);  x += dense SwiGLU (layers < leading_dense) or
      MoE: s = sigmoid(W_r h2) (f64 here, f32 in the model), top-8 of (s + exp_probs_b) (bias for
      SELECTION only), w = s_sel normalized, x expert_weights_scale; plus the shared expert, unscaled.
      No SwiGLU clamp (GLM-5.3 has no swiglu_limit; Maya's Flash default of 10 is a misreading here).
  logits = W_out RMS(x; output_norm)

Lightning indexer, on "full" layers only (indexer.types, or llama.cpp's back-compat rule):
  iq = W_idx_q_b qr, 32 heads x 128, layout [rope 64 | nope 64] (rope FIRST, unlike the MLA query)
  ik = LayerNorm(W_idx_k h; k_norm w, b, eps), one 128-dim key, rope on dims 0..63
  w  = W_idx_proj h * 1/sqrt(n_head_idx * head_dim_idx) (= 1/sqrt(4096): HF splits it into
       n_heads^-0.5 on the weights and head_dim^-0.5 on q.k; ReLU is positively homogeneous, so the
       product is the same, and llama.cpp folds both into the weights)
  I(t, s) = sum_h w_h(t) relu(iq_h(t) . ik(s)) for s <= t;  top-min(top_k, t+1) cells by I, ties to
  the lower position.  "Shared" layers reuse the most recent full layer's selection.

RoPE angles.  HF mode follows transformers: inv_freq = base^(-2i/d) (computed in f64, rounded to f32),
angle = f32(f32(pos) * inv_freq), then cos/sin in f64 of that f32 angle (BUILD-PLAN B3.4).  llama
mode follows ggml-cpu's rope cache: theta = f32(pos), multiplied in f32 by f32(base^(-2/d)) per
pair.  Both rotate adjacent pairs (2i, 2i+1) (LLAMA_ROPE_TYPE_NORM = HF rope_interleave).

Layout conventions are Maya's ref/glm.py: activations (features, tokens); `tensor()` gives (out, in)
so `W @ x` is ggml_mul_mat; a (dim*n_head, T) projection splits into heads as (dim, n_head, T) with
element (e, h) at row e + dim*h.  Seams are recorded under llama.cpp's cb() names.

Usage:
    python ref/glm_dsa.py MODEL.gguf [--tokens 1,2,3] [--mode hf|llama] [--pos0 N]
                          [--mutate NAME[,NAME]] [--mtp [--mtp-out M.f32]] [--logits-out F.npy] [--dump-dir D]
                          [--layers A:B --hidden-in H --hidden-out H2.npy [--positions P] [--report R.json]
                           [--compare-engine E.f32 [--engine-first N]] [--compare-llama L.bin [--llama-first N]]]
                          [--greedy N]

Slice oracle (BUILD-PLAN B7.5): `--layers A:B --hidden-in H` runs layers [A, B) on a dumped hidden state instead of
the whole model, reading every weight exactly through gguf-py and each routed expert only when a token picks it, so
it runs on the real 744B GGUF for a few layers.  H is the residual entering layer A for positions pos0 .. pos0+T-1
(every earlier position is needed: attention reads them): an .npy of (n_embd, T) or (T, n_embd), or a raw float32
file of T rows of n_embd (llama.cpp's LLM_CB_DIR l_out-<A-1>.f32 from tools/llm_logits_dump, the engine slow path's
GLM_CB_DIR l_out-<A-1>.f32 with GLM_CB_APPEND=1, or a previous slice's --hidden-out).  When B is the trunk's last
layer the slice also applies output_norm and the head (read in row chunks) at --positions (default: all), writes
them with --logits-out, and with --compare-engine / --compare-llama reports KL(slice || engine) and
KL(slice || llama) over the slice's top-64 tokens plus the rest (kl_top_rest) and the B7.5 gate:
|KL_e - KL_l| <= 0.01 and KL_e <= 1.5 x KL_l (means over the positions; --kl-floor F also passes KL_e <= F).
--dump-dir writes the slice's seams (ffn_moe_topk-L: the MoE ids; top_k-L: the selection) for
tools/strata/seam_bisect.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from glm_dsa_load import GgufAny  # noqa: E402

F64 = np.float64
ARCH = "glm-dsa"
LLAMA_NORM_CLAMP = 6.103515625e-5      # llama-graph.cpp build_moe_ffn: weights_sum clamped below

# The named wrong readings (BUILD-PLAN B3.3).  Each must move the logits far outside the parity gate.
MUTATIONS = {
    "neox_rope": "RoPE rotates (i, i + d/2) half-split pairs (NEOX) instead of adjacent pairs (NORM)",
    "idx_nope_rope": "indexer head read as [nope | rope]: RoPE on the LAST n_rot dims instead of the first",
    "scale_nope": "attention scale 1/sqrt(qk_nope) (1/sqrt(192)) instead of 1/sqrt(qk_nope + qk_rope)",
    "swiglu_clamp10": "Maya's Flash default: SwiGLU gate clamped above at 10 and up clamped to [-10, 10]",
    "shared_as_full": "shared layers run their own indexer (with extra per-layer weights) instead of reusing",
    "idx_eps_1e5": "indexer key LayerNorm eps 1e-5 (the RMS eps) instead of 1e-6",
    "router_bias_in_weights": "the selection bias exp_probs_b also enters the combine weights",
}

COVER: Counter = Counter()


# ================================ config ================================

class GlmDsaConfig:
    """Geometry and constants from the glm-dsa.* GGUF keys (llama.cpp names, llama-arch.cpp)."""

    def __init__(self, meta: dict):
        arch = meta.get("general.architecture")
        if arch != ARCH:
            raise ValueError(f"general.architecture is {arch!r}, this oracle reads {ARCH!r}")

        def u(key, default=None):
            v = meta.get(f"{ARCH}.{key}")
            return default if v is None else v

        def scalar(key, default=None, n=None):
            v = u(key, default)
            if isinstance(v, list):
                if n is not None and len(v) != n:
                    raise ValueError(f"{key}: array of {len(v)} entries, expected {n}")
                if len(set(v)) != 1:
                    raise ValueError(f"{key}: per-layer values differ ({sorted(set(v))}); not supported")
                return v[0]
            return v

        self.n_layer_all = int(u("block_count"))
        self.n_nextn = int(u("nextn_predict_layers", 0))
        self.n_layer = self.n_layer_all - self.n_nextn
        self.n_embd = int(u("embedding_length"))
        self.n_ctx_train = int(u("context_length", 0))
        self.n_head = int(scalar("attention.head_count", None, self.n_layer_all))
        n_head_kv = int(scalar("attention.head_count_kv", 1, self.n_layer_all))
        if n_head_kv != 1:
            raise ValueError(f"head_count_kv {n_head_kv}: absorbed MLA is MQA (1)")
        self.q_lora = int(u("attention.q_lora_rank"))
        self.kv_lora = int(u("attention.kv_lora_rank"))
        self.n_rot = int(u("rope.dimension_count"))
        self.k_mla = int(u("attention.key_length_mla"))
        self.v_mla = int(u("attention.value_length_mla"))
        self.qk_nope = self.k_mla - self.n_rot
        key_length = int(u("attention.key_length", self.kv_lora + self.n_rot))
        value_length = int(u("attention.value_length", self.kv_lora))
        if key_length != self.kv_lora + self.n_rot or value_length != self.kv_lora:
            raise ValueError(f"key/value_length {key_length}/{value_length} != kv_lora + rope / kv_lora")
        self.rope_base = float(u("rope.freq_base", 10000.0))
        stype = u("rope.scaling.type")
        if (stype not in (None, "none")) or u("rope.scaling.factor") not in (None, 1.0) \
                or u("rope.scaling.yarn_log_multiplier") not in (None, 0.0):
            raise NotImplementedError("rope scaling (YaRN mscale) is not modelled; GLM-5.3 has none")
        self.rms_eps = float(u("attention.layer_norm_rms_epsilon", 1e-5))
        self.n_vocab = int(u("vocab_size", len(meta.get("tokenizer.ggml.tokens", []))))

        self.n_expert = int(u("expert_count", 0))
        self.n_used = int(scalar("expert_used_count", 0, self.n_layer_all))
        self.n_ff_exp = int(scalar("expert_feed_forward_length", 0, self.n_layer_all))
        self.n_ff = int(scalar("feed_forward_length", 0, self.n_layer_all))
        self.n_shared = int(u("expert_shared_count", 0))
        self.dense_lead = int(u("leading_dense_block_count", 0))
        self.w_scale = float(u("expert_weights_scale", 1.0))
        self.w_norm = bool(u("expert_weights_norm", False))
        gating = int(u("expert_gating_func", 2))
        if gating not in (0, 2):
            raise NotImplementedError(f"expert_gating_func {gating}: glm-dsa uses sigmoid (2)")

        self.idx_heads = int(u("attention.indexer.head_count"))
        self.idx_dim = int(u("attention.indexer.key_length"))
        self.top_k = int(u("attention.indexer.top_k"))
        if self.n_rot > self.idx_dim:
            raise ValueError("rope dims exceed the indexer head")
        types = u("attention.indexer.types")
        if types is not None:
            if len(types) != self.n_layer:
                raise ValueError(f"indexer.types has {len(types)} entries, expected {self.n_layer}")
            self.idx_full = [bool(t) for t in types]
            self.idx_types_source = "gguf"
        elif self.n_ctx_train < 1048576:
            self.idx_full = [True] * self.n_layer          # GLM-5/5.1 back-compat: every layer full
            self.idx_types_source = "bc-all-full"
        else:
            self.idx_full = [max(il - 3 + 1, 0) % 4 == 0 for il in range(self.n_layer)]
            self.idx_types_source = "bc-glm52"
        if self.n_layer and not self.idx_full[0]:
            raise ValueError("layer 0 must be a full indexer layer")

    def is_full(self, il: int) -> bool:
        return True if il >= self.n_layer else self.idx_full[il]

    def is_moe(self, il: int) -> bool:
        return il >= self.dense_lead

    def summary(self) -> str:
        nf = sum(self.idx_full)
        return (f"glm-dsa: {self.n_layer} layers (+{self.n_nextn} nextn), n_embd {self.n_embd}, "
                f"heads {self.n_head} x [nope {self.qk_nope} | rope {self.n_rot}] v {self.v_mla}, "
                f"q_lora {self.q_lora} kv_lora {self.kv_lora}, rope base {self.rope_base:g}, "
                f"indexer {self.idx_heads}x{self.idx_dim} top-{self.top_k} ({nf} full, {self.idx_types_source}), "
                f"experts {self.n_expert} x top{self.n_used} ff {self.n_ff_exp}, dense lead {self.dense_lead}, "
                f"vocab {self.n_vocab}")


@dataclasses.dataclass
class Options:
    mode: str = "hf"                       # "hf" or "llama"
    idx_ln_eps: float | None = None        # None: 1e-6 (hf), 0.0 (llama)
    hadamard: bool | None = None           # None: off (hf), on (llama)
    llama_norm_clamp: bool | None = None   # None: off (hf: +1e-20), on (llama: clamp 6.1e-5)
    mtp_dense: bool | None = None          # None: off (hf: MTP runs its indexer), on (llama)
    rope_angles: str | None = None         # None: "hf" (hf) or "ggml" (llama); "f64" exact
    top_k: int | None = None               # None: the GGUF's indexer.top_k (a test hook: 10**9 = dense)
    mutations: frozenset = frozenset()

    def __post_init__(self):
        if self.mode not in ("hf", "llama"):
            raise ValueError(f"mode {self.mode!r}")
        llama = self.mode == "llama"
        if self.idx_ln_eps is None:
            self.idx_ln_eps = 0.0 if llama else 1e-6
        if self.hadamard is None:
            self.hadamard = llama
        if self.llama_norm_clamp is None:
            self.llama_norm_clamp = llama
        if self.mtp_dense is None:
            self.mtp_dense = llama
        if self.rope_angles is None:
            self.rope_angles = "ggml" if llama else "hf"
        self.mutations = frozenset(self.mutations)
        bad = self.mutations - set(MUTATIONS)
        if bad:
            raise ValueError(f"unknown mutations {sorted(bad)}; known: {sorted(MUTATIONS)}")
        if "idx_eps_1e5" in self.mutations:
            self.idx_ln_eps = 1e-5


# ================================ primitives (float64) ================================

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x * x, axis=0, keepdims=True)
    return x / np.sqrt(ms + eps) * w.reshape((-1,) + (1,) * (x.ndim - 1))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    d = x - mu
    var = (d * d).mean(axis=0, keepdims=True)
    return d / np.sqrt(var + eps) * w[:, None] + b[:, None]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def split_heads(x: np.ndarray, dim: int, n_head: int) -> np.ndarray:
    """(dim*n_head, T) -> (dim, n_head, T); element (e, h) at row e + dim*h (ggml reshape order)."""
    return np.ascontiguousarray(x.reshape(n_head, dim, -1).transpose(1, 0, 2))


def join_heads(x: np.ndarray) -> np.ndarray:
    dim, n_head, T = x.shape
    return np.ascontiguousarray(x.transpose(1, 0, 2).reshape(n_head * dim, T))


def hadamard(n: int) -> np.ndarray:
    """Normalized Sylvester-Walsh Hadamard matrix (symmetric, orthonormal), as llama.cpp builds it."""
    if n & (n - 1):
        raise ValueError(f"Hadamard size {n} is not a power of two")
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def rope_angles(pos: np.ndarray, n_rot: int, base: float, how: str) -> np.ndarray:
    """(n_rot/2, T) float64 angles.  `how`: "hf" (transformers, f32 inv_freq and product), "ggml"
    (ggml-cpu rope cache) or "f64" (exact float64 angles, for float64 third-party references)."""
    half = n_rot // 2
    if how == "f64":
        inv = np.array([base ** (-2.0 * i / n_rot) for i in range(half)], dtype=F64)
        return inv[:, None] * np.asarray(pos, dtype=F64)[None, :]
    pos32 = np.asarray(pos, dtype=np.float32)
    if how == "hf":
        inv = np.array([base ** (-2.0 * i / n_rot) for i in range(half)], dtype=np.float32)
        return (pos32[None, :] * inv[:, None]).astype(F64)        # f32 product, then exact widening
    if how == "ggml":
        scale = np.float32(base ** (-2.0 / n_rot))
        out = np.empty((half, pos32.size), dtype=F64)
        theta = pos32.copy()
        for i in range(half):
            out[i] = theta
            theta = (theta * scale).astype(np.float32)
        return out
    raise ValueError(how)


def apply_rope(x: np.ndarray, ang: np.ndarray, neox: bool) -> np.ndarray:
    """Rotate the leading axis of x (n_rot, ..., T) by angles (n_rot/2, T).  NORM pairs (2i, 2i+1);
    NEOX pairs (i, i + n_rot/2)."""
    n = x.shape[0]
    shape = (n // 2,) + (1,) * (x.ndim - 2) + (ang.shape[1],)
    c = np.cos(ang).reshape(shape)
    s = np.sin(ang).reshape(shape)
    out = np.empty_like(x)
    if neox:
        a, b = x[: n // 2], x[n // 2:]
        out[: n // 2] = a * c - b * s
        out[n // 2:] = a * s + b * c
    else:
        a, b = x[0::2], x[1::2]
        out[0::2] = a * c - b * s
        out[1::2] = a * s + b * c
    return out


def top_k_mask(score: np.ndarray, k: int) -> np.ndarray:
    """score (T, S) with -inf on invisible cells -> bool mask of the top-min(k, visible) cells per
    row; ties go to the lower position (stable sort on -score).  Rows with at most k visible cells
    keep all of them without sorting."""
    vis = np.isfinite(score)
    mask = vis.copy()
    rows = np.nonzero(vis.sum(axis=1) > k)[0]
    if rows.size:
        order = np.argsort(-score[rows], axis=1, kind="stable")[:, :k]
        sub = np.zeros((rows.size, score.shape[1]), dtype=bool)
        sub[np.repeat(np.arange(rows.size), k), order.reshape(-1)] = True
        mask[rows] = sub & vis[rows]
    return mask


def kl_rows(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """KL(softmax(p) || softmax(q)) per column of (vocab, T) logits."""
    lp = p_logits - p_logits.max(axis=0, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(axis=0, keepdims=True))
    lq = q_logits - q_logits.max(axis=0, keepdims=True)
    lq = lq - np.log(np.exp(lq).sum(axis=0, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(axis=0)


def kl_top_rest(ref_logits: np.ndarray, test_logits: np.ndarray, k: int = 64) -> np.ndarray:
    """The KL every glm-dsa gate uses (BUILD-PLAN section 6): per position, KL(ref || test) over the
    reference's top-k tokens plus one "rest" bucket holding the remaining probability mass."""
    def logp(z):
        z = z - z.max(axis=0, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=0, keepdims=True))
    lp, lq = logp(ref_logits), logp(test_logits)
    out = np.empty(ref_logits.shape[1])
    for t in range(ref_logits.shape[1]):
        top = np.argsort(-lp[:, t], kind="stable")[:k]
        p, q = np.exp(lp[top, t]), np.exp(lq[top, t])
        pr, qr = max(1.0 - p.sum(), 0.0), max(1.0 - q.sum(), 0.0)
        kl = float((p * (lp[top, t] - lq[top, t])).sum())
        if pr > 0:
            kl += pr * (math.log(pr) - math.log(max(qr, 1e-300)))
        out[t] = kl
    return out


def load_hidden(path, n_embd: int) -> np.ndarray:
    """A hidden-state dump as (n_embd, T) float64: .npy of (n_embd, T) or (T, n_embd), or raw float32 rows of n_embd
    (ggml order: each token's n_embd values contiguous, as llama.cpp's and the engine's seam dumps are)."""
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p)
        if a.ndim == 1:
            a = a.reshape(-1, n_embd)
        if a.ndim != 2:
            raise ValueError(f"{p}: {a.ndim}-D array, expected (n_embd, T) or (T, n_embd)")
        if a.shape[0] == n_embd:
            return a.astype(F64)
        if a.shape[1] == n_embd:
            return a.T.astype(F64)
        raise ValueError(f"{p}: shape {a.shape} has no n_embd ({n_embd}) axis")
    raw = np.fromfile(p, dtype="<f4")
    if raw.size == 0 or raw.size % n_embd:
        raise ValueError(f"{p}: {raw.size} floats is not a whole number of {n_embd}-float rows")
    return raw.reshape(-1, n_embd).T.astype(F64)


def parse_positions(spec: str | None, T: int, pos0: int = 0) -> np.ndarray:
    """Columns (0 .. T-1) for absolute positions: "a:b" (b exclusive), "a,b,c", "-16" (the last 16), or None (all).
    Positions are absolute (pos0 + column)."""
    if not spec:
        return np.arange(T, dtype=np.int64)
    cols: list[int] = []
    if spec.startswith("-") and spec[1:].isdigit():
        n = int(spec[1:])
        return np.arange(max(T - n, 0), T, dtype=np.int64)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            cols.extend(range(int(a) - pos0, int(b) - pos0))
        else:
            cols.append(int(part) - pos0)
    out = np.asarray(sorted(set(cols)), dtype=np.int64)
    if out.size and (out.min() < 0 or out.max() >= T):
        raise ValueError(f"positions {spec!r} fall outside {pos0} .. {pos0 + T - 1}")
    return out


def load_logit_rows(path, n_vocab: int, first: int, cols_abs: np.ndarray) -> np.ndarray:
    """Rows of a (rows, V) float32 logits / log-softmax file whose row 0 is absolute position `first`, picked at the
    absolute positions cols_abs, as (n_vocab, len) float64.  V >= n_vocab (an engine vocabulary padded past the
    GGUF's is cut; padded entries carry no probability)."""
    raw = np.fromfile(str(path), dtype="<f4")
    V = n_vocab
    for cand in (n_vocab, 154880):
        if raw.size % cand == 0:
            V = cand
            break
    if raw.size % V:
        raise ValueError(f"{path}: {raw.size} floats is not a whole number of {n_vocab}-float rows")
    rows = raw.reshape(-1, V)
    idx = np.asarray(cols_abs, dtype=np.int64) - first
    if idx.size and (idx.min() < 0 or idx.max() >= rows.shape[0]):
        raise ValueError(f"{path}: rows cover positions {first} .. {first + rows.shape[0] - 1}, "
                         f"not {int(cols_abs.min())} .. {int(cols_abs.max())}")
    return rows[idx, :n_vocab].T.astype(F64)


def slice_gate(kl_e: float, kl_l: float, floor: float = 0.0) -> tuple[bool, str]:
    """B7.5: |KL(slice || engine) - KL(slice || llama)| <= 0.01 and KL(slice || engine) <= 1.5 x KL(slice || llama);
    KL_e <= floor (when > 0) passes the second clause too."""
    a = abs(kl_e - kl_l) <= 0.01
    b = kl_e <= 1.5 * kl_l or (floor > 0 and kl_e <= floor)
    why = (f"|KL_e - KL_l| = {abs(kl_e - kl_l):.4g} {'<=' if a else '>'} 0.01; "
           f"KL_e {kl_e:.4g} {'<=' if b else '>'} 1.5 x KL_l {1.5 * kl_l:.4g}" + (f" (floor {floor:g})" if floor > 0 else ""))
    return a and b, why


# ================================ the model ================================

class GlmDsa:
    def __init__(self, path, opts: Options | None = None, cache: bool | None = None):
        self.g = GgufAny(path)
        self.cfg = GlmDsaConfig(self.g.metadata)
        self.opts = opts or Options()
        if cache is None:
            cache = sum(p.stat().st_size for p in self.g.paths) < 512 * 2 ** 20
        self.cache_on = cache
        self._cache: dict = {}
        self.dumps: dict[str, np.ndarray] = {}
        self.collect = False
        self._had = None

    def close(self) -> None:
        self.g.close()

    def with_options(self, opts: Options) -> "GlmDsa":
        """A view sharing this model's file and weight cache, with other options."""
        m = object.__new__(GlmDsa)
        m.__dict__.update(self.__dict__)
        m.opts, m.dumps, m._had = opts, {}, None
        return m

    # ---- weights
    def W(self, name: str) -> np.ndarray:
        w = self._cache.get(name)
        if w is None:
            w = self.g.tensor(name).astype(F64)
            if self.cache_on:
                self._cache[name] = w
        return w

    def L(self, il: int, base: str, suffix: str = "weight") -> np.ndarray:
        return self.W(f"blk.{il}.{base}.{suffix}")

    def has(self, il: int, base: str, suffix: str = "weight") -> bool:
        return self.g.has(f"blk.{il}.{base}.{suffix}")

    def expert(self, il: int, base: str, e: int) -> np.ndarray:
        key = (il, base, e)
        w = self._cache.get(key)
        if w is None:
            w = self.g.expert(f"blk.{il}.{base}.weight", e).astype(F64)
            if self.cache_on:
                self._cache[key] = w
        return w

    def embed(self, tokens) -> np.ndarray:
        t = np.asarray(tokens, dtype=np.int64)
        if t.size and (t.min() < 0 or t.max() >= self.cfg.n_vocab):
            raise ValueError(f"token id out of range 0..{self.cfg.n_vocab - 1}")
        if self.cache_on:
            return self.W("token_embd.weight")[t].T.copy()
        return self.g.rows("token_embd.weight", t).astype(F64).T.copy()

    def cb(self, name: str, x: np.ndarray, il: int = -1) -> None:
        if self.collect:
            self.dumps[f"{name}-{il}" if il >= 0 else name] = np.array(x, copy=True)

    # ---- rope helpers
    def _rope(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        ang = rope_angles(pos, self.cfg.n_rot, self.cfg.rope_base, self.opts.rope_angles)
        return apply_rope(x, ang, neox="neox_rope" in self.opts.mutations)

    # ================================ indexer ================================

    def indexer(self, h: np.ndarray, qr: np.ndarray, il: int, pos: np.ndarray) -> np.ndarray:
        """Selection mask (T, T) of this full layer."""
        c, o = self.cfg, self.opts
        D, NH, R = c.idx_dim, c.idx_heads, c.n_rot
        iq = split_heads(self.L(il, "indexer.attn_q_b") @ qr, D, NH)               # (D, NH, T)
        ik = layer_norm(self.L(il, "indexer.attn_k") @ h, self.L(il, "indexer.k_norm"),
                        self.L(il, "indexer.k_norm", "bias"), o.idx_ln_eps)          # (D, T)
        rs = slice(D - R, D) if "idx_nope_rope" in o.mutations else slice(0, R)
        iq[rs] = self._rope(iq[rs], pos)
        ik[rs] = self._rope(ik[rs], pos)
        self.cb("indexer_q", iq, il)
        self.cb("indexer_k", ik, il)
        if o.hadamard:
            if self._had is None or self._had.shape[0] != D:
                self._had = hadamard(D)
            iq = np.einsum("ed,dht->eht", self._had, iq)
            ik = self._had @ ik
        w = (self.L(il, "indexer.proj") @ h) * (1.0 / math.sqrt(D * NH))           # (NH, T)
        self.cb("indexer_weights", w, il)
        T = h.shape[1]
        score = np.zeros((T, T))
        for hh in range(NH):                                                         # sum_h w_h relu(q_h . k)
            score += w[hh][:, None] * np.maximum(iq[:, hh, :].T @ ik, 0.0)
        score[np.triu_indices(T, 1)] = -np.inf
        self.cb("indexer_score", score, il)
        k = c.top_k if o.top_k is None else o.top_k
        mask = top_k_mask(score, k)
        # coverage: rows where selection is active, and boundary near-ties
        active = np.arange(1, T + 1) > k
        COVER["idx_rows_selecting"] += int(active.sum())
        if active.any():
            srt = -np.sort(-score[active], axis=1)
            gap = srt[:, k - 1] - srt[:, k]
            COVER["idx_near_ties"] += int((np.abs(gap) < 1e-6 * (np.abs(srt[:, 0]) + 1e-30)).sum())
        self.cb("top_k", mask, il)
        return mask

    # ================================ attention ================================

    def attention(self, h: np.ndarray, qr: np.ndarray, il: int, pos: np.ndarray, sel: np.ndarray) -> np.ndarray:
        c, o = self.cfg, self.opts
        H = c.n_head
        q = split_heads(self.L(il, "attn_q_b") @ qr, c.k_mla, H)                       # (k_mla, H, T)
        q_nope = q[: c.qk_nope]
        q_pe = self._rope(q[c.qk_nope:], pos)
        kv = self.L(il, "attn_kv_a_mqa") @ h                                          # (kv_lora+rot, T)
        k_pe = self._rope(kv[c.kv_lora:], pos)
        c_kv = rms_norm(kv[: c.kv_lora], self.L(il, "attn_kv_a_norm"), c.rms_eps)
        self.cb("q_pe", q_pe, il)
        self.cb("k_pe", k_pe, il)
        self.cb("kv_cmpr", c_kv, il)
        wk_b = self.L(il, "attn_k_b")                                                 # (H, kv_lora, nope)
        q_abs = np.einsum("hjn,nht->jht", wk_b, q_nope)                               # (kv_lora, H, T)
        self.cb("q_nope_absorbed", q_abs, il)
        scale = 1.0 / math.sqrt(c.qk_nope if "scale_nope" in o.mutations else c.k_mla)
        ctx = np.empty_like(q_abs)                                                    # (kv_lora, H, T)
        for hh in range(H):
            s = (q_abs[:, hh, :].T @ c_kv + q_pe[:, hh, :].T @ k_pe) * scale          # (T, S)
            s[~sel] = -np.inf
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            ctx[:, hh, :] = (p @ c_kv.T).T
        wv_b = self.L(il, "attn_v_b")                                                 # (H, v, kv_lora)
        oh = np.einsum("hvj,jht->vht", wv_b, ctx)                                     # (v, H, T)
        out = join_heads(oh)
        self.cb("kqv_out", out, il)
        attn = self.L(il, "attn_output") @ out
        self.cb("attn_out", attn, il)
        return attn

    # ================================ FFNs ================================

    def swiglu(self, g: np.ndarray, u: np.ndarray) -> np.ndarray:
        COVER["swiglu_over10"] += int((g > 10).sum() + (np.abs(u) > 10).sum())
        if "swiglu_clamp10" in self.opts.mutations:
            g = np.minimum(g, 10.0)
            u = np.clip(u, -10.0, 10.0)
        return silu(g) * u

    def dense_ffn(self, x: np.ndarray, il: int) -> np.ndarray:
        y = self.L(il, "ffn_down") @ self.swiglu(self.L(il, "ffn_gate") @ x, self.L(il, "ffn_up") @ x)
        self.cb("ffn_out", y, il)
        return y

    def moe_ffn(self, x: np.ndarray, il: int) -> np.ndarray:
        c, o = self.cfg, self.opts
        logits = self.L(il, "ffn_gate_inp") @ x                                     # (n_expert, T)
        probs = sigmoid(logits)
        bias = self.L(il, "exp_probs_b", "bias")
        biased = probs + bias[:, None]
        self.cb("ffn_moe_logits", logits, il)
        self.cb("ffn_moe_probs", probs, il)
        top = np.argsort(-biased, axis=0, kind="stable")[: c.n_used]               # (n_used, T)
        cols = np.arange(x.shape[1])[None, :]
        w = (biased if "router_bias_in_weights" in o.mutations else probs)[top, cols]
        if c.w_norm:
            ssum = w.sum(axis=0, keepdims=True)
            w = w / (np.maximum(ssum, LLAMA_NORM_CLAMP) if o.llama_norm_clamp else ssum + 1e-20)
        w = w * c.w_scale
        srt = -np.sort(-biased, axis=0)
        if srt.shape[0] > c.n_used:
            COVER["moe_near_ties"] += int((srt[c.n_used - 1] - srt[c.n_used] < 1e-9).sum())
        self.cb("ffn_moe_topk", top, il)
        self.cb("ffn_moe_weights", w, il)
        moe = np.zeros_like(x)
        for e in np.unique(top):
            hit_k, hit_t = np.nonzero(top == e)
            xe = x[:, hit_t]
            ye = self.expert(il, "ffn_down_exps", e) @ self.swiglu(
                self.expert(il, "ffn_gate_exps", e) @ xe, self.expert(il, "ffn_up_exps", e) @ xe)
            moe[:, hit_t] += ye * w[hit_k, hit_t][None, :]        # an expert is picked once per token
        self.cb("ffn_moe_out", moe, il)
        sh = self.L(il, "ffn_down_shexp") @ self.swiglu(self.L(il, "ffn_gate_shexp") @ x,
                                                         self.L(il, "ffn_up_shexp") @ x)
        self.cb("ffn_shexp", sh, il)
        y = moe + sh
        self.cb("ffn_out", y, il)
        return y

    # ================================ blocks ================================

    def block(self, x: np.ndarray, il: int, pos: np.ndarray, prev_sel, dense: bool = False):
        """One decoder block; returns (x_out, selection used, selection to carry forward)."""
        c, o = self.cfg, self.opts
        h = rms_norm(x, self.L(il, "attn_norm"), c.rms_eps)
        self.cb("attn_norm", h, il)
        qr = rms_norm(self.L(il, "attn_q_a") @ h, self.L(il, "attn_q_a_norm"), c.rms_eps)
        self.cb("qr", qr, il)
        T = x.shape[1]
        if dense:
            sel = np.tril(np.ones((T, T), dtype=bool))
        elif c.is_full(il):
            sel = self.indexer(h, qr, il, pos)
            prev_sel = sel
        elif "shared_as_full" in o.mutations:
            if not self.has(il, "indexer.attn_q_b"):
                raise ValueError(f"shared_as_full: layer {il} carries no indexer weights")
            sel = self.indexer(h, qr, il, pos)
            prev_sel = sel
        else:
            if prev_sel is None:
                raise ValueError(f"shared layer {il} has no previous full layer")
            sel = prev_sel
            self.cb("top_k", sel, il)
        x = x + self.attention(h, qr, il, pos, sel)
        self.cb("ffn_inp", x, il)
        h2 = rms_norm(x, self.L(il, "ffn_norm"), c.rms_eps)
        self.cb("ffn_norm", h2, il)
        x = x + (self.moe_ffn(h2, il) if c.is_moe(il) else self.dense_ffn(h2, il))
        self.cb("l_out", x, il)
        return x, prev_sel

    def positions(self, T: int, pos0: int) -> np.ndarray:
        return np.arange(pos0, pos0 + T, dtype=np.int64)

    def run_layers(self, x: np.ndarray, a: int, b: int, pos: np.ndarray) -> np.ndarray:
        """Layers [a, b) on hidden states x (n_embd, T).  Layer a must be full (a shared layer
        needs the selection of the full layer before it)."""
        if a < self.cfg.n_layer and not self.cfg.is_full(a) and "shared_as_full" not in self.opts.mutations:
            raise ValueError(f"layer {a} is a shared indexer layer: start the slice at a full layer")
        prev = None
        for il in range(a, b):
            x, prev = self.block(x, il, pos, prev)
        return x

    def forward(self, tokens, pos0: int = 0) -> np.ndarray:
        """Teacher-forced pass over `tokens` at positions pos0.. ; returns (vocab, T) float64 logits.
        The post-output-norm hidden (llama.cpp t_h_nextn) is kept in self.h_nextn for mtp()."""
        c = self.cfg
        self.dumps = {}
        x = self.embed(tokens)
        pos = self.positions(x.shape[1], pos0)
        x = self.run_layers(x, 0, c.n_layer, pos)
        hn = rms_norm(x, self.W("output_norm.weight"), c.rms_eps)
        self.h_nextn = hn
        self.cb("result_norm", hn)
        logits = self.W("output.weight") @ hn
        self.cb("result_output", logits)
        return logits

    def head(self, x: np.ndarray, cols=None, chunk_rows: int = 8192) -> np.ndarray:
        """output_norm then the head on columns `cols` of the trunk output x (n_embd, T): (vocab, len(cols)).  The
        head is read in chunks of rows (the real one is 154880 x 6144) unless the weight cache holds it."""
        c = self.cfg
        xs = x if cols is None else x[:, np.asarray(cols, dtype=np.int64)]
        hn = rms_norm(xs, self.W("output_norm.weight"), c.rms_eps)
        self.cb("result_norm", hn)
        if self.cache_on:
            logits = self.W("output.weight") @ hn
        else:
            n_v = self.g.info("output.weight").shape[1]
            logits = np.empty((n_v, hn.shape[1]))
            for r0 in range(0, n_v, chunk_rows):
                r1 = min(r0 + chunk_rows, n_v)
                logits[r0:r1] = self.g.rows("output.weight", np.arange(r0, r1)).astype(F64) @ hn
        self.cb("result_output", logits)
        return logits

    def mtp(self, h_prev: np.ndarray, next_tokens, pos0: int = 0) -> np.ndarray:
        """NextN draft logits (vocab, T): row t combines the trunk's post-norm hidden h_prev[:, t]
        with the embedding of next_tokens[t] (the token at t+1) at position pos0 + t, and predicts
        the token at t+2.  [enorm(e) ; hnorm(h)] -> eh_proj -> one full decoder block (own indexer in
        hf mode, dense MLA in llama mode) -> shared_head_norm (fallback output_norm) -> head."""
        c, o = self.cfg, self.opts
        if c.n_nextn < 1:
            raise ValueError("this GGUF has no NextN block")
        il = c.n_layer
        e = self.embed(next_tokens)
        pos = self.positions(e.shape[1], pos0)
        en = rms_norm(e, self.L(il, "nextn.enorm"), c.rms_eps)
        hn = rms_norm(h_prev, self.L(il, "nextn.hnorm"), c.rms_eps)
        x = self.L(il, "nextn.eh_proj") @ np.concatenate([en, hn], axis=0)
        self.cb("mtp_eh_proj", x, il)
        x, _ = self.block(x, il, pos, None, dense=o.mtp_dense)
        nw = self.L(il, "nextn.shared_head_norm") if self.has(il, "nextn.shared_head_norm") \
            else self.W("output_norm.weight")
        hn2 = rms_norm(x, nw, c.rms_eps)
        head = self.L(il, "nextn.shared_head_head") if self.has(il, "nextn.shared_head_head") \
            else self.W("output.weight")
        logits = head @ hn2
        self.cb("mtp_result_output", logits)
        return logits


# ================================ CLI ================================

def _parse_layers(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


def _write_rows(path: str, arr: np.ndarray) -> None:
    out = np.ascontiguousarray(arr.astype(np.float32))
    if path.endswith(".npy"):
        np.save(path, out)
    else:
        out.tofile(path)


def run_slice(m: "GlmDsa", args) -> int:
    """--layers A:B --hidden-in H (see the module docstring): returns the exit status (1 when a gate fails)."""
    c = m.cfg
    x = load_hidden(args.hidden_in, c.n_embd)
    a, b = args.layers
    if not 0 <= a < b <= c.n_layer:
        raise SystemExit(f"glm_dsa.py: --layers {a}:{b} outside the trunk 0:{c.n_layer}")
    T = x.shape[1]
    cols = parse_positions(args.positions, T, args.pos0)
    y = m.run_layers(x, a, b, m.positions(T, args.pos0))
    if args.hidden_out:
        np.save(args.hidden_out, y.astype(np.float32))
    print(f"layers {a}:{b} on {T} positions from {args.pos0}: out mean {y.mean():.6g} std {y.std():.6g}")
    rep: dict = {"layers": [a, b], "pos0": args.pos0, "T": T, "positions": [int(args.pos0 + i) for i in cols]}
    status = 0
    if b == c.n_layer and (args.logits_out or args.compare_engine or args.compare_llama):
        lg = m.head(y, cols)                                                        # (vocab, P)
        print(f"head at {cols.size} positions: argmax {[int(i) for i in lg.argmax(axis=0)][:16]}")
        if args.logits_out:
            _write_rows(args.logits_out, lg.T)
        cols_abs = cols + args.pos0
        kls = {}
        for key, path, first in (("engine", args.compare_engine, args.engine_first),
                                 ("llama", args.compare_llama, args.llama_first)):
            if not path:
                continue
            other = load_logit_rows(path, c.n_vocab, first, cols_abs)
            k = kl_top_rest(lg, other)
            same = lg.argmax(axis=0) == other.argmax(axis=0)
            kls[key] = float(k.mean())
            rep[key] = {"kl_mean": float(k.mean()), "kl_max": float(k.max()), "kl": [float(v) for v in k],
                        "top1_same": int(same.sum()), "n": int(k.size)}
            print(f"KL(slice || {key}) mean {k.mean():.6g} max {k.max():.4g}, same top-1 {int(same.sum())}/{k.size}")
        if "engine" in kls and "llama" in kls:
            ok, why = slice_gate(kls["engine"], kls["llama"], args.kl_floor)
            rep["gate"] = {"pass": ok, "why": why}
            print(f"B7.5 slice gate: {'PASS' if ok else 'FAIL'} - {why}")
            status = 0 if ok else 1
    elif args.logits_out or args.compare_engine or args.compare_llama:
        print(f"(no head: the slice ends at layer {b}, not the trunk's last {c.n_layer})")
    if args.dump_dir:
        d = pathlib.Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in m.dumps.items():
            np.save(d / f"{name}.npy", arr)
        print(f"dumps: {len(m.dumps)} -> {d}")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(rep, indent=1) + "\n")
    print("coverage:", dict(COVER))
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="glm-dsa reference forward pass (numpy float64)")
    ap.add_argument("model", help="a glm-dsa GGUF (any shard of a split file)")
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8", help="comma-separated token ids")
    ap.add_argument("--mode", choices=("hf", "llama"), default="hf")
    ap.add_argument("--idx-ln-eps", type=float, default=None, help="override the indexer LayerNorm eps")
    ap.add_argument("--pos0", type=int, default=0, help="position of the first token")
    ap.add_argument("--mutate", default="", help=f"comma list of {','.join(MUTATIONS)}")
    ap.add_argument("--mtp", action="store_true", help="also compute NextN draft logits for rows 0..T-2")
    ap.add_argument("--mtp-out", default=None,
                    help="with --mtp: write the (T-1, vocab) float32 draft logits (.npy or raw .f32)")
    ap.add_argument("--logits-out", default=None, help="write (T, vocab) float32 logits (.npy or raw .f32)")
    ap.add_argument("--dump-dir", default=None, help="write every cb() seam as .npy")
    ap.add_argument("--layers", type=_parse_layers, default=None, help="A:B slice (needs --hidden-in)")
    ap.add_argument("--hidden-in", default=None, help="(n_embd, T) .npy hidden state entering layer A")
    ap.add_argument("--hidden-out", default=None, help="write the slice's output hidden state here")
    ap.add_argument("--positions", default=None,
                    help="slice: absolute positions to report (a:b, a,b,c or -N for the last N; default all)")
    ap.add_argument("--compare-engine", default=None, help="slice: engine logits / log-softmax rows (float32)")
    ap.add_argument("--engine-first", type=int, default=0, help="absolute position of --compare-engine's row 0")
    ap.add_argument("--compare-llama", default=None, help="slice: llama.cpp logits rows (llm_logits_dump out.bin)")
    ap.add_argument("--llama-first", type=int, default=0, help="absolute position of --compare-llama's row 0")
    ap.add_argument("--kl-floor", type=float, default=0.0, help="slice gate: KL(slice || engine) <= F also passes")
    ap.add_argument("--report", default=None, help="slice: write the comparison as JSON")
    ap.add_argument("--greedy", type=int, default=0, help="continue greedily for N tokens")
    args = ap.parse_args()

    muts = frozenset(m for m in args.mutate.split(",") if m)
    m = GlmDsa(args.model, Options(mode=args.mode, idx_ln_eps=args.idx_ln_eps, mutations=muts))
    m.collect = bool(args.dump_dir)
    print(m.cfg.summary())
    if args.layers:
        if not args.hidden_in:
            ap.error("--layers needs --hidden-in")
        return run_slice(m, args)
    tokens = [int(t) for t in args.tokens.split(",") if t]
    logits = m.forward(tokens, args.pos0)
    if not np.isfinite(logits).all():
        print("glm_dsa.py: NON-FINITE LOGITS")
        return 1
    print(f"logits {logits.shape}: argmax per position {[int(i) for i in logits.argmax(axis=0)]}")
    if args.logits_out:
        out = np.ascontiguousarray(logits.T.astype(np.float32))
        if args.logits_out.endswith(".npy"):
            np.save(args.logits_out, out)
        else:
            out.tofile(args.logits_out)
    if args.mtp and len(tokens) > 1:
        d = m.mtp(m.h_nextn[:, :-1], tokens[1:], args.pos0)
        print(f"mtp draft argmax per row {[int(i) for i in d.argmax(axis=0)]}")
        if args.mtp_out:
            out = np.ascontiguousarray(d.T.astype(np.float32))
            if args.mtp_out.endswith(".npy"):
                np.save(args.mtp_out, out)
            else:
                out.tofile(args.mtp_out)
    if args.greedy:
        toks = list(tokens)
        for _ in range(args.greedy):
            toks.append(int(np.argmax(m.forward(toks, args.pos0)[:, -1])))
        print("greedy continuation:", toks[len(tokens):])
    if args.dump_dir:
        d = pathlib.Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in m.dumps.items():
            np.save(d / f"{name}.npy", arr)
        print(f"dumps: {len(m.dumps)} -> {d}")
    print("coverage:", dict(COVER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
