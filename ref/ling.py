"""ref/ling.py - Ling-3.2-air's (inclusionAI, llama.cpp arch `bailingmoe3`) reference forward pass over its GGUF, one
op at a time: the specification maya-ling's kernels are parity-tested against.

The definition is llama.cpp's (src/models/bailingmoe3.cpp and build_moe_ffn / build_ffn in src/llama-graph.cpp, at
fc35562ba), whose conversion (conversion/bailingmoe3.py) wrote the file; checked against the HF code
(modeling_bailing_moe_v3_vl.py) where the two could differ - see "HF vs llama.cpp" below. Conventions as ref/glm.py's:
activations (features, tokens), features fastest like a ggml ne[0]; weights from ref/load.py as (out, in), so y = W @ x
is ggml_mul_mat; heads split in ggml's reshape order (split_heads / join_heads, imported from glm.py); every seam a
kernel needs recorded by cb() under llama.cpp's cb name ("<name>-<il>"). Float64 throughout (the seams are kept as
float32 copies). Teacher-forced over the whole sequence; decode and chunked prefill are the engine's.

    python ref/ling.py --gguf MODEL.gguf [--tokens 1,2,3 | --tokens @ids.txt] [--dump-dir DIR] [--mtp]

The forward (42 layers; 43 in block_count with the MTP block):
  x     = token_embd[tokens]
  layer il:
    a   = rms(x) * attn_norm
    KDA layer (head_count_kv[il] == 0; 35 of 42):
      q, k, v = silu(causal_conv4(W_{q,k,v} a))        per channel, taps oldest first, zero history
      q, k    = l2(q), l2(k) per head                   ggml_l2_norm: x / max(|x|, eps)
      g       = lb * sigmoid((W_f a + dt_bias) * A)     A = ssm_a = exp(A_log) per head, lb = -5 (safe gate);
                                                        a log-decay per KEY channel, in (lb, 0)
      beta    = sigmoid(W_beta a)                       per head
      per head, state S (dk, dv) from zero:
        S = diag(exp(g_t)) S;  S += k_t (beta_t (v_t - S^T k_t))^T;  o_t = S^T q_t / sqrt(dk)
      out     = W_o (rms(o) * ssm_norm * sigmoid(W_g a))            (the norm per head over dk)
    MLA layer (every 6th: 5, 11, .., 41):
      q       = W_q a, per head [nope 128 | rope 64]    (no q LoRA in this checkpoint; the LoRA form is supported)
      [c | k_pe] = W_kv_a a,  c = rms(c) * kv_a_norm (512), k_pe (64) shared by the heads, not normed
      rope:   NORM (pairs 2i, 2i+1), theta 6e6 over 64 dims, on q_rope and k_pe
      score   = ((W_k_b,h q_nope) . c_s + q_rope . k_pe_s) / sqrt(192), causal softmax over s
      o_h     = W_v_b,h (sum_s p_s c_s)
      out     = W_o (o_h * sigmoid(W_gate a)_h)         the head-wise gate, from the attention's input
    h   = x + out
    f   = rms(h) * ffn_norm
    FFN: layers 0-1 dense SwiGLU (6144); after that
      p       = sigmoid(W_r f) (512), s = p + exp_probs_b (the bias only for the choice)
      groups: 8 of 64 experts, a group's score the sum of its best two s; the best 4 groups kept, the rest -inf
      top-8 by s; weights = p / max(sum p, 2^-14) * 2.5
      moe     = sum_j w_j down_e(swiglu_L(gate_e f, up_e f)),  L = swiglu_clamp_exp[il]
      shared  = down_sh(swiglu_L'(gate_sh f, up_sh f)),       L' = swiglu_clamp_shexp[il] (also the dense layers')
      swiglu_L(g, u) = min(silu(g), L) * clamp(u, -L, L)  - the clamp AFTER the SiLU; L = 0 means none
      (exp: L = 4 on 35-41; shexp: 5 on 34-39, 7 on 40-41; both 0 on the MTP block)
    x   = h + moe + shared
  h_nextn = rms(x) * output_norm;  logits = output h_nextn

The MTP block (blk.42, an MLA layer + MoE; mtp_logits): row r pairs the trunk's h_nextn at position t with the token at
t + 1 (llama.cpp's common/speculative.cpp and HF's roll(-1) agree), rows attending causally to one another:
  m = eh_proj [rms(emb(token)) * enorm ; rms(h) * hnorm]     the embedding's half FIRST
  then the MLA layer and the MoE as above (il = 42), then rms * layer_output_norm (HF's final_layernorm) -> output.

GGUF: arch bailingmoe3; block_count 43 = 42 + nextn_predict_layers 1; head_count_kv a per-layer array (0 = KDA);
key_length 576 (= kv_lora + rope), key_length_mla 192, value_length_mla 128, rope.dimension_count 64;
swiglu_clamp_{exp,shexp} per-layer arrays padded with 0 to block_count; ssm_a holds exp(A_log) (positive);
ssm_conv1d_* [4, 1, 4096, 1]; attn_k_b [128, 512, 32] (W_k_b,h: nope -> latent), attn_v_b [512, 128, 32];
ffn_gate_inp F32; the head-wise attention gate is attn_gate, the KDA output gate ssm_g_a (full, no LoRA), the KDA
decay projection ssm_f_a (full).

HF vs llama.cpp (the forms this file follows are llama.cpp's; none of these differ beyond rounding):
  * q/k L2 norm: FLA's kernel x * rsqrt(sum x^2 + 1e-6), ggml x / max(|x|, 1e-6);
  * routing weight renorm: HF sum + 1e-20, llama.cpp max(sum, 2^-14);
  * RoPE: HF's rope_interleave path on pair-interleaved weights == ggml NORM; the MTP rows sit one position later in
    llama.cpp than in HF, which RoPE (relative) does not see.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from glm import join_heads, split_heads, top_k_desc  # noqa: E402
from load import Gguf  # noqa: E402

F64 = np.float64
RENORM_MIN = 6.103515625e-5      # build_moe_ffn's floor on the routing weights' sum (the smallest normal f16)
CLAMP_OFF = 1e-6                 # a clamp limit at or below this is "no clamp" (llama-graph.cpp's eps)

# Clamp hits per forward, keyed (exp|shexp, layer, gate|up, limit): a fixture that never hits a clamp proves nothing
# about it.
COVER: Counter = Counter()


class LingConfig:
    """every number the forward uses, from the GGUF's bailingmoe3.* keys (llama.cpp's LLM_KV names); a key the
    forward cannot do without is required, so a misnamed key fails here instead of running a different model"""

    def __init__(self, md: dict):
        a = "bailingmoe3."

        def req(key):
            if a + key not in md:
                raise KeyError(a + key)
            return md[a + key]

        def opt(key, default):
            v = md.get(a + key)
            return default if v is None else v

        self.n_layer_all = int(req("block_count"))
        self.n_nextn = int(opt("nextn_predict_layers", 0))
        self.n_layers = self.n_layer_all - self.n_nextn
        self.n_embd = int(req("embedding_length"))
        self.n_head = int(req("attention.head_count"))
        kv = req("attention.head_count_kv")
        self.n_head_kv = [int(x) for x in kv] if isinstance(kv, list) else [int(kv)] * self.n_layer_all
        self.eps = float(req("attention.layer_norm_rms_epsilon"))

        self.kv_lora = int(req("attention.kv_lora_rank"))
        self.q_lora = int(opt("attention.q_lora_rank", 0))
        self.qk_head = int(req("attention.key_length_mla"))
        self.v_head = int(req("attention.value_length_mla"))
        self.n_rot = int(req("rope.dimension_count"))
        self.qk_nope = self.qk_head - self.n_rot
        self.rope_base = float(req("rope.freq_base"))
        if int(req("attention.key_length")) != self.kv_lora + self.n_rot:
            raise ValueError("attention.key_length != kv_lora_rank + rope dims")

        self.d_conv = int(req("ssm.conv_kernel"))
        self.kda_head = int(req("kda.head_dim"))
        self.kda_lb = float(req("kda.gate_lower_bound"))
        if not bool(opt("kda.safe_gate", True)) or self.kda_lb >= 0:
            raise ValueError("bailingmoe3 needs the safe KDA gate with a negative lower bound")
        self.d_inner = self.kda_head * self.n_head

        self.n_expert = int(req("expert_count"))
        self.n_expert_used = int(req("expert_used_count"))
        self.n_group = int(opt("expert_group_count", 0))
        self.n_group_used = int(opt("expert_group_used_count", 0))
        if int(req("expert_gating_func")) != 2:
            raise ValueError("bailingmoe3: only sigmoid gating (expert_gating_func 2) is implemented")
        self.n_ff = int(req("feed_forward_length"))
        self.n_ff_exp = int(req("expert_feed_forward_length"))
        self.n_ff_shexp = int(opt("expert_shared_feed_forward_length", 0)) or \
            self.n_ff_exp * max(1, int(req("expert_shared_count")))
        self.n_dense_lead = int(req("leading_dense_block_count"))
        self.w_scale = float(opt("expert_weights_scale", 0.0))
        self.w_norm = bool(opt("expert_weights_norm", False))

        def per_layer(key):
            v = opt(key, 0.0)
            v = [float(x) for x in v] if isinstance(v, list) else [float(v)] * self.n_layer_all
            if len(v) != self.n_layer_all:
                raise ValueError(f"{a}{key}: {len(v)} values for {self.n_layer_all} blocks")
            return v

        self.clamp_exp = per_layer("swiglu_clamp_exp")
        self.clamp_shexp = per_layer("swiglu_clamp_shexp")

    def is_recr(self, il: int) -> bool:
        return self.n_head_kv[il] == 0


# ================================ float64 primitives ================================

def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    """ggml_rms_norm over the feature axis (ne[0]); the weight broadcasts over every trailing axis"""
    y = x / np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)
    return y if w is None else y * w.reshape((-1,) + (1,) * (x.ndim - 1))


def l2_norm(x: np.ndarray, eps: float) -> np.ndarray:
    """ggml_l2_norm: x / max(||x||, eps) over the feature axis"""
    return x / np.maximum(np.sqrt(np.sum(x * x, axis=0, keepdims=True)), eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def swiglu(gate: np.ndarray, up: np.ndarray, limit: float, tag: tuple) -> np.ndarray:
    """build_ffn / build_moe_ffn's SwiGLU with the bailingmoe3 limit: up = clamp(up, -L, L), act = min(silu(gate), L).
    The gate is clamped AFTER the SiLU and above only (DeepSeek-V4 clamps it before - not this arch)."""
    a = silu(gate)
    if limit <= CLAMP_OFF:
        return a * up
    COVER[(*tag, "gate", limit)] += int((a > limit).sum())
    COVER[(*tag, "up", limit)] += int((np.abs(up) > limit).sum())
    return np.minimum(a, limit) * np.clip(up, -limit, limit)


def rope_norm(x: np.ndarray, pos: np.ndarray, base: float) -> np.ndarray:
    """ggml_rope_ext, LLAMA_ROPE_TYPE_NORM, no scaling: x is (n_rot, n_head, T); the pairs are (2i, 2i+1) and pair i
    turns by pos * base^(-2i / n_rot)"""
    n_rot = x.shape[0]
    inv = base ** (-np.arange(0, n_rot, 2, dtype=F64) / n_rot)
    ang = inv[:, None] * pos[None, :].astype(F64)                  # (n_rot / 2, T)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x0 * c - x1 * s
    out[1::2] = x0 * s + x1 * c
    return out


# ================================ the model ================================

class Ling:
    def __init__(self, path):
        self.g = Gguf(str(path))
        self.cfg = LingConfig(self.g.g.metadata)
        self.dumps: dict[str, np.ndarray] = {}
        self.h_nextn: np.ndarray | None = None
        self._output: np.ndarray | None = None

    # ---- weights (out, in), so every use below is y = W @ x
    def t(self, name: str) -> np.ndarray:
        return self.g.tensor(name).astype(F64)

    def lt(self, il: int, base: str, suffix: str = "weight") -> np.ndarray:
        return self.t(f"blk.{il}.{base}.{suffix}")

    def has(self, name: str) -> bool:
        return name in self.g._by_name

    def embed(self, tokens) -> np.ndarray:
        return self.g.rows("token_embd.weight", tokens).T.astype(F64)        # (n_embd, T)

    def output(self) -> np.ndarray:
        if self._output is None:
            self._output = self.t("output.weight" if self.has("output.weight") else "token_embd.weight")
        return self._output

    def cb(self, name: str, x: np.ndarray, il: int = -1) -> None:
        self.dumps[f"{name}-{il}" if il >= 0 else name] = np.array(x, dtype=np.float32)

    # ================================ KDA ================================

    def conv_silu(self, x: np.ndarray, il: int, s: str) -> np.ndarray:
        """ggml_ssm_conv over [zero history | x] then SiLU: out[c, t] = sum_j w[c, j] x[c, t - (d_conv - 1) + j]"""
        w = self.t(f"blk.{il}.ssm_conv1d_{s}.weight").reshape(-1, self.cfg.d_conv)      # (d_inner, d_conv)
        d = self.cfg.d_conv
        xp = np.concatenate([np.zeros((x.shape[0], d - 1), dtype=F64), x], axis=1)
        out = sum(w[:, j:j + 1] * xp[:, j:j + x.shape[1]] for j in range(d))
        return silu(out)

    def kda_layer(self, cur: np.ndarray, il: int) -> np.ndarray:
        c = self.cfg
        dk, H, T = c.kda_head, c.n_head, cur.shape[1]
        q = l2_norm(split_heads(self.conv_silu(self.lt(il, "attn_q") @ cur, il, "q"), dk, H), c.eps)
        k = l2_norm(split_heads(self.conv_silu(self.lt(il, "attn_k") @ cur, il, "k"), dk, H), c.eps)
        v = split_heads(self.conv_silu(self.lt(il, "attn_v") @ cur, il, "v"), dk, H)

        f = self.lt(il, "ssm_f_a") @ cur + self.lt(il, "ssm_dt", "bias")[:, None]
        A = self.t(f"blk.{il}.ssm_a").reshape(-1)                                     # (H,), exp(A_log)
        g = c.kda_lb * sigmoid(split_heads(f, dk, H) * A[None, :, None])               # (dk, H, T)
        self.cb("kda_gate", g, il)
        beta = sigmoid(self.lt(il, "ssm_beta") @ cur)                                  # (H, T)

        scale = 1.0 / np.sqrt(dk)
        self.cb("q_in", q * scale, il)
        self.cb("k_in", k, il)
        self.cb("v_in", v, il)
        self.cb("b_in", beta[None], il)

        S = np.zeros((H, dk, dk), dtype=F64)                                           # (head, key, value)
        o = np.empty((dk, H, T), dtype=F64)
        for t in range(T):
            S *= np.exp(g[:, :, t]).T[:, :, None]
            kt, qt = k[:, :, t].T, q[:, :, t].T
            delta = (v[:, :, t].T - np.einsum("hkv,hk->hv", S, kt)) * beta[:, t, None]
            S += kt[:, :, None] * delta[:, None, :]
            o[:, :, t] = np.einsum("hkv,hk->hv", S, qt).T * scale
        self.cb("attn_output", o, il)

        og = split_heads(self.lt(il, "ssm_g_a") @ cur, dk, H)
        o = rms_norm(o, self.t(f"blk.{il}.ssm_norm.weight"), c.eps) * sigmoid(og)
        out = self.lt(il, "attn_output") @ join_heads(o)
        self.cb("kda_out", out, il)
        return out

    # ================================ MLA ================================

    def mla_layer(self, cur: np.ndarray, il: int, pos: np.ndarray) -> np.ndarray:
        c = self.cfg
        H, T = c.n_head, cur.shape[1]
        if c.q_lora:
            q = self.lt(il, "attn_q_b") @ rms_norm(self.lt(il, "attn_q_a") @ cur, self.lt(il, "attn_q_a_norm"), c.eps)
        else:
            q = self.lt(il, "attn_q") @ cur
        q = split_heads(q, c.qk_head, H)                                               # (192, H, T)
        q_nope, q_pe = q[:c.qk_nope], rope_norm(q[c.qk_nope:], pos, c.rope_base)

        kv_all = self.lt(il, "attn_kv_a_mqa") @ cur                                    # (576, T)
        kv = rms_norm(kv_all[:c.kv_lora], self.lt(il, "attn_kv_a_norm"), c.eps)        # (512, T)
        k_pe = rope_norm(kv_all[c.kv_lora:, None, :], pos, c.rope_base)[:, 0, :]       # (64, T)

        wk_b = self.lt(il, "attn_k_b")                                                 # (H, 512, 128)
        wv_b = self.lt(il, "attn_v_b")                                                 # (H, 128, 512)
        q_abs = np.einsum("hln,nht->lht", wk_b, q_nope)                                # (512, H, T)
        score = (np.einsum("lht,ls->hts", q_abs, kv) + np.einsum("rht,rs->hts", q_pe, k_pe)) / np.sqrt(c.qk_head)
        score = np.where(np.tril(np.ones((T, T), dtype=bool))[None], score, -np.inf)
        p = np.exp(score - score.max(axis=2, keepdims=True))
        p /= p.sum(axis=2, keepdims=True)
        ctx = np.einsum("hts,ls->lht", p, kv)                                          # (512, H, T)
        o = np.einsum("hvl,lht->vht", wv_b, ctx)                                       # (128, H, T)
        # llama.cpp's kqv_out is build_attn_mha's cont_2d((v, H, T)): v fastest per head, heads outer;
        # dump it in that flat order so the two compare element for element
        self.cb("kqv_out", o.transpose(1, 0, 2), il)

        gate = sigmoid(self.lt(il, "attn_gate") @ cur)                                 # (H, T)
        out = self.lt(il, "attn_output") @ join_heads(o * gate[None])
        self.cb("mla_out", out, il)
        return out

    # ================================ FFNs ================================

    def dense_ffn(self, cur: np.ndarray, il: int, base: str = "") -> np.ndarray:
        """build_ffn (LLM_FFN_SILU, LLM_FFN_PAR): a dense layer (base "") or the shared expert (base "_shexp"); both
        take swiglu_clamp_shexp[il]"""
        lim = self.cfg.clamp_shexp[il]
        act = swiglu(self.lt(il, f"ffn_gate{base}") @ cur, self.lt(il, f"ffn_up{base}") @ cur, lim, ("shexp", il))
        if lim > CLAMP_OFF:
            self.cb("ffn_swiglu_limited", act, il)
        return self.lt(il, f"ffn_down{base}") @ act

    def route(self, cur: np.ndarray, il: int) -> tuple[np.ndarray, np.ndarray]:
        """build_moe_ffn's selection: (k, T) expert ids in llama.cpp's order (descending biased score) and their
        weights.  ggml's argsort ties go either way; top_k_desc breaks them toward the lower index."""
        c = self.cfg
        probs = sigmoid(self.lt(il, "ffn_gate_inp") @ cur)                             # (E, T), unbiased
        sel = probs + self.lt(il, "exp_probs_b", "bias")[:, None]
        self.cb("ffn_moe_probs_biased", sel, il)
        T = cur.shape[1]
        ids = np.empty((c.n_expert_used, T), dtype=np.int64)
        w = np.empty((c.n_expert_used, T), dtype=F64)
        for t in range(T):
            s = sel[:, t].copy()
            if c.n_group > 1:
                grp = s.reshape(c.n_group, -1)
                gscore = np.sort(grp, axis=1)[:, -2:].sum(axis=1)
                keep = np.zeros(c.n_group, dtype=bool)
                keep[top_k_desc(gscore, c.n_group_used)] = True
                s = np.where(np.repeat(keep, grp.shape[1]), s, -np.inf)
            e = top_k_desc(s, c.n_expert_used)
            wt = probs[e, t]
            if c.w_norm:
                wt = wt / max(wt.sum(), RENORM_MIN)
            if c.w_scale not in (0.0, 1.0):
                wt = wt * c.w_scale
            ids[:, t], w[:, t] = e, wt
        self.dumps[f"ffn_moe_topk-{il}"] = ids
        self.cb("ffn_moe_weights_scaled", w[None], il)
        return ids, w

    def moe_ffn(self, cur: np.ndarray, il: int) -> np.ndarray:
        """the routed experts (only the picked ones are read: ref/load.py's lazy expert()) plus the shared expert"""
        c = self.cfg
        ids, w = self.route(cur, il)
        lim = c.clamp_exp[il]
        T = cur.shape[1]
        act = np.zeros((c.n_ff_exp, c.n_expert_used, T), dtype=F64) if lim > CLAMP_OFF else None
        moe = np.zeros_like(cur)
        b = f"blk.{il}."
        for e in np.unique(ids):
            slot, tok = np.nonzero(ids == e)
            x = cur[:, tok]
            gate = self.g.expert(b + "ffn_gate_exps.weight", int(e)).astype(F64) @ x
            up = self.g.expert(b + "ffn_up_exps.weight", int(e)).astype(F64) @ x
            h = swiglu(gate, up, lim, ("exp", il))
            if act is not None:
                act[:, slot, tok] = h
            y = self.g.expert(b + "ffn_down_exps.weight", int(e)).astype(F64) @ h
            moe[:, tok] += y * w[slot, tok][None]                # a token picks an expert at most once
        if act is not None:
            self.cb("ffn_moe_swiglu_limited", act, il)
        self.cb("ffn_moe_out", moe, il)
        shared = self.dense_ffn(cur, il, "_shexp")
        self.cb("ffn_shexp", shared, il)
        return moe + shared

    # ================================ the trunk and the MTP block ================================

    def block(self, x: np.ndarray, il: int, pos: np.ndarray) -> np.ndarray:
        c = self.cfg
        cur = rms_norm(x, self.lt(il, "attn_norm"), c.eps)
        self.cb("attn_norm", cur, il)
        if il < c.n_layers and c.is_recr(il):
            cur = self.kda_layer(cur, il)
        else:
            cur = self.mla_layer(cur, il, pos)
        h = x + cur
        self.cb("ffn_inp", h, il)
        cur = rms_norm(h, self.lt(il, "ffn_norm"), c.eps)
        cur = self.dense_ffn(cur, il) if il < c.n_dense_lead else self.moe_ffn(cur, il)
        out = h + cur
        self.cb("l_out", out, il)
        return out

    def forward(self, tokens, log=None) -> np.ndarray:
        """one teacher-forced pass; returns (vocab, T) float64 logits and keeps h_nextn (n_embd, T) for the MTP block"""
        c = self.cfg
        x = self.embed(tokens)
        self.cb("model.input_embed", x)
        pos = np.arange(x.shape[1])
        for il in range(c.n_layers):
            t0 = time.time()
            x = self.block(x, il, pos)
            if log:
                log(f"layer {il:2d} {'KDA' if c.is_recr(il) else 'MLA'} {time.time() - t0:6.1f} s")
        self.h_nextn = rms_norm(x, self.t("output_norm.weight"), c.eps)
        self.cb("result_norm", self.h_nextn)
        logits = self.output() @ self.h_nextn
        self.cb("result_output", logits)
        return logits

    def mtp_logits(self, h: np.ndarray, next_tokens, pos: np.ndarray | None = None) -> np.ndarray:
        """the MTP block (blk.<n_layers>): h (n_embd, R) the trunk's h_nextn, next_tokens the R tokens that follow
        them; row r predicts the token after next_tokens[r].  The rows attend causally to one another, i.e. this is
        the MTP layer's own cache, teacher-forced; pos defaults to 1..R (only differences matter).
        Returns (vocab, R)."""
        c = self.cfg
        if c.n_nextn < 1:
            raise ValueError("no MTP block in this GGUF (nextn_predict_layers 0)")
        il = c.n_layers
        R = h.shape[1]
        pos = np.arange(1, R + 1) if pos is None else np.asarray(pos)
        e = rms_norm(self.embed(next_tokens), self.lt(il, "nextn.enorm"), c.eps)
        hn = rms_norm(h, self.lt(il, "nextn.hnorm"), c.eps)
        x = self.lt(il, "nextn.eh_proj") @ np.concatenate([e, hn], axis=0)
        self.cb("mtp_eh_proj", x, il)
        x = self.block(x, il, pos)
        x = rms_norm(x, self.lt(il, "layer_output_norm"), c.eps)
        self.cb("mtp_h_nextn", x, il)
        return self.output() @ x


def _tokens(arg: str) -> list[int]:
    s = pathlib.Path(arg[1:]).read_text() if arg.startswith("@") else arg
    return [int(x) for x in re.findall(r"-?\d+", s)]


def main() -> int:
    ap = argparse.ArgumentParser(description="bailingmoe3 (Ling-3.2-air) reference forward pass")
    ap.add_argument("--gguf", required=True, help="a bailingmoe3 GGUF (any single file or shard set ref/load.py reads)")
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8", help="comma-separated ids, or @FILE holding them")
    ap.add_argument("--dump-dir", default=None, help="write every cb() seam, the logits and h_nextn here as .npy")
    ap.add_argument("--mtp", action="store_true", help="also run the MTP block over (h_nextn[t], tokens[t + 1])")
    ap.add_argument("--quiet", action="store_true", help="no per-layer timing")
    args = ap.parse_args()

    m = Ling(args.gguf)
    c = m.cfg
    print(f"bailingmoe3: n_embd {c.n_embd}, layers {c.n_layers} (+{c.n_nextn} MTP; "
          f"{sum(c.is_recr(i) for i in range(c.n_layers))} KDA / {sum(not c.is_recr(i) for i in range(c.n_layers))} MLA), "
          f"experts {c.n_expert} x top{c.n_expert_used} in {c.n_group} groups (keep {c.n_group_used}), "
          f"dense lead {c.n_dense_lead}, mla {c.qk_nope}+{c.n_rot}/kv {c.kv_lora}/v {c.v_head}, kda {c.n_head}x{c.kda_head}",
          flush=True)
    tokens = _tokens(args.tokens)
    log = None if args.quiet else (lambda s: print(s, flush=True))
    logits = m.forward(tokens, log)
    if not np.isfinite(logits).all():
        print("ling.py: NON-FINITE LOGITS")
        return 1
    print(f"logits {logits.shape}; argmax per position: {[int(i) for i in logits.argmax(axis=0)]}")
    mtp = None
    if args.mtp:
        mtp = m.mtp_logits(m.h_nextn[:, :-1], tokens[1:])
        print(f"mtp logits {mtp.shape}; argmax per row: {[int(i) for i in mtp.argmax(axis=0)]}")
    hits = [f"{t} {il} {p} {n}" for (t, il, p, _), n in sorted(COVER.items()) if n]
    print("clamp hits (kind layer part n):", ", ".join(hits) if hits else "none")
    if args.dump_dir:
        d = pathlib.Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in m.dumps.items():
            np.save(d / f"{name}.npy", arr)
        np.save(d / "logits.npy", logits)
        np.save(d / "h_nextn.npy", m.h_nextn)
        (d / "tokens.txt").write_text(",".join(str(t) for t in tokens))
        if mtp is not None:
            np.save(d / "mtp_logits.npy", mtp)
        (d / "cover.txt").write_text("".join(f"{t} {il} {p} {lim:g} {n}\n" for (t, il, p, lim), n in sorted(COVER.items())))
        print(f"dumps: {len(m.dumps)} seams -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
