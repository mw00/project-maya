"""ref/inkling2.py - Inkling 2's (thinkingmachines/Inkling-2-Small) reference forward pass over an `inkling2` GGUF,
one op at a time: the specification an Inkling 2 engine's kernels are parity-tested against.

The definition is transformers' own (models/inkling2/modeling_inkling2.py, in its main from 109a7d19bf); the file
layout llama.cpp's Inkling 2 conversion (ggml-org/llama.cpp PR #32187, unsloth/Inkling-2-Small-GGUF's files; written
by tools/inkling2_gguf.py, its conventions in that file's docstring). Checked against transformers end to end:
tests/inkling2_ref_test.py - the tiny model's log-probabilities within fp32's rounding of transformers'.

    python ref/inkling2.py MODEL.gguf [--tokens 1,2,3] [--dump-dir DIR]

Conventions (as ref/glm.py's where they apply): weights from ref/load.py as (out, in), y = W @ x; activations here
(tokens, features) - torch's - with every intermediate a kernel needs recorded by cb() under the op's name
(blk.L.<op>), so a kernel's output can be compared with it a seam at a time. Float64 throughout: this is the spec,
not a fast path. Teacher-forced over the whole sequence; decode and chunked prefill are the engine's.

The forward, a layer (L: sliding but every 6th, layers 5, 11, .. full):
  h_in  = attn_res(blocks, h) with blk.L.attn_res_score (none at layer 0; a full layer's 3 queries give its q, k
          and v inputs)
  attn  = attention(norm(h_in)): q, k, v, r (relative states), gate, sink projections; k and v through their short
          convs (a full layer's dynamic: static taps + per-token taps from the normed k / v input); q's scaled L2
          clip, k's L2 clip, v's weightless RMS norm, r's scaled L2 clip; a full layer's q and r times
          tau = 1 + alpha ln max(1, (pos + 1) / n_floor); logits q.k / sqrt(head_dim) + the relative bias
          r[t] . rel_proj[:, t - s] (0 <= t - s < extent, else 0), causal (sliding: t - s < window), a sink logit
          a head a token in the softmax (its value zero); the output times sigmoid(gate), then attn_output
  a     = shortconv_attn(attn)
  at a block boundary (L % 6 == 0): blocks += [h] and the stream restarts: r = a; else r = h + a
  m     = shortconv_mlp(moe(ffn_norm(attn_res(blocks, r) with ffn_res_score)))
  h     = r + m
  the MoE: sigmoid(router logits); the top 8 by score + exp_probs_b, their weights the scores normalized x 8;
          an expert: g = W_g x * s_g[e], u = W_u x * s_u[e] (the NVFP4 scale; 1 without one),
          W_d (4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25)) * s_d[e]
  at the end: output_norm(attn_res(blocks, h) with output_res_score) -> output -> 30 tanh(l / 30), the first
          unpadded_vocab_size logits
  attn_res(blocks, x): the states [blocks..., x] each RMS-normed (no weight, eps 1e-6), softmax(q . s / sqrt(C)),
          the normed states weighted by it
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from load import Gguf  # noqa: E402

DT = torch.float64
RES_EPS = 1e-6   # the attention residual's norm (transformers' Inkling2UnweightedRMSNorm() default, not the config's)


class Config:
    """every number the forward uses, from the GGUF's inkling2.* keys"""

    def __init__(self, md):
        a = "inkling2."
        g = lambda k: md[a + k]  # noqa: E731
        self.n_layers, self.n_embd = g("block_count"), g("embedding_length")
        self.n_head, self.n_kv, self.hd = g("attention.head_count"), g("attention.head_count_kv"), g("attention.key_length")
        self.eps = float(g("attention.layer_norm_rms_epsilon"))
        self.n_exp, self.k_exp, self.route_scale = g("expert_count"), g("expert_used_count"), float(g("expert_weights_scale"))
        self.window = g("attention.sliding_window")
        self.sliding = [bool(x) for x in g("attention.sliding_window_pattern")]
        self.d_rel, self.extent, self.extent_swa = g("d_rel"), g("rel_extent"), g("rel_extent_swa")
        self.taps = g("shortconv_kernel")
        self.n_floor, self.alpha = g("log_scaling_n_floor"), float(g("log_scaling_alpha"))
        self.qk_clip, self.rel_clip = float(g("qk_clip_threshold")), float(g("rel_clip_threshold"))
        self.block = g("attn_res.block_size")
        self.gate_cap, self.up_cap = float(g("mlp_gate_softcap")), float(g("mlp_up_softcap"))
        self.logit_cap = float(g("final_logit_softcapping"))
        self.n_vocab = g("unpadded_vocab_size")
        assert md.get(a + "value_norm", True) and md.get(a + "attn_res_normalize_values", True)


class Ref:
    def __init__(self, path, dump=None):
        self.g = Gguf(path)
        self.c = Config(self.g.g.metadata)
        self.dump = dump
        self.cache = {}

    def w(self, name):
        if name not in self.cache:
            self.cache[name] = torch.from_numpy(np.array(self.g.tensor(name))).to(DT)
        return self.cache[name]

    def drop(self, prefix):
        """forget a layer's weights once it is done (the real model's dense weights are ~38 GB in float64)"""
        for k in [k for k in self.cache if k.startswith(prefix)]:
            del self.cache[k]

    def has(self, name):
        return name in self.g._by_name

    def cb(self, name, t):
        if self.dump is not None:
            self.dump[name] = t.detach().float().numpy().copy()
        return t

    # ---- the ops
    def rms(self, x, w=None, eps=None):
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + (self.c.eps if eps is None else eps))
        return x if w is None else x * w

    @staticmethod
    def l2clip(x, th):
        return x * (th / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)

    def conv(self, x, w, dyn=None, heads=0):
        """the folded short conv: w [C, taps], tap taps-1 the current token (its residual and scale inside);
        a full layer's dyn [T, heads * taps] by lag (lag 0 the current token), shared by a head's channels"""
        T, C = x.shape
        k = w.shape[1]
        xp = torch.cat([x.new_zeros(k - 1, C), x])
        y = sum(w[:, j] * xp[j:j + T] for j in range(k))
        if dyn is not None:
            d = dyn.view(T, heads, 1, k).expand(T, heads, C // heads, k).reshape(T, C, k)
            y = y + sum(d[:, :, lag] * xp[k - 1 - lag:k - 1 - lag + T] for lag in range(k))
        return y

    def attn_res(self, blocks, x, q):
        s = self.rms(torch.stack(blocks + [x], 1), eps=RES_EPS)   # [T, n, C]
        q = q.reshape(-1, s.shape[-1])                             # [nq, C]
        p = torch.softmax(torch.einsum("qc,tnc->tqn", q, s) / math.sqrt(s.shape[-1]), -1)
        return torch.einsum("tqn,tnc->tqc", p, s).unbind(1)

    def attention(self, L, xq, xk, xv, pos):
        c, b = self.c, f"blk.{L}."
        sliding = c.sliding[L]
        T, H, KV, D = xq.shape[0], c.n_head, c.n_kv, c.hd
        q = xq @ self.w(b + "attn_q.weight").T
        k = xk @ self.w(b + "attn_k.weight").T
        v = xv @ self.w(b + "attn_v.weight").T
        r = xq @ self.w(b + "attn_r.weight").T
        gate = xq @ self.w(b + "attn_gate.weight").T
        sink = xq @ self.w(b + "attn_sink_proj.weight").T                       # [T, H]
        if sliding:
            k, v = self.conv(k, self.w(b + "shortconv_k.weight")), self.conv(v, self.w(b + "shortconv_v.weight"))
        else:
            k = self.conv(k, self.w(b + "shortconv_k.weight"), xk @ self.w(b + "shortconv_k_dyn.weight").T, KV)
            v = self.conv(v, self.w(b + "shortconv_v.weight"), xv @ self.w(b + "shortconv_v_dyn.weight").T, KV)
        self.cb(b + "k_conv", k), self.cb(b + "v_conv", v)
        q = self.l2clip(q.view(T, H, D) * self.w(b + "attn_q_clip.weight"), c.qk_clip)
        k = self.l2clip(k.view(T, KV, D), c.qk_clip)
        v = self.rms(v.view(T, KV, D))
        r = self.l2clip(r.view(T, H, c.d_rel) * self.w(b + "attn_r_clip.weight"), c.rel_clip)
        if not sliding:
            tau = 1 + c.alpha * torch.log(((pos.to(DT) + 1) / c.n_floor).clamp(min=1.0))
            q, r = q * tau[:, None, None], r * tau[:, None, None]
        self.cb(b + "q", q), self.cb(b + "k", k), self.cb(b + "v", v), self.cb(b + "r", r)
        extent = c.extent_swa if sliding else c.extent
        rel = torch.einsum("thr,re->hte", r, self.w(b + "attn_rel_proj.weight"))        # [H, T, extent]
        dist = pos[:, None] - pos[None, :]
        bias = rel.gather(-1, dist.clamp(0, extent - 1).expand(H, T, T))
        bias = bias.masked_fill((dist < 0) | (dist >= extent), 0.0)
        kk, vv = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        logits = torch.einsum("thd,shd->hts", q, kk) / math.sqrt(D) + bias
        ok = (dist >= 0) & ((dist < c.window) if sliding else True)
        logits = logits.masked_fill(~ok, -math.inf)
        p = torch.softmax(torch.cat([logits, sink.T[:, :, None]], -1), -1)[..., :-1]
        o = torch.einsum("hts,shd->thd", p, vv).reshape(T, H * D)
        o = self.cb(b + "attn_gated", o * torch.sigmoid(gate))
        return o @ self.w(b + "attn_output.weight").T

    def moe(self, L, x):
        c, b = self.c, f"blk.{L}."
        scores = torch.sigmoid(x @ self.w(b + "ffn_gate_inp.weight").T)
        pick = (scores + self.w(b + "exp_probs_b.bias")).topk(c.k_exp, -1).indices
        wts = scores.gather(-1, pick)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * c.route_scale
        self.cb(b + "picks", pick.sort(-1).values.to(DT))
        out = torch.zeros_like(x)
        sc = {nm: (self.w(f"{b}ffn_{nm}_exps.scale") if self.has(f"{b}ffn_{nm}_exps.scale") else None)
              for nm in ("gate", "up", "down")}
        for e in pick.unique().tolist():
            tok, j = torch.where(pick == e)
            W = {nm: torch.from_numpy(self.g.expert(f"{b}ffn_{nm}_exps.weight", e)).to(DT) for nm in ("gate", "up", "down")}
            s = {nm: (1.0 if sc[nm] is None else sc[nm][e]) for nm in sc}
            gt = x[tok] @ W["gate"].T * s["gate"]
            up = x[tok] @ W["up"].T * s["up"]
            h = torch.tanh(gt / c.gate_cap) * c.gate_cap * torch.sigmoid(gt) * torch.tanh(up / c.up_cap) * c.up_cap
            out.index_add_(0, tok, (h @ W["down"].T * s["down"]) * wts[tok, j, None])
        return out

    def layer(self, L, h, blocks, pos):
        c, b = self.c, f"blk.{L}."
        res = h
        if L == 0:
            ins = (h, h, h)
        else:
            ins = self.attn_res(blocks, h, self.w(b + "attn_res_score.weight"))
            ins = ins * 3 if len(ins) == 1 else ins
        if c.sliding[L]:
            x = self.rms(ins[0], self.w(b + "attn_norm.weight"))
            a = self.attention(L, x, x, x, pos)
        else:
            a = self.attention(L, self.rms(ins[0], self.w(b + "attn_q_in_norm.weight")),
                               self.rms(ins[1], self.w(b + "attn_k_in_norm.weight")),
                               self.rms(ins[2], self.w(b + "attn_v_in_norm.weight")), pos)
        a = self.cb(b + "attn_out", self.conv(a, self.w(b + "shortconv_attn.weight")))
        if L % c.block == 0:
            blocks = blocks + [res]
            r = a
        else:
            r = res + a
        m_in = self.attn_res(blocks, r, self.w(b + "ffn_res_score.weight"))[0]
        m = self.moe(L, self.rms(m_in, self.w(b + "ffn_norm.weight")))
        m = self.cb(b + "ffn_out", self.conv(m, self.w(b + "shortconv_mlp.weight")))
        return self.cb(b + "out", r + m)

    def forward(self, tokens):
        """the log-probabilities over the real vocabulary, a row a position (row i predicting token i + 1)"""
        c = self.c
        ids = torch.as_tensor(tokens, dtype=torch.long)
        pos = torch.arange(len(ids))
        h = self.cb("embd", self.rms(torch.from_numpy(self.g.rows("token_embd.weight", ids.numpy())).to(DT),
                                    self.w("token_embd_norm.weight")))
        blocks = []
        for L in range(c.n_layers):
            new = self.layer(L, h, blocks, pos)
            self.drop(f"blk.{L}.")
            if L % c.block == 0:
                blocks = blocks + [h]
            h = new
        h = self.attn_res(blocks, h, self.w("output_res_score.weight"))[0]
        h = self.rms(h, self.w("output_norm.weight"))
        logits = h @ self.w("output.weight").T
        logits = torch.tanh(logits / c.logit_cap) * c.logit_cap
        return self.cb("logprobs", torch.log_softmax(logits[:, :c.n_vocab], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tokens", default="1,2,3")
    ap.add_argument("--dump-dir", default="")
    a = ap.parse_args()
    dump = {} if a.dump_dir else None
    r = Ref(a.gguf, dump)
    lp = r.forward([int(t) for t in a.tokens.split(",")])
    print("top-1:", lp.argmax(-1).tolist())
    if dump is not None:
        out = pathlib.Path(a.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k, v in dump.items():
            np.save(out / f"{k}.npy", v)
        print(f"{len(dump)} seams -> {out}")


if __name__ == "__main__":
    main()
