"""ref/inkling2.py - Inkling 2's (thinkingmachines/Inkling-2-Small, and the looped Inkling-2-Mini) reference forward
pass over an `inkling2` GGUF, one op at a time: the specification an Inkling 2 engine's kernels are parity-tested
against.

The definition is transformers' own (models/inkling2/modeling_inkling2.py, in its main from 109a7d19bf; the loop:
models/inkling2_loop, transformers PR #51050); the file layout llama.cpp's Inkling 2 conversion, in either of its two
forms: ggml-org/llama.cpp PR #32187's (written by tools/inkling2_gguf.py, its conventions in that file's docstring)
and the one llama.cpp's master reads (unsloth's current files; a file with blk.0.ffn_gate_up_exps.weight). The
second has names of its own, the short convs' residual and output scale outside their weights, gate and up fused
(gate's rows first) under one scale, and the relative projection as [extent][d_rel]; it is folded into the first as
it is read. Checked against transformers end to end: tests/inkling2_ref_test.py - the tiny model's log-probabilities
within fp32's rounding of transformers'.

    python ref/inkling2.py MODEL.gguf [--tokens 1,2,3] [--dump-dir DIR]

Conventions (as ref/glm.py's where they apply): weights from ref/load.py as (out, in), y = W @ x; activations here
(tokens, features) - torch's - with every intermediate a kernel needs recorded by cb() under the op's name
(blk.V.<op>, V the layer visit: the layer itself without a loop), so a kernel's output can be compared with it a seam
at a time. Float64 throughout: this is the spec, not a fast path. Teacher-forced over the whole sequence; decode and
chunked prefill are the engine's.

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
          an expert: g = W_g x * s_g[e], u = W_u x * s_u[e] (the NVFP4 scale; 1 without one - never beside NVFP4),
          W_d (4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25)) * s_d[e]
  at the end: output_norm(attn_res(blocks, h) with output_res_score) -> output -> 30 tanh(l / 30), the first
          unpadded_vocab_size logits
  attn_res(blocks, x): the states [blocks..., x] each RMS-normed (no weight, eps 1e-6), softmax(q . s / sqrt(C)),
          the normed states weighted by it

The loop (inkling2.num_loops passes over the layers loop.core_start .. loop.core_end - 1, whole blocks; Inkling-2-Mini:
3 over layers 6 .. 17 of its 24, so 48 visits a token): the layers before the core once, then the core num_loops
times, then the rest. Each pass starts again from the stream and the blocks the core was first given. A pass after
the first also reads the one before it: its bank = the blocks that pass added after its first, and its final stream.
  messages of core layer i = attn_res(bank, last) with loop_res_query rows 2 i (q's), 2 i + 1 (the MoE's) and
          loop_kv_res_query rows 2 i (k's), 2 i + 1 (v's)
  control(x, m) = (1 - tanh(F g)) x + tanh(G g) m,  g = D [rms(x); rms(m)]   (blk.L.loop_{attn,k,v,mlp}_{down,
          forget,gain}; the norms weightless, eps 1e-6)
  the layer's q, k and v inputs are control(h_in, their message) before the input norms (a sliding layer then norms
          three inputs with its one weight), and the MoE's input control(attn_res(blocks, r), the MoE's message)
          before ffn_norm.
The weights of a core layer are one set; its state (K / V, the convs' histories) is a visit's own.
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

# llama.cpp master's names of PR #32187's tensors
REWRITE = {
    "attn_r.weight": "attn_rel.weight", "attn_r_clip.weight": "attn_rel_clip.weight",
    "attn_rel_proj.weight": "attn_rel_bias.weight", "attn_res_score.weight": "attn_res_query.weight",
    "ffn_res_score.weight": "ffn_res_query.weight", "output_res_score.weight": "output_res_query.weight",
    "attn_q_in_norm.weight": "attn_q_inp_norm.weight", "attn_k_in_norm.weight": "attn_k_inp_norm.weight",
    "attn_v_in_norm.weight": "attn_v_inp_norm.weight", "shortconv_k.weight": "attn_k_conv.weight",
    "shortconv_v.weight": "attn_v_conv.weight", "shortconv_attn.weight": "attn_post_conv.weight",
    "shortconv_mlp.weight": "ffn_post_conv.weight", "shortconv_k_dyn.weight": "attn_k_conv_proj.weight",
    "shortconv_v_dyn.weight": "attn_v_conv_proj.weight",
}


class Config:
    """every number the forward uses, from the GGUF's inkling2.* keys (PR #32187's names, then master's)"""

    def __init__(self, md):
        a = "inkling2."

        def g(*keys):
            for k in keys:
                if a + k in md:
                    return md[a + k]
            raise KeyError(a + keys[0])

        def one(v):   # (master writes a value a visit where every layer has the same)
            if isinstance(v, (list, tuple)):
                assert len(set(v)) == 1, v
                return v[0]
            return v

        self.n_visits, self.n_embd = g("block_count"), g("embedding_length")
        self.n_head, self.n_kv = one(g("attention.head_count")), one(g("attention.head_count_kv"))
        self.hd = g("attention.key_length")
        self.eps = float(g("attention.layer_norm_rms_epsilon"))
        self.n_exp, self.k_exp, self.route_scale = g("expert_count"), g("expert_used_count"), float(g("expert_weights_scale"))
        self.window = g("attention.sliding_window")
        self.d_rel = g("d_rel", "attention.relative_length")
        self.extent = g("rel_extent", "attention.relative_extent")
        self.extent_swa = md.get(a + "rel_extent_swa", self.window)   # (master: a sliding layer's extent is its window)
        self.taps = g("shortconv_kernel", "shortconv.l_cache")
        self.n_floor = g("log_scaling_n_floor", "attention.log_scale_floor")
        self.alpha = float(g("log_scaling_alpha", "attention.log_scale_alpha"))
        self.qk_clip = float(g("qk_clip_threshold", "attention.qk_clip"))
        self.rel_clip = float(g("rel_clip_threshold", "attention.relative_clip"))
        self.block = g("attn_res.block_size")
        self.gate_cap = float(g("mlp_gate_softcap", "activation.situ_beta"))
        self.up_cap = float(g("mlp_up_softcap", "activation.situ_linear_beta"))
        self.logit_cap = float(g("final_logit_softcapping"))
        self.n_vocab = g("unpadded_vocab_size", "vocab_size_unpadded")
        assert md.get(a + "value_norm", True) and md.get(a + "attn_res_normalize_values", True)
        # the loop: the core's layers [core0, core1) run n_pass times; block_count counts the visits
        self.n_pass = md.get(a + "num_loops", 1)
        self.core0, self.core1 = (g("loop.core_start"), g("loop.core_end")) if self.n_pass > 1 else (0, 0)
        n_core = self.core1 - self.core0
        self.n_layers = self.n_visits - (self.n_pass - 1) * n_core
        if self.n_pass > 1:
            assert self.core0 > 0 and self.core0 % self.block == 0 and n_core % self.block == 0, (self.core0, self.core1)
        # the pattern is a visit's; a layer's is its first visit's (the same in every pass)
        pat = [bool(x) for x in g("attention.sliding_window_pattern")]
        assert len(pat) == self.n_visits
        self.sliding = [pat[L if L < self.core1 else L + (self.n_pass - 1) * n_core] for L in range(self.n_layers)]


class Ref:
    def __init__(self, path, dump=None):
        self.g = Gguf(path)
        self.c = Config(self.g.g.metadata)
        self.rw = "blk.0.ffn_gate_up_exps.weight" in self.g._by_name   # llama.cpp master's layout
        self.dump = dump
        self.cache = {}

    def n(self, name):
        """a tensor's name in this file: PR #32187's, or master's of it"""
        if not self.rw:
            return name
        if name.startswith("blk."):
            L, _, leaf = name[4:].partition(".")
            return f"blk.{L}.{REWRITE.get(leaf, leaf)}"
        return REWRITE.get(name, name)

    def w(self, name):
        name = self.n(name)
        if name not in self.cache:
            self.cache[name] = torch.from_numpy(np.array(self.g.tensor(name))).to(DT)
        return self.cache[name]

    def convw(self, name, dynamic=False):
        """a short conv's weights [C, taps] with its residual (the last tap: the current token) and its output scale
        folded in - as PR #32187's layout stores them; master's keeps both outside"""
        w = self.w(name)
        if not self.rw:
            return w
        key = self.n(name) + "#folded"
        if key not in self.cache:
            sc = 1.0 / math.sqrt((2 if dynamic else 1) * w.shape[1] + 1)
            f = w * sc
            f[:, -1] += sc
            self.cache[key] = f
        return self.cache[key]

    def drop(self, prefix):
        """forget a layer's weights once it is done (the real model's dense weights are ~38 GB in float64)"""
        for k in [k for k in self.cache if k.startswith(prefix)]:
            del self.cache[k]

    def has(self, name):
        return self.n(name) in self.g._by_name

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

    def control(self, L, x, msg, module):
        """the loop's message gate of a core layer's input: module attn (q's), k, v or mlp"""
        b = f"blk.{L}.loop_{module}_"
        g = torch.cat([self.rms(x, eps=RES_EPS), self.rms(msg, eps=RES_EPS)], -1) @ self.w(b + "down.weight").T
        forget = torch.tanh(g @ self.w(b + "forget.weight").T)
        gain = torch.tanh(g @ self.w(b + "gain.weight").T)
        return (1 - forget) * x + gain * msg

    def attention(self, L, V, xq, xk, xv, pos):
        c, b, s = self.c, f"blk.{L}.", f"blk.{V}."
        sliding = c.sliding[L]
        T, H, KV, D = xq.shape[0], c.n_head, c.n_kv, c.hd
        q = xq @ self.w(b + "attn_q.weight").T
        k = xk @ self.w(b + "attn_k.weight").T
        v = xv @ self.w(b + "attn_v.weight").T
        r = xq @ self.w(b + "attn_r.weight").T
        gate = xq @ self.w(b + "attn_gate.weight").T
        sink = xq @ self.w(b + "attn_sink_proj.weight").T                       # [T, H]
        if sliding:
            k, v = self.conv(k, self.convw(b + "shortconv_k.weight")), self.conv(v, self.convw(b + "shortconv_v.weight"))
        else:
            dsc = 1.0 / math.sqrt(2 * c.taps + 1) if self.rw else 1.0           # (master: the dynamic taps unscaled)
            k = self.conv(k, self.convw(b + "shortconv_k.weight", True), xk @ self.w(b + "shortconv_k_dyn.weight").T * dsc, KV)
            v = self.conv(v, self.convw(b + "shortconv_v.weight", True), xv @ self.w(b + "shortconv_v_dyn.weight").T * dsc, KV)
        self.cb(s + "k_conv", k), self.cb(s + "v_conv", v)
        q = self.l2clip(q.view(T, H, D) * self.w(b + "attn_q_clip.weight"), c.qk_clip)
        k = self.l2clip(k.view(T, KV, D), c.qk_clip)
        v = self.rms(v.view(T, KV, D))
        r = self.l2clip(r.view(T, H, c.d_rel) * self.w(b + "attn_r_clip.weight"), c.rel_clip)
        if not sliding:
            tau = 1 + c.alpha * torch.log(((pos.to(DT) + 1) / c.n_floor).clamp(min=1.0))
            q, r = q * tau[:, None, None], r * tau[:, None, None]
        self.cb(s + "q", q), self.cb(s + "k", k), self.cb(s + "v", v), self.cb(s + "r", r)
        extent = c.extent_swa if sliding else c.extent
        proj = self.w(b + "attn_rel_proj.weight")                               # [d_rel, extent]; master's [extent, d_rel]
        rel = torch.einsum("thr,re->hte", r, proj.T if self.rw else proj)       # [H, T, extent]
        dist = pos[:, None] - pos[None, :]
        bias = rel.gather(-1, dist.clamp(0, extent - 1).expand(H, T, T))
        bias = bias.masked_fill((dist < 0) | (dist >= extent), 0.0)
        kk, vv = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        logits = torch.einsum("thd,shd->hts", q, kk) / math.sqrt(D) + bias
        ok = (dist >= 0) & ((dist < c.window) if sliding else True)
        logits = logits.masked_fill(~ok, -math.inf)
        p = torch.softmax(torch.cat([logits, sink.T[:, :, None]], -1), -1)[..., :-1]
        o = torch.einsum("hts,shd->thd", p, vv).reshape(T, H * D)
        o = self.cb(s + "attn_gated", o * torch.sigmoid(gate))
        return o @ self.w(b + "attn_output.weight").T

    def moe(self, L, V, x):
        c, b = self.c, f"blk.{L}."
        scores = torch.sigmoid(x @ self.w(b + "ffn_gate_inp.weight").T)
        pick = (scores + self.w(b + "exp_probs_b.bias")).topk(c.k_exp, -1).indices
        wts = scores.gather(-1, pick)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * c.route_scale
        self.cb(f"blk.{V}.picks", pick.sort(-1).values.to(DT))
        out = torch.zeros_like(x)
        # the experts' tensors by role: master's gate and up are one tensor (gate's rows first) under one scale
        src = {"gate": "gate_up", "up": "gate_up", "down": "down"} if self.rw else {"gate": "gate", "up": "up", "down": "down"}
        sc = {}
        for nm in ("gate", "up", "down"):   # (an absent scale is 1.0, except beside NVFP4: docs/INKLING2.md)
            if self.has(f"{b}ffn_{src[nm]}_exps.scale"):
                sc[nm] = self.w(f"{b}ffn_{src[nm]}_exps.scale")
            else:
                assert self.g.info(f"{b}ffn_{src[nm]}_exps.weight").type_name != "NVFP4", \
                    f"{b}ffn_{src[nm]}_exps: an NVFP4 expert without its .scale"
                sc[nm] = None
        for e in pick.unique().tolist():
            tok, j = torch.where(pick == e)
            if self.rw:
                gu = torch.from_numpy(self.g.expert(f"{b}ffn_gate_up_exps.weight", e)).to(DT)
                F = gu.shape[0] // 2
                W = {"gate": gu[:F], "up": gu[F:], "down": torch.from_numpy(self.g.expert(f"{b}ffn_down_exps.weight", e)).to(DT)}
            else:
                W = {nm: torch.from_numpy(self.g.expert(f"{b}ffn_{nm}_exps.weight", e)).to(DT) for nm in ("gate", "up", "down")}
            s = {nm: (1.0 if sc[nm] is None else sc[nm][e]) for nm in sc}
            gt = x[tok] @ W["gate"].T * s["gate"]
            up = x[tok] @ W["up"].T * s["up"]
            h = torch.tanh(gt / c.gate_cap) * c.gate_cap * torch.sigmoid(gt) * torch.tanh(up / c.up_cap) * c.up_cap
            out.index_add_(0, tok, (h @ W["down"].T * s["down"]) * wts[tok, j, None])
        return out

    def layer(self, L, V, h, blocks, pos, msgs=None):
        """layer L's visit V; msgs: a later pass's messages (q's, k's, v's, the MoE's) of a core layer"""
        c, b, s = self.c, f"blk.{L}.", f"blk.{V}."
        res = h
        if L == 0:
            ins = (h, h, h)
        else:
            ins = self.attn_res(blocks, h, self.w(b + "attn_res_score.weight"))
            ins = ins * 3 if len(ins) == 1 else ins
        if msgs is not None:   # (before the input norms: a sliding layer then norms three inputs)
            ins = tuple(self.control(L, ins[j], msgs[j], ("attn", "k", "v")[j]) for j in range(3))
            self.cb(s + "loop_q_in", ins[0])
        if c.sliding[L]:
            wn = self.w(b + "attn_norm.weight")
            xs = [self.rms(x, wn) for x in ins] if msgs is not None else [self.rms(ins[0], wn)] * 3
        else:
            xs = [self.rms(ins[0], self.w(b + "attn_q_in_norm.weight")), self.rms(ins[1], self.w(b + "attn_k_in_norm.weight")),
                  self.rms(ins[2], self.w(b + "attn_v_in_norm.weight"))]
        a = self.attention(L, V, xs[0], xs[1], xs[2], pos)
        a = self.cb(s + "attn_out", self.conv(a, self.convw(b + "shortconv_attn.weight")))
        if L % c.block == 0:
            blocks = blocks + [res]
            r = a
        else:
            r = res + a
        m_in = self.attn_res(blocks, r, self.w(b + "ffn_res_score.weight"))[0]
        if msgs is not None:
            m_in = self.cb(s + "loop_ffn_in", self.control(L, m_in, msgs[3], "mlp"))
        m = self.moe(L, V, self.rms(m_in, self.w(b + "ffn_norm.weight")))
        m = self.cb(s + "ffn_out", self.conv(m, self.convw(b + "shortconv_mlp.weight")))
        return self.cb(s + "out", r + m)

    def forward(self, tokens):
        """the log-probabilities over the real vocabulary, a row a position (row i predicting token i + 1)"""
        c = self.c
        ids = torch.as_tensor(tokens, dtype=torch.long)
        pos = torch.arange(len(ids))
        h = self.cb("embd", self.rms(torch.from_numpy(self.g.rows("token_embd.weight", ids.numpy())).to(DT),
                                    self.w("token_embd_norm.weight")))
        blocks = []
        visit = [0]

        def run(L, h, blocks, msgs=None, last=True):
            new = self.layer(L, visit[0], h, blocks, pos, msgs)
            visit[0] += 1
            if last:
                self.drop(f"blk.{L}.")
            if L % c.block == 0:
                blocks = blocks + [h]
            return new, blocks

        for L in range(c.core0 if c.n_pass > 1 else c.n_layers):
            h, blocks = run(L, h, blocks)
        if c.n_pass > 1:
            pre_h, pre_blocks = h, blocks
            for p in range(c.n_pass):
                bank = last = None
                if p > 0:   # the pass before: the blocks it added after its first, and its final stream
                    bank, last = blocks[len(pre_blocks) + 1:], h
                    h, blocks = pre_h, pre_blocks
                for L in range(c.core0, c.core1):
                    msgs = None
                    if p > 0:
                        i = L - c.core0
                        m_q, m_mlp = self.attn_res(bank, last, self.w("loop_res_query.weight")[2 * i:2 * i + 2])
                        m_k, m_v = self.attn_res(bank, last, self.w("loop_kv_res_query.weight")[2 * i:2 * i + 2])
                        msgs = (m_q, m_k, m_v, m_mlp)
                    h, blocks = run(L, h, blocks, msgs, last=p == c.n_pass - 1)
            for L in range(c.core1, c.n_layers):
                h, blocks = run(L, h, blocks)
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
