"""tools/ds4/ref_ernie55.py - a whole-sequence ERNIE-5.5-Flash forward in torch (F32, CPU or GPU) over the GGUF's own
tensors: the oracle maya-ds4's ernie55 path is checked against.  It shares none of the engine's code, nor its pack:
it reads the GGUF shards by the checkpoint's names (model.layers.N.*), each weight decoded when its layer runs.

    python tools/ds4/ref_ernie55.py --model ~/models/ERNIE-5.5-Flash-Preview-Maya-S --ids 1,2,3 [--dump out.npz]
           [--gen 8] [--dev cpu] [--workers 8]

It follows baidu/vLLM-ERNIE (src/vllm_ernie/models/ernie5_5, Apache-2.0) and SGLang's ernie55.py, which implement the
architecture (config.json's Ernie5_5_MoeForCausalLM), at start_pos 0 - every position at once:

  * mHC: hc_mult residual streams; each block's pre / post / comb mixes from the streams' RMS with the epsilon outside
    the square root (1 / (sqrt(mean) + eps)), Sinkhorn on comb, the block's input the pre-weighted sum of the streams
    (this block's own pre - no lagged pre as V4.1 has); the head collapses the streams with mhc_contract's mix.
  * HCA layers (compress_ratios 128): DeepSeek-V4's attention - q per-head RMS normed, a single 512-wide KV head, a
    sliding window of 128 raw rows plus one compressed row per 128 tokens (softmax-gated pooling with the compressor's
    ape), YaRN RoPE at compress_rope_theta on the last 64 dims (interleaved pairs), a learned sink, the inverse RoPE on
    the output - and ERNIE's VHA post-mix over the heads and sigmoid output gate on the grouped wo_a's output.
  * MLA layers (compress_ratios -2): DeepSeek-V2 MLA (q / kv LoRA, 192 + 64 qk dims, 256 v dims, plain RoPE at
    rope_theta, interleaved), a learned sink, VHA, and a sigmoid output gate; the DSA lightning indexer picks
    index_topk of the rows older than the window (all of them while there are no more than that), the window the rest.
  * the FFN: SiTU-GLU (beta * tanh(g / beta) * sigmoid(g) * linear_beta * tanh(u / linear_beta)); layer 0 dense, the
    others a latent MoE - two routers whose sigmoids add, the bias-corrected top-k, the picks renormalised and scaled
    by their own learned factor, the routed experts in moe_latent_size between fc1_latent_proj and latent_norm +
    fc2_latent_proj, the shared expert on the hidden state.
  * the logits: lm_head, then the learned multimax transform.

Weights: BF16 / F32 as stored, the dense FP8 (E4M3 bytes, stored as I8, with a F32 scale a 128 x 128 block) exactly,
the routed experts' low-bit blocks through gguf-py's dequantisers ($GGUF_PY or ~/llama.cpp/gguf-py).  Activations are
F32 throughout (the served model keeps BF16 between ops and FP8 in its caches: small differences are expected).

--dump writes the residual streams after every layer ([T, hc, dim] F32), the logits and the ids for the engine's
parity checks (tools/ds4/compare.py reads it).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gguf_reader import GGUFFile  # noqa: E402

HCA, MLA = 128, -2   # compress_ratios' two layer kinds


def _R():
    """tools/ds4/ref.py (its roundings: --sim-kv), imported when asked for."""
    sys.path.insert(0, str(HERE))
    import ref
    return ref


# ------------------------------------------------------------------ the GGUF's tensors
class Shards:
    """Every tensor of a multi-file GGUF by name, memory-mapped."""

    def __init__(self, model: pathlib.Path):
        files = sorted(model.glob("*.gguf"))
        if not files:
            raise SystemExit("%s: no .gguf files" % model)
        self.t, self.mm, self.meta = {}, [], {}
        for i, p in enumerate(files):
            g = GGUFFile(p)
            if i == 0:
                self.meta = g.metadata
            self.mm.append(np.memmap(p, dtype=np.uint8, mode="r"))
            for ti in g.tensors:
                self.t[ti.name] = (ti.type_name, i, g.data_start + ti.offset, ti.shape)
        arch = self.meta.get("general.architecture")
        if arch != "ernie55":
            raise SystemExit("%s: general.architecture %r, not ernie55" % (model, arch))
        self.cfg = json.loads(self.meta["ernie55.config_json"])
        self.fp8 = set(self.meta.get("ernie55.raw_f8_e4m3", []))

    def has(self, name: str) -> bool:
        return name in self.t

    def raw(self, name: str) -> np.ndarray:
        dt, fi, off, shape = self.t[name]
        n = 1
        for d in shape:
            n *= d
        if dt in ("F32", "I32"):
            nb = 4 * n
        elif dt in ("BF16", "F16"):
            nb = 2 * n
        elif dt == "I8":
            nb = n
        else:
            be, bb = BLOCKS[dt]
            nb = n // be * bb
        return self.mm[fi][off:off + nb]

    def shape(self, name: str) -> tuple:
        return tuple(reversed(self.t[name][3]))   # torch order: outermost first


BLOCKS = {"IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82), "IQ3_XXS": (256, 98), "IQ3_S": (256, 110),
          "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "Q2_K": (256, 84),
          "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_0": (32, 34)}


def gguf_py():
    try:
        import gguf  # noqa: F401
    except ImportError:
        for d in (os.environ.get("GGUF_PY", ""), os.path.expanduser("~/llama.cpp/gguf-py")):
            if d and os.path.isdir(d):
                sys.path.insert(0, d)
        import gguf  # noqa: F401
    import gguf
    from gguf.quants import dequantize
    return gguf, dequantize


def decode(S: Shards, name: str) -> torch.Tensor:
    """A tensor as F32 (CPU), shaped [out][in] for a matrix."""
    dt = S.t[name][0]
    shape = S.shape(name)
    r = np.array(S.raw(name))
    if dt == "F32":
        v = torch.from_numpy(r.view(np.float32).copy())
    elif dt == "BF16":
        v = torch.from_numpy(r.view(np.int16).copy()).view(torch.bfloat16).float()
    elif dt == "F16":
        v = torch.from_numpy(r.view(np.float16).astype(np.float32))
    elif dt == "I8" and name in S.fp8:   # E4M3 bytes; the scale a 128 x 128 block beside it
        v = torch.from_numpy(r.copy()).view(torch.float8_e4m3fn).float().reshape(shape)
        sn = (name[:-len(".weight")] if name.endswith(".weight") else name) + (
            ".weight_scale_inv" if name.endswith(".weight") else "_scale_inv")
        sc = decode(S, sn)
        sc = sc.repeat_interleave(128, 0)[:shape[0]].repeat_interleave(128, 1)[:, :shape[1]]
        return v * sc
    else:
        gguf, dq = gguf_py()
        v = torch.from_numpy(np.ascontiguousarray(dq(r, gguf.GGMLQuantizationType[dt]), dtype=np.float32))
    return v.reshape(shape)


# ------------------------------------------------------------------ the pieces
def rms(x, w, eps):
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """DeepSeek-V4's precompute_freqs_cis: YaRN's ramp when original_seq_len > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corr(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    f = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(f), f)


def rope_pairs(x, fc, inverse=False):
    """Adjacent pairs as complex numbers (interleaved).  x [T, d] or [T, h, d]; fc [T, d/2]."""
    c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    if inverse:
        fc = fc.conj()
    if x.ndim == 3:
        fc = fc[:, None, :]
    return torch.view_as_real(c * fc).flatten(-2)


def rope_neox(x, cos, sin):
    """NeoX halves on x's last dim (the indexer's leading rope dims).  x [T, d] or [T, h, d]; cos / sin [T, d/2]."""
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    if x.ndim == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def hadamard(n):
    """Sylvester's Hadamard matrix of order n (a power of two), orthonormal."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def situ(g, u, beta, lin):
    return beta * torch.tanh(g / beta) * torch.sigmoid(g) * (lin * torch.tanh(u / lin))


def sinked_attn(scores, values, sink):
    """scores [T, H, K] (-inf masked), values [T, K, Dv] or [K, Dv], sink [H] -> [T, H, Dv]."""
    m = torch.maximum(scores.amax(-1), sink[None, :])
    m = torch.maximum(m, torch.full_like(m, -1e30))
    p = torch.exp(scores - m.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink[None, :] - m)
    if values.ndim == 2:
        o = torch.einsum("thk,kd->thd", p, values)
    else:
        o = torch.einsum("thk,tkd->thd", p, values)
    return o / den.unsqueeze(-1)


_PAR = None


def _par_init(model):
    global _PAR
    torch.set_num_threads(1)
    _PAR = Shards(pathlib.Path(model))


def _par_expert(job):
    """(P, xs, w, beta, lin) -> sum of w * expert(xs) rows, in a worker."""
    P, xs, w, beta, lin = job
    xs = torch.from_numpy(xs)
    g = xs @ decode(_PAR, P + "w1.weight").T
    u = xs @ decode(_PAR, P + "w3.weight").T
    return ((situ(g, u, beta, lin) * torch.from_numpy(w)[:, None]) @ decode(_PAR, P + "w2.weight").T).numpy()


class Ernie55:
    def __init__(self, model: pathlib.Path, dev: str = "cpu", max_seq: int = 8192, workers: int = 1):
        self.model = model
        self.S = Shards(model)
        c = self.c = self.S.cfg
        self.dev = dev
        self.dim, self.hc, self.eps, self.hc_eps = c["hidden_size"], c["hc_mult"], c["rms_norm_eps"], c["hc_eps"]
        self.nl = c["num_hidden_layers"]
        self.ratios = c["compress_ratios"][: self.nl]
        self.beta, self.lin = c["activation_situ_beta"], c["activation_situ_linear_beta"]
        rs, rd = c["rope_scaling"], c["qk_rope_head_dim"]
        self.fc_hca = freqs_cis(rd, max_seq, rs["original_max_position_embeddings"], c["compress_rope_theta"],
                                rs["factor"], rs["beta_fast"], rs["beta_slow"]).to(dev)
        mrd = c["hybrid_mla_qk_rope_head_dim"]
        self.fc_mla = freqs_cis(mrd, max_seq, 0, c["rope_theta"], 1.0, 32, 1).to(dev)
        ird = mrd   # the indexer's rope dims (the MLA layers' rope width), NeoX halves at rope_theta
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, ird, 2, dtype=torch.float32) / ird))
        ang = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv)
        self.ix_cos, self.ix_sin = ang.cos().to(dev), ang.sin().to(dev)
        self.had = hadamard(c["index_head_dim"]).to(dev)
        self.workers = workers
        self.pool = None
        self.timing = {}
        self.last_route = {}
        self.verbose = False
        self.sim_kv = False

    def W(self, name):
        return decode(self.S, name).to(self.dev)

    def lin_(self, x, name):   # x @ W.T for a .weight
        return x @ self.W(name + ".weight").T

    # mHC
    def hc_mixes(self, h, B, kind):
        P = B + ("self_attention_hyper_connection." if kind == "attn" else "mlp_hyper_connection.")
        hc = self.hc
        fn = self.W(P + "mapping_proj.weight")                 # stored [in][out] - the checkpoint's own layout
        fn = fn.T if fn.shape[0] == hc * self.dim else fn      # -> [(2 + hc) hc][hc dim]
        base = self.W(P + "bias")
        scale = torch.stack([self.W(P + a).reshape(()) for a in ("alpha_pre", "alpha_post", "alpha_res")])
        xf = h.flatten(1).float()
        mixes = (xf @ fn.T) / (xf.square().mean(-1, keepdim=True).sqrt() + self.eps)
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc])
        post = 2.0 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
        e = self.hc_eps
        comb = comb.softmax(-1) + e
        comb = comb / (comb.sum(-2, keepdim=True) + e)
        for _ in range(self.c["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + e)
            comb = comb / (comb.sum(-2, keepdim=True) + e)
        x = ((pre + e).unsqueeze(-1) * h).sum(1)
        return x, post, comb

    @staticmethod
    def hc_post(y, res, post, comb):
        return post.unsqueeze(-1) * y.unsqueeze(1) + (comb.unsqueeze(-1) * res.unsqueeze(2)).sum(1)

    def vha(self, A, o):
        """o [T, H, d] -> o[h] + sum_g (V U^T)[h, g] o[g]."""
        U, V = self.W(A + "vha_postmix_U"), self.W(A + "vha_postmix_V")   # [H, r]
        return o + torch.einsum("hr,gr,tgd->thd", V, U, o)

    # attention
    def hca(self, li, x):
        c = self.c
        A = "model.layers.%d.self_attn." % li
        T = x.shape[0]
        H, D, rd = c["num_attention_heads"], c["head_dim"], c["qk_rope_head_dim"]
        G, R, win, ratio = c["o_groups"], c["o_lora_rank"], c["sliding_window"], HCA
        fc = self.fc_hca[:T]
        qr = rms(self.lin_(x, A + "linear_q_down_proj"), self.W(A + "q_layernorm.weight"), self.eps)
        gate = self.lin_(qr, A + "gate_proj")                                     # [T, G R]
        q = self.lin_(qr, A + "linear_q_up_proj").view(T, H, D)
        q = rms(q, None, self.eps)                                                # per head, no weight
        q = torch.cat([q[..., :-rd], rope_pairs(q[..., -rd:], fc)], -1)
        kv = rms(self.lin_(x, A + "linear_kv_proj"), self.W(A + "kv_layernorm.weight"), self.eps)
        kv = torch.cat([kv[:, :-rd], rope_pairs(kv[:, -rd:], fc)], -1)
        if self.sim_kv:   # (maya-ds4's window ring: FP8 values, a power-of-two scale a 32)
            kv = _R().fp8_round(kv, 32)
        C = A + "core_attention.compressor."
        nb = T // ratio
        if nb > 0:
            cut = nb * ratio
            kvc = self.lin_(x[:cut], C + "linear_wkv").view(nb, ratio, D)
            sc = self.lin_(x[:cut], C + "linear_wgate").view(nb, ratio, D) + self.W(C + "ape")
            ck = rms((kvc * sc.softmax(1)).sum(1), self.W(C + "norm.weight"), self.eps)
            ck = torch.cat([ck[:, :-rd], rope_pairs(ck[:, -rd:], fc[:cut:ratio])], -1)
            if self.sim_kv:   # (maya-ds4's compressed rows: E2M1 values, an E4M3 scale a 16)
                ck = _R().fp4_round(ck, 16, True)
            keys = torch.cat([kv, ck], 0)
        else:
            keys = kv
        t = torch.arange(T, device=self.dev)
        j = torch.arange(keys.shape[0], device=self.dev)
        raw = j[None, :] < T
        vis = torch.where(raw, (j[None, :] <= t[:, None]) & (j[None, :] > t[:, None] - win),
                          (j[None, :] - T) < ((t[:, None] + 1) // ratio))
        s = torch.einsum("thd,kd->thk", q, keys) * D ** -0.5
        s = s.masked_fill(~vis.unsqueeze(1), float("-inf"))
        o = sinked_attn(s, keys, self.W(A + "core_attention.attn_sink").float())
        o = torch.cat([o[..., :-rd], rope_pairs(o[..., -rd:], fc, inverse=True)], -1)
        o = self.vha(A, o)
        wo_a = self.W(A + "linear_o_group_proj").view(G, R, -1)                  # [G, R, H / G * D]
        z = torch.einsum("tgd,grd->tgr", o.reshape(T, G, -1), wo_a).reshape(T, G * R)
        z = z * torch.sigmoid(gate)
        return self.lin_(z, A + "o_proj")

    def mla(self, li, x):
        c = self.c
        A = "model.layers.%d.self_attn." % li
        T = x.shape[0]
        H = c["hybrid_mla_num_attention_heads"]
        dn, dr, dv = c["hybrid_mla_qk_nope_head_dim"], c["hybrid_mla_qk_rope_head_dim"], c["hybrid_mla_v_head_dim"]
        kvr = c["hybrid_mla_kv_lora_rank"]
        fc = self.fc_mla[:T]
        qc = rms(self.lin_(x, A + "q_a_proj"), self.W(A + "q_a_layernorm.weight"), self.eps)
        gate = self.lin_(qc, A + "gate_proj")                                     # [T, H dv]
        q = self.lin_(qc, A + "q_b_proj").view(T, H, dn + dr)
        q = torch.cat([q[..., :dn], rope_pairs(q[..., dn:], fc)], -1)
        kva = self.lin_(x, A + "kv_a_proj_with_mqa")
        ckv = rms(kva[:, :kvr], self.W(A + "kv_a_layernorm.weight"), self.eps)
        kpe = rope_pairs(kva[:, kvr:], fc)                                         # [T, dr], one head
        kvb = (ckv @ self.W(A + "kv_b_proj.weight").T).view(T, H, dn + dv)
        kn, v = kvb[..., :dn], kvb[..., dn:]
        s = (torch.einsum("thd,shd->ths", q[..., :dn], kn) + torch.einsum("thd,sd->ths", q[..., dn:], kpe))
        s = s * (dn + dr) ** -0.5
        s = s.masked_fill(~self.dsa_mask(li, x, qc, T).unsqueeze(1), float("-inf"))
        sink = self.W(A + "core_attention.softmax_offset").float()
        m = torch.maximum(s.amax(-1), sink[None, :])
        p = torch.exp(s - m.unsqueeze(-1))
        den = p.sum(-1) + torch.exp(sink[None, :] - m)
        o = torch.einsum("ths,shd->thd", p, v) / den.unsqueeze(-1)                 # [T, H, dv]
        o = self.vha(A, o)
        z = o.reshape(T, H * dv) * torch.sigmoid(gate)
        return self.lin_(z, A + "o_proj")

    def dsa_mask(self, li, x, qc, T):
        """[T, T] the rows each query reads: the window's, plus the indexer's top-k of the older ones."""
        c = self.c
        win, topk = c["sliding_window"], c["index_topk"]
        t = torch.arange(T, device=self.dev)
        s_ = t[None, :]
        causal = s_ <= t[:, None]
        if T <= win + topk:   # no row has more older rows than the indexer keeps: every one is read
            return causal
        I = "model.layers.%d.self_attn.core_attention.indexer." % li
        nh, hd, rd = c["index_n_heads"], c["index_head_dim"], c["hybrid_mla_qk_rope_head_dim"]
        q = self.lin_(qc, I + "wq_b").view(T, nh, hd)
        k = F.layer_norm(self.lin_(x, I + "wk"), (hd,), self.W(I + "k_norm.weight"), self.W(I + "k_norm.bias"), 1e-6)
        cos, sin = self.ix_cos[:T], self.ix_sin[:T]
        q = torch.cat([rope_neox(q[..., :rd], cos, sin), q[..., rd:]], -1) @ self.had
        k = torch.cat([rope_neox(k[..., :rd], cos, sin), k[..., rd:]], -1) @ self.had
        w = self.lin_(x, I + "weights_proj") * (hd ** -0.5 * nh ** -0.5)        # [T, nh]
        score = (torch.einsum("thd,sd->ths", q, k).relu() * w.unsqueeze(-1)).sum(1)   # [T, T]
        older = s_ <= t[:, None] - win
        score = score.masked_fill(~older, float("-inf"))
        keep = torch.zeros(T, T, dtype=torch.bool, device=self.dev)
        top = score.topk(topk, dim=-1)
        keep.scatter_(1, top.indices, top.values > float("-inf"))
        return keep | (causal & (s_ > t[:, None] - win))

    # the FFN
    def dense(self, li, x):
        M = "model.layers.%d.mlp." % li
        return self.lin_(situ(self.lin_(x, M + "gate_proj"), self.lin_(x, M + "up_proj"), self.beta, self.lin),
                         M + "down_proj")

    def moe(self, li, x):
        c = self.c
        M = "model.layers.%d.block_sparse_moe." % li
        T = x.shape[0]
        scores = torch.sigmoid(self.lin_(x, M + "gate")) + torch.sigmoid(x @ self.W(M + "gate.weight_1").T)
        idx = (scores + self.W(M + "e_score_correction_bias")).topk(c["num_experts_per_tok"], dim=-1)[1]
        wts = scores.gather(1, idx)
        if c.get("norm_topk_prob", True):
            wts = wts / wts.sum(-1, keepdim=True).clamp_min(1e-20)
        wts = wts * self.W(M + "gate.routed_scaling_factor_param")[idx]
        self.last_route[li] = (idx, wts)
        lat = self.lin_(x, M + "fc1_latent_proj")
        y = torch.zeros(T, lat.shape[1], device=self.dev)
        experts = torch.unique(idx).tolist()
        if self.workers > 1:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_par_init, initargs=(str(self.model),))
            futs = []
            for e in experts:
                rows, slot = torch.where(idx == e)
                futs.append((rows, self.pool.submit(_par_expert, (M + "experts.%d." % e, lat[rows].cpu().numpy(),
                                                                  wts[rows, slot].cpu().numpy(), self.beta,
                                                                  self.lin))))
            for rows, f in futs:
                y.index_add_(0, rows, torch.from_numpy(f.result()).to(self.dev))
        else:
            for e in experts:
                P = M + "experts.%d." % e
                rows, slot = torch.where(idx == e)
                xs = lat[rows]
                h = situ(xs @ self.W(P + "w1.weight").T, xs @ self.W(P + "w3.weight").T, self.beta, self.lin)
                y.index_add_(0, rows, (h * wts[rows, slot, None]) @ self.W(P + "w2.weight").T)
        y = self.lin_(rms(y, self.W(M + "latent_norm.weight"), self.eps), M + "fc2_latent_proj")
        S = M + "shared_experts."
        return y + self.lin_(situ(self.lin_(x, S + "w1"), self.lin_(x, S + "w3"), self.beta, self.lin), S + "w2")

    @torch.inference_mode()
    def forward(self, ids, dump=None):
        T = len(ids)
        emb = self.S.raw("model.embed_tokens.weight").view(np.int16).reshape(-1, self.dim)
        h = torch.from_numpy(np.array(emb[np.asarray(ids)])).view(torch.bfloat16).float().to(self.dev)
        h = h.unsqueeze(1).repeat(1, self.hc, 1)
        for li in range(self.nl):
            t0 = time.time()
            B = "model.layers.%d." % li
            res = h
            x, post, comb = self.hc_mixes(h, B, "attn")
            x = rms(x, self.W(B + "input_layernorm.weight"), self.eps)
            y = self.mla(li, x) if self.ratios[li] == MLA else self.hca(li, x)
            h = self.hc_post(y, res, post, comb)
            res = h
            x, post, comb = self.hc_mixes(h, B, "ffn")
            x = rms(x, self.W(B + "post_attention_layernorm.weight"), self.eps)
            y = self.dense(li, x) if li < self.c["first_k_dense_replace"] else self.moe(li, x)
            h = self.hc_post(y, res, post, comb)
            if dump is not None:
                dump.append(h.cpu().numpy())
            self.timing[li] = time.time() - t0
            if self.verbose:
                print("layer %d/%d %.1f s" % (li + 1, self.nl, self.timing[li]), file=sys.stderr, flush=True)
        # the head: mhc_contract collapses the streams, then the norm, lm_head and multimax
        fn = self.W("model.mhc_contract.hc_head_fn")
        fn = fn.T if fn.shape[0] == self.hc * self.dim else fn                   # -> [hc][hc dim]
        xf = h.flatten(1)
        mixes = (xf @ fn.T) / (xf.square().mean(-1, keepdim=True).sqrt() + self.eps)
        pre = torch.sigmoid(mixes * self.W("model.mhc_contract.hc_head_scale").reshape(()) +
                            self.W("model.mhc_contract.hc_head_base")) + self.hc_eps
        x = rms((pre.unsqueeze(-1) * h).sum(1), self.W("model.norm.weight"), self.eps)
        z = x @ self.W("lm_head.weight").T
        r, t = self.W("model.lm_head.multimax_ranges"), self.W("model.lm_head.multimax_ts")
        return (z + t[0] * (r[0] - z).relu() + t[1] * (z - r[1]).relu() + t[2] * (r[2] - z).relu().square() +
                t[3] * (z - r[3]).relu().square())

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
            self.pool = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the folder of the ERNIE-5.5 GGUF shards")
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to append (a whole forward each)")
    ap.add_argument("--dump", help="write the per-layer residual streams, the logits and the ids (.npz)")
    ap.add_argument("--no-hs", action="store_true", help="--dump without the per-layer residual streams")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--workers", type=int, default=1, help="routed experts in N CPU worker processes")
    ap.add_argument("--verbose", action="store_true", help="each layer's time on stderr")
    ap.add_argument("--sim-kv", action="store_true",
                    help="round the HCA layers' window rows to FP8 and their compressed rows to FP4 as maya-ds4 keeps them")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    r = Ernie55(pathlib.Path(a.model).expanduser(), a.dev, workers=a.workers)
    r.verbose = a.verbose
    r.sim_kv = a.sim_kv
    try:
        print("tokens:", len(ids), ids[:32], flush=True)
        dump = [] if a.dump and not a.no_hs else None
        t0 = time.time()
        logits = r.forward(ids, dump)
        print("forward %.1fs" % (time.time() - t0), flush=True)
        lp = torch.log_softmax(logits.float(), -1)
        if len(ids) > 1:
            tgt = torch.tensor(ids[1:], device=lp.device)
            nll = -lp[:-1].gather(1, tgt[:, None]).squeeze(1)
            print("ppl over %d tokens: %.4f" % (len(ids) - 1, math.exp(nll.mean().item())))
        top = lp[-1].topk(8)
        print("next:", list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])))
        if a.dump:
            np.savez(a.dump, ids=np.array(ids), logits=logits.cpu().numpy(), hs=np.stack(dump) if dump else np.zeros(0))
        out = list(ids)
        eos = int(r.c.get("eos_token_id", 2))
        for _ in range(a.gen):
            nxt = int(lp[-1].argmax())
            out.append(nxt)
            print("gen ids:", out[len(ids):], flush=True)
            if nxt == eos:
                break
            lp = torch.log_softmax(r.forward(out).float(), -1)
        return 0
    finally:
        r.close()


if __name__ == "__main__":
    sys.exit(main())
