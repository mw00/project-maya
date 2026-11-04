"""tools/mm3/ref.py - a whole-sequence MiniMax-M3.1-Flash forward in torch over the EXL3 pack (the engine's oracle).

    python tools/mm3/ref.py --model ~/models/MiniMax-M3.1-Flash-EXL3-3.5bpw --text "The capital of France is" \
        [--dump out.npz] [--gen 8]
    python tools/mm3/ref.py --selftest                 # this file's layer against transformers', random weights
    python tools/mm3/ref.py --model ... --check-hf 5   # layer 5 of the pack against transformers' layer

It follows transformers' modeling_minimax_m3_vl.py (MiniMaxM3VLDecoderLayer and its parts) at start_pos 0 - the
prompt path, every position at once - with each layer's weights decoded from EXL3 when the layer runs
(tools/ds4/exl3_torch.py) and freed after it, so one GPU of 24 GB carries the 179 GiB pack:

  - Gemma RMSNorm everywhere: x * rsqrt(mean x^2 + eps) * (1 + w), the layer norms, the final norm, the per-head
    q / k norms (one [128] weight for every head) and the indexer's q / k norms.
  - GQA, 64 query heads on 4 KV heads of 128, NeoX RoPE on the first 64 dims of each head (theta 1e7), after the
    QK norm.
  - MiniMax Sparse Attention on every layer (sparse_attention_freq): an indexer of 4 query heads and one shared key,
    each normed and rotated like q / k; per KV group h and query q, score[t] = iq[q, h] . ik[t] over t <= q, a
    block's score the max over its 128 tokens, the query's own block forced in, the top 16 blocks kept (positions
    below 16 blocks attend everything); then exact causal softmax attention over the kept blocks, scale 1/sqrt(128).
  - Layers 0-2 a dense swigluoai MLP (12288), the rest 128 experts top-4 plus a shared expert: router logits in F32,
    s = sigmoid, chosen by s + e_score_correction_bias, weighted by s over the chosen four renormalised, the routed
    sum times routed_scaling_factor (2.0), plus the shared expert.
  - swigluoai: g = min(g, 7), u = clamp(u, -7, 7), out = g * sigmoid(1.702 g) * (u + 1); w1 = gate, w3 = up,
    w2 = down.

Activations are F32 throughout (the reference keeps BF16 between ops; the EXL3 matrices are exact either way).

--dump writes the ids, the logits of the last --dump-last positions (all of them by default, up to 64), the residual
stream after every layer at those positions (unless --no-hs), and the kept blocks of every layer for the positions
in --dump-blocks, for the engine's parity checks.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exl3_torch as ET  # noqa: E402
from st import SafeTensors  # noqa: E402

LM = "language_model."
PFX = LM + "model."


# ------------------------------------------------------------------ the pieces, each as transformers computes it
def gemma_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def rope_tables(n_rot: int, theta: float, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, n_rot, 2, dtype=torch.float32, device=pos.device) / n_rot))
    fr = pos.float()[:, None] * inv[None, :]
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX rotate-half on the first cos.shape[-1] dims of x [T, heads, D]; the rest pass through."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    h = r // 2
    rot = torch.cat([-xr[..., h:], xr[..., :h]], -1)
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([xr * c + rot * s, xp], -1)


def swigluoai(g: torch.Tensor, u: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    g = g.clamp(max=limit)
    u = u.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(g * alpha) * (u + 1.0)


def msa_select(iq: torch.Tensor, ik: torch.Tensor, pos: torch.Tensor, blk: int, topk: int, local: int) -> torch.Tensor:
    """Kept blocks [H, Tq, topk] (-1 where fewer exist) for queries iq [Tq, H, D] at positions pos [Tq] over the index
    keys ik [Tk, D] of positions 0..Tk-1 (MiniMaxM3VLIndexer.forward)."""
    Tq, H, _ = iq.shape
    Tk = ik.shape[0]
    nblk = -(-Tk // blk)
    sc = torch.einsum("qhd,kd->hqk", iq.float(), ik.float())
    future = torch.arange(Tk, device=iq.device)[None, None, :] > pos[None, :, None]
    sc = sc.masked_fill(future, float("-inf"))
    pad = nblk * blk - Tk
    if pad:
        sc = F.pad(sc, (0, pad), value=float("-inf"))
    bs = sc.view(H, Tq, nblk, blk).amax(-1)
    if local > 0:
        qb = pos // blk
        li = (qb[:, None] - torch.arange(local, device=iq.device)[None, :]).clamp(min=0)
        bs.scatter_(-1, li[None].expand(H, -1, -1), float("inf"))
    k = min(topk, nblk)
    v, ix = bs.topk(k, dim=-1)
    ix = ix.masked_fill(v == float("-inf"), -1)
    return F.pad(ix, (0, topk - k), value=-1) if k < topk else ix


def msa_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kept: torch.Tensor, pos: torch.Tensor, blk: int,
               scale: float) -> torch.Tensor:
    """q [Tq, Hq, D], k / v [Tk, Hkv, D], kept [Hkv, Tq, topk] -> [Tq, Hq, D]: softmax attention of each query head
    over its KV group's kept blocks, causal at the token."""
    Tq, Hq, D = q.shape
    Tk, Hkv, _ = k.shape
    G = Hq // Hkv
    nblk = -(-Tk // blk)
    keep = torch.zeros(Hkv, Tq, nblk + 1, dtype=torch.bool, device=q.device)
    keep.scatter_(-1, kept.masked_fill(kept < 0, nblk), True)
    keep = keep[..., :nblk].repeat_interleave(blk, -1)[..., :Tk]
    keep &= ~(torch.arange(Tk, device=q.device)[None, None, :] > pos[None, :, None])
    qg = q.float().view(Tq, Hkv, G, D).permute(1, 2, 0, 3)                  # [Hkv, G, Tq, D]
    kg = k.float().permute(1, 2, 0)                                         # [Hkv, D, Tk]
    s = (qg @ kg[:, None]) * scale                                          # [Hkv, G, Tq, Tk]
    s = s.masked_fill(~keep[:, None], float("-inf"))
    p = torch.softmax(s, -1)
    o = p @ v.float().permute(1, 0, 2)[:, None]                             # [Hkv, G, Tq, D]
    return o.permute(2, 0, 1, 3).reshape(Tq, Hq, D)


# ------------------------------------------------------------------ the engine's roundings (--emulate)
class Emu:
    """An EXL3 matrix computed as maya-mm3's kernels do: y = H(FP16(H(x * suh)) W') * svh, W' the decoded FP16 values,
    F32 sums; x16 first: x itself rounded to FP16 before (the dense matrices' FP16 input rows)."""

    def __init__(self, inner: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, x16: bool):
        self.inner, self.suh, self.svh, self.x16 = inner, suh.float(), svh.float(), x16

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.x16:
            x = x.half().float()
        H = ET.had(x.device)
        sh = x.shape
        a = ((x * self.suh).reshape(-1, sh[-1] // 128, 128) @ H).reshape(sh).half().float()
        y = a @ self.inner
        return (y.reshape(-1, y.shape[-1] // 128, 128) @ H).reshape(*sh[:-1], -1) * self.svh

    def to(self, dev):
        return Emu(self.inner.to(dev), self.suh.to(dev), self.svh.to(dev), self.x16)


def lin(x: torch.Tensor, w) -> torch.Tensor:
    """x W^T for a decoded weight [n, k], or the emulated product."""
    return w(x) if isinstance(w, Emu) else x @ w.T


def f16(x: torch.Tensor, on: bool) -> torch.Tensor:
    return x.half().float() if on else x


def bf16(x: torch.Tensor, on: bool) -> torch.Tensor:
    return x.bfloat16().float() if on else x


# ------------------------------------------------------------------ the model
class Geo:
    def __init__(self, tc: dict):
        self.dim = tc["hidden_size"]
        self.n_layers = tc["num_hidden_layers"]
        self.n_head = tc["num_attention_heads"]
        self.n_kv = tc["num_key_value_heads"]
        self.hd = tc["head_dim"]
        self.n_rot = tc.get("rotary_dim") or int(self.hd * tc.get("partial_rotary_factor", 1.0))
        self.theta = float(tc["rope_theta"])
        self.eps = float(tc["rms_norm_eps"])
        self.vocab = tc["vocab_size"]
        self.n_exp = tc["num_local_experts"]
        self.k_exp = tc["num_experts_per_tok"]
        self.alpha = float(tc["swiglu_alpha"])
        self.limit = float(tc["swiglu_limit"])
        self.scale = float(tc["routed_scaling_factor"])
        self.moe = [bool(f) for f in tc["moe_layer_freq"]]
        sa = tc.get("sparse_attention_config") or {}
        self.sparse = [bool(f) for f in sa.get("sparse_attention_freq", [0] * self.n_layers)]
        self.ix_heads = sa.get("sparse_num_index_heads", 4)
        self.ix_dim = sa.get("sparse_index_dim", 128)
        self.blk = sa.get("sparse_block_size", 128)
        self.topk = sa.get("sparse_topk_blocks", 16)
        self.local = sa.get("sparse_local_block", 1)
        if sa.get("sparse_init_block", 0):
            raise SystemExit("sparse_init_block > 0 is not covered (transformers and vLLM force none)")
        if any(not d for d, s in zip(sa.get("sparse_disable_index_value", [1] * self.n_layers), self.sparse) if s):
            raise SystemExit("an index-value branch (sparse_disable_index_value 0) is not covered")


class Ref:
    def __init__(self, model: pathlib.Path, dev: str = "cuda:0", emulate: bool = False):
        self.dir = model
        self.emulate = emulate
        cfg = json.loads((model / "config.json").read_text())
        self.tc = cfg.get("text_config", cfg)
        self.g = Geo(self.tc)
        self.g.emulate = emulate
        self.st = SafeTensors(model)
        self.dev = dev

    def T(self, name: str) -> torch.Tensor:
        return ET.tensor(self.st, name, self.dev)

    def W(self, prefix: str, x16: bool = True):
        """A linear weight [n, k] float32: EXL3 decoded (an Emu with --emulate), or the stored float tensor."""
        if ET.is_exl3(self.st, prefix):
            if self.emulate:
                return Emu(ET.decode_inner(self.T(prefix + ".trellis"), ET.codebook(self.st, prefix)),
                           self.T(prefix + ".suh"), self.T(prefix + ".svh"), x16)
            return ET.decode(self.st, prefix, self.dev)
        return self.T(prefix + ".weight").float()

    def layer_weights(self, i: int) -> dict:
        p = f"{PFX}layers.{i}."
        a = p + "self_attn."
        w = {
            "in_norm": self.T(p + "input_layernorm.weight"), "post_norm": self.T(p + "post_attention_layernorm.weight"),
            "q": self.W(a + "q_proj"), "k": self.W(a + "k_proj"), "v": self.W(a + "v_proj"), "o": self.W(a + "o_proj"),
            "q_norm": self.T(a + "q_norm.weight"), "k_norm": self.T(a + "k_norm.weight"),
        }
        if self.g.sparse[i]:
            w.update({"iq": self.W(a + "index_q_proj"), "ik": self.W(a + "index_k_proj"),
                      "iq_norm": self.T(a + "index_q_norm.weight"), "ik_norm": self.T(a + "index_k_norm.weight")})
        if self.g.moe[i]:
            m = p + "block_sparse_moe."
            w.update({"gate": self.T(m + "gate.weight").float(), "bias": self.T(m + "e_score_correction_bias").float(),
                      "sh_g": self.W(m + "shared_experts.gate_proj"), "sh_u": self.W(m + "shared_experts.up_proj"),
                      "sh_d": self.W(m + "shared_experts.down_proj")})
        else:
            m = p + "mlp."
            w.update({"mlp_g": self.W(m + "gate_proj"), "mlp_u": self.W(m + "up_proj"), "mlp_d": self.W(m + "down_proj")})
        return w

    def experts(self, i: int, ids: list[int], out_dev: str | None = None) -> dict[int, tuple]:
        """Experts ids of layer i decoded: {e: (w1 gate [ff, dim], w3 up [ff, dim], w2 down [dim, ff])} (on out_dev,
        moved there 16 at a time, when given)."""
        m = f"{PFX}layers.{i}.block_sparse_moe.experts."
        out = {}
        if self.emulate:   # (the experts' inputs are the F32 rows: no FP16 rounding before their rotation)
            for e in ids:
                ws = tuple(self.W(f"{m}{e}.{n}", x16=False) for n in ("w1", "w3", "w2"))
                out[e] = ws if out_dev is None else tuple(w.to(out_dev) for w in ws)
            return out
        for c in range(0, len(ids), 16):
            grp = ids[c:c + 16]
            w1 = ET.decode_many(self.st, [f"{m}{e}.w1" for e in grp], self.dev)
            w3 = ET.decode_many(self.st, [f"{m}{e}.w3" for e in grp], self.dev)
            w2 = ET.decode_many(self.st, [f"{m}{e}.w2" for e in grp], self.dev)
            for j, e in enumerate(grp):
                out[e] = (w1[j], w3[j], w2[j]) if out_dev is None else (w1[j].to(out_dev), w3[j].to(out_dev),
                                                                         w2[j].to(out_dev))
            del w1, w3, w2
        return out


def attention(g: Geo, w: dict, x: torch.Tensor, pos: torch.Tensor, sparse: bool, chunk: int = 256,
              kept_out: list | None = None) -> torch.Tensor:
    """The attention block (input norm already applied): x [T, dim] -> o_proj output [T, dim]."""
    T = x.shape[0]
    em = getattr(g, "emulate", False)
    q = lin(x, w["q"]).view(T, g.n_head, g.hd)
    k = lin(x, w["k"]).view(T, g.n_kv, g.hd)
    v = lin(x, w["v"]).view(T, g.n_kv, g.hd)
    q = gemma_rms(q, w["q_norm"], g.eps)
    k = gemma_rms(k, w["k_norm"], g.eps)
    cos, sin = rope_tables(g.n_rot, g.theta, pos)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    k, v = bf16(k, em), bf16(v, em)                     # (the engine's BF16 KV cache)
    if sparse:
        iq = gemma_rms((x @ w["iq"].T).view(T, g.ix_heads, g.ix_dim), w["iq_norm"], g.eps)
        ik = gemma_rms((x @ w["ik"].T).view(T, 1, g.ix_dim), w["ik_norm"], g.eps)
        iq, ik = bf16(rope(iq, cos, sin), em), bf16(rope(ik, cos, sin)[:, 0], em)
    out = torch.empty(T, g.n_head, g.hd, device=x.device)
    scale = 1.0 / math.sqrt(g.hd)
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        # the keys a query can see end at its own position; queries a..b-1 see keys 0..b-1
        if sparse:
            kept = msa_select(iq[a:b], ik[:b], pos[a:b], g.blk, g.topk, g.local)
            if kept_out is not None:
                kept_out.append(kept.cpu())
        else:
            nb = -(-b // g.blk)
            kept = torch.arange(nb, device=x.device).expand(g.n_kv, b - a, nb)
        out[a:b] = msa_attend(q[a:b], k[:b], v[:b], kept, pos[a:b], g.blk, scale)
    return lin(out.view(T, -1), w["o"])


def ffn(ref: Ref | None, g: Geo, i: int, w: dict, x: torch.Tensor, experts=None, route_out: list | None = None):
    """The MLP block (post-attention norm already applied): x [T, dim] -> [T, dim]."""
    if not g.moe[i]:
        em = getattr(g, "emulate", False)
        return lin(f16(swigluoai(lin(x, w["mlp_g"]), lin(x, w["mlp_u"]), g.alpha, g.limit), em), w["mlp_d"])
    logits = x.float() @ w["gate"].T
    s = torch.sigmoid(logits)
    _, top = torch.topk(s + w["bias"], g.k_exp, dim=-1, sorted=False)
    wt = s.gather(1, top)
    wt = wt / wt.sum(-1, keepdim=True)
    if route_out is not None:
        route_out.append((top.cpu(), wt.cpu()))
    used = sorted(set(top.flatten().tolist()))
    y = torch.zeros_like(x)
    for c in range(0, len(used), 16):   # (decoded 16 at a time: a layer's 128 are 29 GB in F32)
        grp = used[c:c + 16]
        ex = experts if experts is not None else ref.experts(i, grp)
        for e in grp:
            tok, slot = torch.where(top == e)
            w1, w3, w2 = ex[e]
            h = lin(f16(swigluoai(lin(x[tok], w1), lin(x[tok], w3), g.alpha, g.limit), getattr(g, "emulate", False)),
                    w2)
            y.index_add_(0, tok, h * wt[tok, slot, None])
        if experts is None:
            del ex
    sh = lin(f16(swigluoai(lin(x, w["sh_g"]), lin(x, w["sh_u"]), g.alpha, g.limit), getattr(g, "emulate", False)),
             w["sh_d"])
    return y * g.scale + sh


def layer(ref: Ref | None, g: Geo, i: int, w: dict, h: torch.Tensor, pos: torch.Tensor, experts=None,
          kept_out: list | None = None, route_out: list | None = None) -> torch.Tensor:
    a = attention(g, w, gemma_rms(h, w["in_norm"], g.eps), pos, g.sparse[i], kept_out=kept_out)
    h = h + a
    return h + ffn(ref, g, i, w, gemma_rms(h, w["post_norm"], g.eps), experts, route_out)


def forward(ref: Ref, ids: list[int], hs_at: torch.Tensor | None = None, blocks_at: list[int] | None = None,
            log=print, n_layers: int | None = None) -> tuple[torch.Tensor, list, list]:
    """Logits at every position [T, vocab] (F32), the residual stream after each layer at positions hs_at, and the
    kept blocks [H, len(blocks_at), topk] of every layer."""
    g, dev = ref.g, ref.dev
    T = len(ids)
    emb = ref.st
    rows = np.array(emb.raw(PFX + "embed_tokens.weight").view(np.uint16)).reshape(g.vocab, g.dim)[ids]
    h = torch.from_numpy(rows.astype(np.int32) << 16).view(torch.float32).to(dev)    # BF16 -> F32, exact
    pos = torch.arange(T, device=dev)
    hs, blocks = [], []
    for i in range(g.n_layers if n_layers is None else n_layers):
        t0 = time.time()
        w = ref.layer_weights(i)
        kept: list = []
        h = layer(ref, g, i, w, h, pos, kept_out=kept)
        if hs_at is not None:
            hs.append(h[hs_at].cpu())
        if blocks_at is not None and kept:
            allk = torch.cat(kept, 1)
            blocks.append(allk[:, blocks_at])
        del w
        torch.cuda.empty_cache()
        log(f"layer {i:2d} {time.time() - t0:5.1f}s  |h| {h.norm(dim=-1).mean().item():.1f}")
    if n_layers is not None and n_layers < g.n_layers:
        return torch.zeros(T, 1, device=dev), hs, blocks
    h = gemma_rms(h, ref.T(PFX + "norm.weight"), g.eps)
    head = ref.W(LM + "lm_head")
    logits = torch.cat([lin(h[a:a + 1024], head) for a in range(0, T, 1024)])
    return logits, hs, blocks


# ------------------------------------------------------------------ checks against transformers
def hf_layer(tc: dict, i: int, w: dict, experts: dict | None, dev: str):
    """transformers' MiniMaxM3VLDecoderLayer i carrying the weights w (F32, eager attention)."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDecoderLayer
    cfg = MiniMaxM3VLTextConfig(**{k: v for k, v in tc.items() if k not in ("architectures", "model_type")})
    cfg._attn_implementation = "eager"
    L = MiniMaxM3VLDecoderLayer(cfg, i).to(dev, torch.float32).eval()
    sd = {"input_layernorm.weight": w["in_norm"], "post_attention_layernorm.weight": w["post_norm"],
          "self_attn.q_proj.weight": w["q"], "self_attn.k_proj.weight": w["k"], "self_attn.v_proj.weight": w["v"],
          "self_attn.o_proj.weight": w["o"], "self_attn.q_norm.weight": w["q_norm"],
          "self_attn.k_norm.weight": w["k_norm"]}
    if "iq" in w:
        sd.update({"self_attn.indexer.q_proj.weight": w["iq"], "self_attn.indexer.k_proj.weight": w["ik"],
                   "self_attn.indexer.q_norm.weight": w["iq_norm"], "self_attn.indexer.k_norm.weight": w["ik_norm"]})
    if "gate" in w:
        E = cfg.num_local_experts
        sd.update({"mlp.gate.weight": w["gate"], "mlp.gate.e_score_correction_bias": w["bias"],
                   "mlp.shared_experts.gate_up_proj.weight": torch.cat([w["sh_g"], w["sh_u"]]),
                   "mlp.shared_experts.down_proj.weight": w["sh_d"],
                   "mlp.experts.gate_up_proj": torch.stack([torch.cat([experts[e][0], experts[e][1]]) for e in range(E)]),
                   "mlp.experts.down_proj": torch.stack([experts[e][2] for e in range(E)])})
    else:
        sd.update({"mlp.gate_up_proj.weight": torch.cat([w["mlp_g"], w["mlp_u"]]), "mlp.down_proj.weight": w["mlp_d"]})
    missing, unexpected = L.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing {missing} unexpected {unexpected}")
    return L, cfg


def hf_run(L, cfg, h: torch.Tensor) -> torch.Tensor:
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding
    T = h.shape[0]
    pos = torch.arange(T, device=h.device)[None]
    pe = MiniMaxM3VLRotaryEmbedding(cfg).to(h.device)(h[None], pos)
    with torch.no_grad():
        return L(h[None], position_embeddings=pe, attention_mask=None, position_ids=pos)[0]


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    d = (a - b).abs().max().item()
    rel = ((a - b).norm() / b.norm()).item()
    ok = rel < 1e-4
    print(f"{name}: max |diff| {d:.3e}  rel L2 {rel:.3e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def selftest(dev: str, T: int) -> int:
    """This file's layer against transformers' on a small random MiniMax (the real head layout, fewer and narrower
    everything else), dense-MLP and MoE layers, sparse attention past 16 blocks."""
    torch.manual_seed(0)
    tc = {"hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 2,
          "head_dim": 128, "rotary_dim": 64, "partial_rotary_factor": 0.5, "rope_theta": 1e7, "rms_norm_eps": 1e-6,
          "vocab_size": 512, "intermediate_size": 128, "dense_intermediate_size": 384, "shared_intermediate_size": 128,
          "num_local_experts": 8, "num_experts_per_tok": 4, "n_shared_experts": 1, "scoring_func": "sigmoid",
          "use_routing_bias": True, "routed_scaling_factor": 2.0, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
          "moe_layer_freq": [0, 1], "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
          "sparse_attention_config": {"use_sparse_attention": True, "sparse_index_dim": 128,
                                      "sparse_num_index_heads": 2, "sparse_topk_blocks": 16, "sparse_block_size": 128,
                                      "sparse_score_type": "max", "sparse_init_block": 0, "sparse_local_block": 1,
                                      "sparse_attention_freq": [1, 1], "sparse_disable_index_value": [1, 1]},
          "max_position_embeddings": 65536, "hidden_act": "swigluoai"}
    g = Geo(tc)
    D, hd = g.dim, g.hd

    def r(*s, sc=0.05):
        return torch.randn(*s, device=dev) * sc

    ok = True
    for i in range(2):
        w = {"in_norm": r(D, sc=0.1), "post_norm": r(D, sc=0.1), "q": r(g.n_head * hd, D), "k": r(g.n_kv * hd, D),
             "v": r(g.n_kv * hd, D), "o": r(D, g.n_head * hd), "q_norm": r(hd, sc=0.1), "k_norm": r(hd, sc=0.1),
             "iq": r(g.ix_heads * g.ix_dim, D, sc=0.2), "ik": r(g.ix_dim, D, sc=0.2), "iq_norm": r(g.ix_dim, sc=0.1),
             "ik_norm": r(g.ix_dim, sc=0.1)}
        experts = None
        if g.moe[i]:
            ff = tc["intermediate_size"]
            w.update({"gate": r(g.n_exp, D, sc=0.2), "bias": r(g.n_exp, sc=0.1), "sh_g": r(ff, D, sc=0.1),
                      "sh_u": r(ff, D, sc=0.1), "sh_d": r(D, ff, sc=0.1)})
            experts = {e: (r(ff, D, sc=0.1), r(ff, D, sc=0.1), r(D, ff, sc=0.1)) for e in range(g.n_exp)}
        else:
            ff = tc["dense_intermediate_size"]
            w.update({"mlp_g": r(ff, D, sc=0.1), "mlp_u": r(ff, D, sc=0.1), "mlp_d": r(D, ff, sc=0.1)})
        h = torch.randn(T, D, device=dev)
        pos = torch.arange(T, device=dev)
        kept: list = []
        with torch.no_grad():
            mine = layer(None, g, i, w, h, pos, experts=experts, kept_out=kept)
            L, cfg = hf_layer(tc, i, w, experts, dev)
            theirs = hf_run(L, cfg, h)
            # the selection itself, as unordered sets, against transformers' indexer
            ix = L.self_attn.indexer
            xin = L.input_layernorm(h[None])
            pe_cos, pe_sin = rope_tables(g.n_rot, g.theta, pos)
            hf_kept = ix(xin, (pe_cos[None], pe_sin[None]), None, pos[None])[0]          # [H, T, topk]
        mk = torch.cat(kept, 1)
        same = all(set(mk[h_, q].tolist()) == set(hf_kept[h_, q].cpu().tolist())
                   for h_ in range(g.ix_heads) for q in range(T))
        print(f"layer {i} ({'MoE' if g.moe[i] else 'dense'}), T={T}, {-(-T // 128)} blocks: kept blocks "
              f"{'identical' if same else 'DIFFER'}")
        ok &= same
        ok &= compare(f"layer {i} output", mine, theirs)
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def check_hf(ref: Ref, i: int, T: int, run_dev: str) -> int:
    """Layer i of the pack, this file's forward against transformers', on a random residual stream of T positions
    (the weights decoded on ref.dev, both forwards on run_dev: an MoE layer's 128 experts are 29 GB in F32)."""
    torch.manual_seed(i)
    w = {k: v.to(run_dev) for k, v in ref.layer_weights(i).items()}
    experts = None
    if ref.g.moe[i]:
        experts = ref.experts(i, list(range(ref.g.n_exp)), out_dev=run_dev)
    h = torch.randn(T, ref.g.dim, device=run_dev) * 0.5
    with torch.no_grad():
        mine = layer(ref, ref.g, i, w, h, torch.arange(T, device=run_dev), experts=experts)
        L, cfg = hf_layer(ref.tc, i, w, experts, run_dev)
        theirs = hf_run(L, cfg, h)
    return 0 if compare(f"pack layer {i}, T={T}", mine, theirs) else 1


def tokenize(model: pathlib.Path, text: str) -> list[int]:
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(model / "tokenizer.json")).encode(text, add_special_tokens=False).ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(pathlib.Path("~/models/MiniMax-M3.1-Flash-EXL3-3.5bpw").expanduser()))
    ap.add_argument("--text", help="raw text (tokenized as is, no BOS added)")
    ap.add_argument("--ids", help="comma-separated token ids, or @FILE with one id per line / comma-separated")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to append (a whole forward each)")
    ap.add_argument("--dump", help="write the ids, logits, residual streams and kept blocks (.npz)")
    ap.add_argument("--dump-last", type=int, default=64, help="positions (the last N) whose logits / streams to dump")
    ap.add_argument("--dump-blocks", default="", help="comma-separated positions whose kept blocks to dump")
    ap.add_argument("--no-hs", action="store_true", help="--dump without the per-layer residual streams")
    ap.add_argument("--selftest", action="store_true", help="this file's layer against transformers', random weights")
    ap.add_argument("--check-hf", type=int, help="layer N of the pack against transformers' layer")
    ap.add_argument("--T", type=int, default=2600, help="positions for --selftest / --check-hf (> 2048 is sparse)")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--layers", type=int, help="run only the first N layers (for --dump-blocks / the streams)")
    ap.add_argument("--emulate", action="store_true", help="maya-mm3's roundings: FP16 inputs to the EXL3 products, "
                    "BF16 KV and index caches, FP16 expert activations")
    ap.add_argument("--check-dev", help="the device --check-hf runs both forwards on (default --dev; cpu for MoE)")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.dev, a.T)
    ref = Ref(pathlib.Path(a.model).expanduser(), a.dev, a.emulate)
    if a.check_hf is not None:
        return check_hf(ref, a.check_hf, a.T, a.check_dev or a.dev)
    if a.ids:
        src = pathlib.Path(a.ids[1:]).read_text() if a.ids.startswith("@") else a.ids
        ids = [int(x) for x in src.replace(",", " ").split()]
    elif a.text is not None:
        ids = tokenize(ref.dir, a.text)
    else:
        raise SystemExit("--text or --ids")
    for step in range(a.gen + 1):
        T = len(ids)
        last = min(T, a.dump_last)
        hs_at = torch.arange(T - last, T, device=a.dev)
        blocks_at = [int(x) for x in a.dump_blocks.split(",") if x] or None
        with torch.no_grad():
            logits, hs, blocks = forward(ref, ids, hs_at if a.dump and not a.no_hs else None, blocks_at,
                                         n_layers=a.layers)
        if a.layers is not None and a.layers < ref.g.n_layers:
            break
        nxt = int(logits[-1].argmax())
        print(f"T={T} next {nxt} top5 {logits[-1].topk(5).indices.tolist()}")
        if step < a.gen:
            ids.append(nxt)
    if a.dump:
        out = {"ids": np.array(ids), "logits": logits[-last:].cpu().numpy(), "pos": np.arange(T - last, T)}
        if hs:
            out["hs"] = torch.stack(hs).numpy()                        # [layers, last, dim]
        if blocks:
            out["blocks"] = torch.stack(blocks).numpy()                # [layers, H, len(blocks_at), topk]
            out["blocks_at"] = np.array(blocks_at)
        np.savez(a.dump, **out)
        print("wrote", a.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
