"""tools/ling_quant/ling_calib.py - the reference and statistics pass of the Ling-3.2-air quants.

The Ling-3.2-air block-FP8 checkpoint runs layer by layer on one GPU, every weight dequantised exactly (FP8 x its
128 x 128 block scale, in fp32), each layer built from the checkpoint's own modeling code (trust_remote_code:
BailingMoeV3DecoderLayer - KDA, gated MLA, RoPE exactly as inclusionAI ship them) with the routed experts replaced by
a module that also records, per expert (tools/maya_quant/calib.py's statistics, Ling's router):

    n          tokens routed to it
    imat_gu    sum over its tokens of x^2 per input channel (the imatrix of gate/up)
    imat_d     the same for down's input h = swiglu(gate, up)
    sal        sum of routing weight * ||expert output||  (REAP saliency)

Ling's router: sigmoid scores, the expert bias added for the choice only, 8 groups of 64 experts ranked by the sum of
their best two, the best 4 groups kept, the top 8 experts among them; weights normalised and x 2.5.  SwiGLU clamps per
layer (expert_swiglu_limit_list / share_expert_swiglu_limit_list).

The hidden states of every sequence live in two bf16 memmaps between layers, so the pass resumes at the first layer
whose stats file is missing.  After layer 41: the eval sequences' reference (the final norm, lm_head; top-64
log-probs and the logsumexp at every position: ref_topk.npz) - the yardstick every quant is scored against - and
then the MTP layer (42) over all sequences, for its own statistics and its reference (ref_mtp_topk.npz).  Each MoE
layer's input over the calibration sequences is kept (ffn_in_XX.bin, bf16 [n_cal, T, 2560]) for GPTQ.

    python ling_calib.py --model DIR --calib calib.npy --eval eval.npy --out DIR [--layers 0-41] [--batch 16]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "maya_quant"))
from hf_map import Checkpoint  # noqa: E402  (its get() dequantises any block-FP8 tensor)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

P = "model."


def remote(model_dir: str, name: str):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    return get_class_from_dynamic_module(f"modeling_bailing_moe_v3_vl.{name}", model_dir)


def text_config(model_dir: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tc = cfg.text_config
    if getattr(tc, "mrope_section", None) is None:
        tc.mrope_section = getattr(cfg, "mrope_section", None) or [8, 12, 12]
    tc._attn_implementation = "eager"
    return tc


def is_moe(cfg, il: int) -> bool:
    return il >= int(cfg.first_k_dense_replace)


def limit(lst, il):
    if lst is None or not 0 <= il < len(lst):
        return None
    return float(lst[il]) or None


def getd(ck: Checkpoint, name: str, dev) -> torch.Tensor:
    """ck.get() with the dequantisation on the device: FP8 x its 128 x 128 block scale, fp32."""
    t = ck.raw(name)
    if t.dtype != torch.float8_e4m3fn:
        return t.to(dev).float()
    w = t.to(dev).float()
    s = ck.raw(name + "_scale_inv").to(dev).float()
    s = s.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
    return w * s


def load_module(ck: Checkpoint, module: torch.nn.Module, prefix: str, skip: tuple[str, ...] = ()):
    """Copy every parameter of `module` from the checkpoint (prefix + its name), dequantised, reshaped to fit."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.startswith(skip):
                continue
            w = getd(ck, prefix + name, p.device)
            if tuple(w.shape) != tuple(p.shape):
                w = w.reshape(p.shape)
            p.copy_(w.to(p.dtype))


def build_layer(ck: Checkpoint, cfg, model_dir: str, il: int, dev, cls_name="BailingMoeV3DecoderLayer"):
    Layer = remote(model_dir, cls_name)
    with torch.device("meta"):
        layer = Layer(cfg, il)
    if is_moe(cfg, il):
        layer.mlp = torch.nn.Identity()   # replaced by StatMoE
    layer = layer.to_empty(device=dev)
    load_module(ck, layer, f"{P}layers.{il}.")
    layer.eval()
    return layer


class Stat:
    """Per-input-channel sums of squares and a token count (an imatrix entry)."""

    def __init__(self, n_in: int, dev):
        self.ss = torch.zeros(n_in, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, x: torch.Tensor):
        self.ss += x.double().square().sum(0)
        self.n += x.shape[0]


def route(x: torch.Tensor, router: torch.Tensor, bias: torch.Tensor, k: int, n_group: int, topk_group: int,
          scale: float, norm: bool):
    """Ling's router (BailingMoeV3Gate): x fp32 [N, H] -> (idx [N, k], weights [N, k])."""
    scores = F.linear(x.float(), router.float()).sigmoid()
    sel = scores + bias
    N, E = sel.shape
    g = sel.view(N, n_group, E // n_group).topk(2, dim=-1)[0].sum(-1)
    gi = torch.topk(g, topk_group, dim=-1, sorted=False)[1]
    gm = torch.zeros_like(g).scatter_(1, gi, 1.0)
    m = gm.unsqueeze(-1).expand(N, n_group, E // n_group).reshape(N, E)
    idx = torch.topk(sel.masked_fill(m == 0, float("-inf")), k, dim=-1)[1]
    w = scores.gather(1, idx)
    if norm and k > 1:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * scale


class StatMoE(torch.nn.Module):
    def __init__(self, ck: Checkpoint, cfg, il: int, dev):
        super().__init__()
        L = f"{P}layers.{il}.mlp."
        self.E, self.k = int(cfg.num_experts), int(cfg.num_experts_per_tok)
        self.n_group, self.topk_group = int(cfg.n_group), int(cfg.topk_group)
        self.scale, self.norm = float(cfg.routed_scaling_factor), bool(cfg.norm_topk_prob)
        self.lim = limit(getattr(cfg, "expert_swiglu_limit_list", None), il)
        self.slim = limit(getattr(cfg, "share_expert_swiglu_limit_list", None), il)
        self.router = ck.get(L + "gate.weight").to(dev)
        self.bias = ck.get(L + "gate.expert_bias").to(dev)
        H, I = int(cfg.hidden_size), int(cfg.moe_intermediate_size)
        self.I = I
        # experts as fp16 on the device: 512 x 3 x 768 x 2560 x 2 B = 6 GB
        self.wgu = torch.empty(self.E, 2 * I, H, dtype=torch.float16, device=dev)
        self.wd = torch.empty(self.E, H, I, dtype=torch.float16, device=dev)
        for e in range(self.E):
            self.wgu[e, :I].copy_(getd(ck, f"{L}experts.{e}.gate_proj.weight", dev))
            self.wgu[e, I:].copy_(getd(ck, f"{L}experts.{e}.up_proj.weight", dev))
            self.wd[e].copy_(getd(ck, f"{L}experts.{e}.down_proj.weight", dev))
        S = f"{L}shared_experts."
        self.sg = getd(ck, S + "gate_proj.weight", dev)
        self.su = getd(ck, S + "up_proj.weight", dev)
        self.sd = getd(ck, S + "down_proj.weight", dev)
        self.collect = True
        self.n = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.sal = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.imat_gu = torch.zeros(self.E, H, dtype=torch.float64, device=dev)
        self.imat_d = torch.zeros(self.E, I, dtype=torch.float64, device=dev)
        self.sh_gu = Stat(H, dev)
        self.sh_d = Stat(int(self.sd.shape[1]), dev)

    @staticmethod
    def _swiglu(g, u, lim):
        """BailingMoeV3MLP: the clamp comes AFTER the SiLU - min(silu(gate), L) * clamp(up, -L, L) (GLM clamps the gate
        before it; Ling does not)."""
        g = F.silu(g)
        if lim:
            g = g.clamp(max=lim)
            u = u.clamp(min=-lim, max=lim)
        return g * u

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        shape = hs.shape
        x = hs.reshape(-1, shape[-1]).float()
        idx, w = route(x, self.router, self.bias, self.k, self.n_group, self.topk_group, self.scale, self.norm)
        out = torch.zeros_like(x)
        flat = idx.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // self.k
        wk = w.reshape(-1)[order]
        pos = 0
        for e in range(self.E):
            c = counts[e]
            if c == 0:
                continue
            t = tok[pos:pos + c]
            we = wk[pos:pos + c]
            pos += c
            xe = x[t]
            gu = (xe.half() @ self.wgu[e].t()).float()
            h = self._swiglu(gu[:, :self.I], gu[:, self.I:], self.lim)
            y = (h.half() @ self.wd[e].t()).float()
            if self.collect:
                self.n[e] += c
                self.imat_gu[e] += xe.double().square().sum(0)
                self.imat_d[e] += h.double().square().sum(0)
                self.sal[e] += (we.double() * y.double().norm(dim=1)).sum()
            out.index_add_(0, t, y * we[:, None])
        hsh = self._swiglu(F.linear(x, self.sg), F.linear(x, self.su), self.slim)
        out += F.linear(hsh, self.sd)
        if self.collect:
            self.sh_gu.add(x)
            self.sh_d.add(hsh)
        return out.reshape(shape).to(hs.dtype)

    def stats(self):
        return {"n": self.n.cpu(), "sal": self.sal.cpu(), "imat_gu": self.imat_gu.float().cpu(),
                "imat_d": self.imat_d.float().cpu(), "sh_gu": self.sh_gu.ss.float().cpu(),
                "sh_d": self.sh_d.ss.float().cpu(), "sh_n": self.sh_gu.n}


def hook_linear_stats(layer, collect):
    """imatrix sums for every nn.Linear of the layer (attention, KDA, dense FFN, MTP eh_proj), by module name."""
    stats, hooks = {}, []
    for name, m in layer.named_modules():
        if isinstance(m, torch.nn.Linear):
            def h(mod, inp, out, name=name):
                if not collect[0]:
                    return
                x = inp[0].reshape(-1, inp[0].shape[-1])
                s = stats.get(name)
                if s is None:
                    s = stats[name] = Stat(x.shape[-1], x.device)
                s.add(x)
            hooks.append(m.register_forward_hook(h))
    return stats, hooks


def causal_mask(B, T, dev):
    m = torch.full((T, T), float("-inf"), device=dev).triu(1)
    return m[None, None].expand(B, 1, T, T)


def attn_half(layer, h, mask4, mask2, pos, pe):
    """BailingMoeV3DecoderLayer.forward up to the FFN: (residual after attention, the FFN input)."""
    x = layer.input_layernorm(h)
    if getattr(layer, "attention_layer_type", "attention") == "attention":   # the MTP layer: always MLA
        x = layer.attention(hidden_states=x, attention_mask=mask4, position_ids=pos, past_key_values=None,
                            use_cache=False, cache_position=pos[0], position_embeddings=pe)[0]
    else:
        x = layer.attention(hidden_states=x, attention_mask=mask2, past_key_value=None, position_ids=pos,
                            use_cache=False, output_attentions=False)[0]
    h = h + x
    return h, layer.post_attention_layernorm(h)


def save_stats(out, il, lin_stats, moe):
    st = {"layer": il, "linear": {k: {"ss": v.ss.float().cpu(), "n": v.n} for k, v in lin_stats.items()}}
    if moe is not None:
        st["moe"] = moe.stats()
    torch.save(st, out / f"stats_{il:02d}.pt")


def topk_ref(lg: torch.Tensor, topk: int):
    lse = torch.logsumexp(lg, -1)
    v, i = torch.topk(lg, topk, -1)
    return (v - lse[:, None]).cpu().numpy(), i.int().cpu().numpy(), lse.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Ling-3.2-air FP8 checkpoint")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="a-b of the main layers (default: all, resuming)")
    ap.add_argument("--batch", type=int, default=16, help="sequences per batch")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--save-ffn-in", type=int, default=1)
    ap.add_argument("--no-mtp", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = Checkpoint(a.model)
    cfg = text_config(a.model)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib, evals = np.load(a.calib), np.load(a.eval)
    if calib.shape[1] != evals.shape[1]:
        raise SystemExit("calib and eval sequences must share their length")
    seqs = np.concatenate([calib, evals], 0).astype(np.int64)
    n_cal, n_seq, T = calib.shape[0], seqs.shape[0], seqs.shape[1]
    if n_cal % a.batch:
        raise SystemExit(f"--batch {a.batch} must divide the {n_cal} calibration sequences")
    H = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    shape = (n_seq, T, H)
    bufs = [out / "streams_a.bin", out / "streams_b.bin"]
    first, last = 0, n_layers - 1
    if a.layers:
        first, last = (int(v) for v in a.layers.split("-"))
    while first <= last and (out / f"stats_{first:02d}.pt").exists():
        first += 1
    mm = [np.memmap(b, dtype=np.uint16, mode="r+" if b.exists() else "w+", shape=shape) for b in bufs]
    if first == 0:
        emb = ck.get(P + "word_embeddings.weight")
        for s in range(n_seq):
            mm[0][s] = emb[torch.from_numpy(seqs[s])].to(torch.bfloat16).view(torch.uint16).numpy()
        mm[0].flush()
        del emb
    Rope = remote(a.model, "BailingMoeV3VLRotaryEmbedding")
    rope = Rope(cfg).to(dev)
    json.dump({"n_cal": n_cal, "n_eval": n_seq - n_cal, "T": T, "layers": n_layers}, open(out / "meta.json", "w"))
    for il in range(first, last + 1):
        t0 = time.time()
        src, dst = mm[il % 2], mm[(il + 1) % 2]
        layer = build_layer(ck, cfg, a.model, il, dev)
        moe = StatMoE(ck, cfg, il, dev) if is_moe(cfg, il) else None
        ffn_in = None
        if moe is not None and a.save_ffn_in:
            ffn_in = np.memmap(out / f"ffn_in_{il:02d}.bin", dtype=np.uint16, mode="w+", shape=(n_cal, T, H))
        collect = [True]
        lin_stats, hooks = hook_linear_stats(layer, collect)
        t_load = time.time() - t0
        for b0 in range(0, n_seq, a.batch):
            bs = list(range(b0, min(n_seq, b0 + a.batch)))
            B = len(bs)
            # statistics from the calibration sequences only (a batch never straddles: n_cal % batch == 0 is asked)
            collect[0] = bs[-1] < n_cal
            if moe is not None:
                moe.collect = collect[0]
            with torch.no_grad():
                h = torch.from_numpy(np.asarray(src[bs[0]:bs[-1] + 1])).view(torch.bfloat16).to(dev).float()
                pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
                pe = rope(h, pos)
                h, x = attn_half(layer, h, causal_mask(B, T, dev), None, pos, pe)
                if ffn_in is not None and collect[0]:
                    ffn_in[bs[0]:bs[-1] + 1] = x.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
                y = moe(x) if moe is not None else layer.mlp(x)
                h = h + y
                dst[bs[0]:bs[-1] + 1] = h.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
        for hk in hooks:
            hk.remove()
        dst.flush()
        if ffn_in is not None:
            ffn_in.flush()
            del ffn_in
        save_stats(out, il, lin_stats, moe)
        del layer, moe
        torch.cuda.empty_cache()
        print(f"layer {il:2d}: {time.time() - t0:6.1f} s (load {t_load:.1f} s)", flush=True)
    if last == n_layers - 1:
        final = mm[(last + 1) % 2]
        reference(ck, cfg, final, seqs, n_cal, a.topk, out, dev)
        if not a.no_mtp and int(getattr(cfg, "num_nextn_predict_layers", 0)):
            mtp(ck, cfg, a.model, final, seqs, n_cal, a.batch, a.topk, out, dev, rope, a.save_ffn_in)


def final_norm(ck, cfg, dev):
    w = ck.get(P + "norm.weight").to(dev)
    eps = float(cfg.rms_norm_eps)
    return lambda h: h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * w


def reference(ck, cfg, streams, seqs, n_cal, topk, out, dev):
    """The eval sequences' next-token distributions: the final norm, lm_head (fp32)."""
    norm = final_norm(ck, cfg, dev)
    lm = ck.get("lm_head.weight").to(dev)
    vals, idxs, lses, nll = [], [], [], []
    with torch.no_grad():
        for s in range(n_cal, seqs.shape[0]):
            h = torch.from_numpy(np.asarray(streams[s])).view(torch.bfloat16).to(dev).float()
            lg = norm(h) @ lm.t()
            v, i, l = topk_ref(lg, topk)
            vals.append(v)
            idxs.append(i)
            lses.append(l)
            tgt = torch.from_numpy(seqs[s, 1:]).to(dev)
            nll.append(float((torch.logsumexp(lg[:-1], -1) - lg[:-1].gather(1, tgt[:, None])[:, 0]).mean()))
    np.savez(out / "ref_topk.npz", logprob=np.stack(vals), ids=np.stack(idxs), lse=np.stack(lses))
    print(f"reference: {seqs.shape[0] - n_cal} sequences x {seqs.shape[1]} positions, top-{topk}; "
          f"perplexity {float(np.exp(np.mean(nll))):.4f}", flush=True)


def mtp(ck, cfg, model_dir, streams, seqs, n_cal, batch, topk, out, dev, rope, save_ffn_in):
    """The NextN layer (42) over every sequence: its statistics (calibration) and its reference (eval)."""
    il = int(cfg.num_hidden_layers)
    t0 = time.time()
    layer = build_layer(ck, cfg, model_dir, il, dev, "BailingMoeV3MTPLayer")
    moe = StatMoE(ck, cfg, il, dev)
    norm = final_norm(ck, cfg, dev)
    emb = ck.get(P + "word_embeddings.weight")
    lm = ck.get("lm_head.weight").to(dev)
    n_seq, T = seqs.shape
    H = int(cfg.hidden_size)
    ffn_in = np.memmap(out / f"ffn_in_{il:02d}.bin", dtype=np.uint16, mode="w+", shape=(n_cal, T, H)) if save_ffn_in else None
    collect = [True]
    lin_stats, hooks = hook_linear_stats(layer, collect)
    vals, idxs, lses = [], [], []
    with torch.no_grad():
        for b0 in range(0, n_seq, batch):
            bs = list(range(b0, min(n_seq, b0 + batch)))
            B = len(bs)
            collect[0] = bs[-1] < n_cal
            moe.collect = collect[0]
            hfin = norm(torch.from_numpy(np.asarray(streams[bs[0]:bs[-1] + 1])).view(torch.bfloat16).to(dev).float())
            nxt = np.roll(seqs[bs[0]:bs[-1] + 1], -1, axis=1)   # roll_tensor(shifts=-1): the next token's embedding
            e = emb[torch.from_numpy(nxt)].to(dev)
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
            pe = rope(hfin, pos)
            x = layer.eh_proj(torch.cat([layer.enorm(e), layer.hnorm(hfin)], -1))
            h, x = attn_half(layer, x, causal_mask(B, T, dev), None, pos, pe)
            if ffn_in is not None and collect[0]:
                ffn_in[bs[0]:bs[-1] + 1] = x.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            h = layer.final_layernorm(h + moe(x))
            if not collect[0]:
                for i in range(B):
                    v, ix, l = topk_ref(h[i] @ lm.t(), topk)
                    vals.append(v)
                    idxs.append(ix)
                    lses.append(l)
    for hk in hooks:
        hk.remove()
    if ffn_in is not None:
        ffn_in.flush()
    save_stats(out, il, lin_stats, moe)
    np.savez(out / "ref_mtp_topk.npz", logprob=np.stack(vals), ids=np.stack(idxs), lse=np.stack(lses))
    print(f"mtp layer {il}: {time.time() - t0:6.1f} s", flush=True)


if __name__ == "__main__":
    main()
