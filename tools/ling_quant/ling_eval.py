"""tools/ling_quant/ling_eval.py - a Ling-3.2-air GGUF scored against the reference: KL, same top token, perplexity.

The held-out eval sequences run through the model layer by layer exactly as ling_calib.py's reference did (Ling's own
modeling code for KDA, gated MLA and RoPE; the same routed-expert module; bf16 hidden states between layers), but with
every weight taken from a GGUF - a quant, or the BF16 conversion - dequantised by ggml's own code.  So the numbers
measure the quant's weights and nothing else, before any engine runs them.  Each position's next-token distribution
is compared with the reference's top-64 (ref_topk.npz): KL(reference || quant) over those 64 plus the rest of the
mass as one bucket, the share of positions whose most likely token is the same, and the perplexity of the text.

--mix BASE.gguf REGEX: tensors whose GGUF name matches REGEX come from --gguf, the rest from BASE (a sensitivity run:
one layer's experts at a low type, everything else BF16).

    python ling_eval.py --gguf Q.gguf --model DIR --eval eval.npy --ref ref_topk.npz [--seqs 0-31] [--mix BF16.gguf 'blk\\.7\\.ffn_.*_exps']
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import json
import os
import pathlib
import re
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "maya_quant"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, P.GGUF_PY)
import gguf  # noqa: E402
from ling_calib import (StatMoE, attn_half, causal_mask, is_moe, remote, text_config)  # noqa: E402
from ling_map import is_full_attention  # noqa: E402

QT = gguf.GGMLQuantizationType


class _Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


class Weights:
    """GGUF tensors (one file set, or a base with some names from another) dequantised to float32 by ggml."""

    def __init__(self, first: str, lib_path: str, threads: int, mix: tuple[str, str] | None = None):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.pool = cf.ThreadPoolExecutor(threads)
        self.main = self._open(first)
        self.base, self.rx = (self._open(mix[0]), re.compile(mix[1])) if mix else (None, None)
        self._fn = {}

    @staticmethod
    def _open(first):
        paths = sorted(pathlib.Path(first).parent.glob(re.sub(r"-00001-of-", "-*-of-", pathlib.Path(first).name)))
        return {t.name: t for p in paths for t in gguf.GGUFReader(str(p)).tensors}

    def _src(self, name):
        if self.base is not None and not self.rx.search(name):
            return self.base[name]
        return self.main[name]

    def has(self, name):
        return name in self.main or (self.base is not None and name in self.base)

    def type_of(self, name):
        return self._src(name).tensor_type

    def get(self, name: str) -> torch.Tensor:
        t = self._src(name)
        shape = tuple(reversed([int(v) for v in t.shape]))
        raw = np.asarray(t.data)
        if t.tensor_type == QT.F32:
            return torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32).reshape(shape))
        tt = int(t.tensor_type)
        fn = self._fn.get(tt)
        if fn is None:
            tr = self.lib.ggml_get_type_traits(tt).contents
            fn = self._fn[tt] = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tr.to_float)
        n_per_row = int(t.shape[0])
        rows = int(np.prod(shape)) // n_per_row
        rb = int(self.lib.ggml_row_size(tt, n_per_row))
        src = np.ascontiguousarray(raw.view(np.uint8).reshape(rows, rb))
        out = np.empty((rows, n_per_row), dtype=np.float32)
        step = max(1, -(-rows // 256))

        def one(r0):
            r1 = min(rows, r0 + step)
            fn(src[r0:r1].ctypes.data, out[r0:r1].ctypes.data, (r1 - r0) * n_per_row)
        list(self.pool.map(one, range(0, rows, step)))
        return torch.from_numpy(out.reshape(shape))


class CkWeights:
    """The FP8 checkpoint itself, under the GGUF names (ling_map's transforms, fp32): `--gguf fp8` scores the reference
    weights through this evaluation path - the noise floor of the measurement (batching, kernel order), not of a
    quant."""

    def __init__(self, model_dir: str):
        from ling_map import Checkpoint, gguf_tensors
        self.ck = Checkpoint(model_dir)
        self.th = dict(gguf_tensors(self.ck, experts=False))
        self.E = int(self.ck.text["num_experts"])

    def has(self, name):
        return name in self.th or name.endswith("_exps.weight")

    def get(self, name: str) -> torch.Tensor:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
        if m:
            from ling_calib import getd
            L = f"model.layers.{m.group(1)}.mlp.experts."
            dev = torch.device("cuda")
            return torch.stack([getd(self.ck, f"{L}{e}.{m.group(2)}_proj.weight", dev) for e in range(self.E)])
        return self.th[name]().float()


def layer_params(W: Weights, T: dict, il: int) -> dict:
    """The HF parameter names of decoder layer il (or the MTP layer) -> tensors, the inverse of ling_map's transforms."""
    B = f"blk.{il}."
    p = {"input_layernorm.weight": W.get(B + "attn_norm.weight"),
         "post_attention_layernorm.weight": W.get(B + "ffn_norm.weight")}
    if not is_full_attention(T, il):
        A = "attention."
        for c in "qkv":
            p[A + f"{c}_proj.weight"] = W.get(B + f"attn_{c}.weight")
            p[A + f"{c}_conv1d.weight"] = W.get(B + f"ssm_conv1d_{c}.weight")
        p[A + "f_proj.weight"] = W.get(B + "ssm_f_a.weight")
        p[A + "b_proj.weight"] = W.get(B + "ssm_beta.weight")
        p[A + "A_log"] = torch.log(W.get(B + "ssm_a").reshape(-1))
        p[A + "dt_bias"] = W.get(B + "ssm_dt.bias")
        p[A + "g_proj.weight"] = W.get(B + "ssm_g_a.weight")
        p[A + "o_norm.weight"] = W.get(B + "ssm_norm.weight")
        p[A + "o_proj.weight"] = W.get(B + "attn_output.weight")
    else:
        A = "attention."
        p[A + "q_proj.weight"] = W.get(B + "attn_q.weight")
        p[A + "kv_a_proj_with_mqa.weight"] = W.get(B + "attn_kv_a_mqa.weight")
        p[A + "kv_a_layernorm.weight"] = W.get(B + "attn_kv_a_norm.weight")
        kb = W.get(B + "attn_k_b.weight").transpose(1, 2)   # [n_head, nope, kv_lora]
        vb = W.get(B + "attn_v_b.weight")                     # [n_head, v, kv_lora]
        p[A + "kv_b_proj.weight"] = torch.cat([kb, vb], 1).reshape(-1, kb.shape[-1])
        p[A + "g_proj.weight"] = W.get(B + "attn_gate.weight")
        p[A + "dense.weight"] = W.get(B + "attn_output.weight")
    if il < int(T["first_k_dense_replace"]):
        for r in ("gate", "up", "down"):
            p[f"mlp.{r}_proj.weight"] = W.get(B + f"ffn_{r}.weight")
    if W.has(B + "nextn.eh_proj.weight"):
        p["eh_proj.weight"] = W.get(B + "nextn.eh_proj.weight")
        p["enorm.weight"] = W.get(B + "nextn.enorm.weight")
        p["hnorm.weight"] = W.get(B + "nextn.hnorm.weight")
        p["final_layernorm.weight"] = W.get(B + "layer_output_norm.weight")
    return p


class EvalMoE(StatMoE):
    """ling_calib's routed-expert module with its weights from the GGUF (and no statistics)."""

    def __init__(self, W: Weights, cfg, il: int, dev):  # noqa: D107 - StatMoE's fields, filled from the GGUF
        torch.nn.Module.__init__(self)
        from ling_calib import limit
        B = f"blk.{il}."
        self.E, self.k = int(cfg.num_experts), int(cfg.num_experts_per_tok)
        self.n_group, self.topk_group = int(cfg.n_group), int(cfg.topk_group)
        self.scale, self.norm = float(cfg.routed_scaling_factor), bool(cfg.norm_topk_prob)
        self.lim = limit(getattr(cfg, "expert_swiglu_limit_list", None), il)
        self.slim = limit(getattr(cfg, "share_expert_swiglu_limit_list", None), il)
        self.router = W.get(B + "ffn_gate_inp.weight").to(dev)
        self.bias = W.get(B + "exp_probs_b.bias").to(dev)
        self.I = int(cfg.moe_intermediate_size)
        g = W.get(B + "ffn_gate_exps.weight").to(dev, torch.float16)
        u = W.get(B + "ffn_up_exps.weight").to(dev, torch.float16)
        self.wgu = torch.cat([g, u], 1)
        del g, u
        self.wd = W.get(B + "ffn_down_exps.weight").to(dev, torch.float16)
        self.sg = W.get(B + "ffn_gate_shexp.weight").to(dev)
        self.su = W.get(B + "ffn_up_shexp.weight").to(dev)
        self.sd = W.get(B + "ffn_down_shexp.weight").to(dev)
        self.collect = False


def build(W: Weights, cfg, T: dict, model_dir: str, il: int, dev, cls_name="BailingMoeV3DecoderLayer"):
    Layer = remote(model_dir, cls_name)
    with torch.device("meta"):
        layer = Layer(cfg, il)
    moe = is_moe(cfg, il)
    if moe:
        layer.mlp = torch.nn.Identity()
    layer = layer.to_empty(device=dev)
    params = layer_params(W, T, il)
    with torch.no_grad():
        for name, prm in layer.named_parameters():
            w = params.pop(name)
            prm.copy_(w.reshape(prm.shape).to(dev, prm.dtype))
    if params:
        raise SystemExit(f"layer {il}: unused {sorted(params)}")
    layer.eval()
    return layer, (EvalMoE(W, cfg, il, dev) if moe else None)


def score(lg: torch.Tensor, ref_lp: torch.Tensor, ref_ids: torch.Tensor, tgt: torch.Tensor):
    """lg [N, V] fp32 logits; ref top-k log-probs/ids [N, k]; tgt [N] next tokens -> per-position (kl, same, nll)."""
    lq = torch.log_softmax(lg, -1)
    pr = ref_lp.exp()
    q_top = lq.gather(1, ref_ids)
    p_tail = (1 - pr.sum(-1)).clamp(min=0)
    q_tail = (1 - q_top.exp().sum(-1)).clamp(min=1e-12)
    kl = (pr * (ref_lp - q_top)).sum(-1) + torch.where(p_tail > 1e-12, p_tail * (p_tail.clamp(min=1e-12).log() - q_tail.log()),
                                                         torch.zeros_like(p_tail))
    same = (lq.argmax(-1) == ref_ids[:, 0]).float()
    nll = -lq.gather(1, tgt[:, None])[:, 0]
    return kl, same, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", default=P.FP8, required=P.FP8 is None)
    ap.add_argument("--eval", default=P.EVAL)
    ap.add_argument("--ref", default=P.REF)
    ap.add_argument("--seqs", default=None, help="a-b or a,b,c (default: all)")
    ap.add_argument("--ctx", type=int, default=0, help="positions before scoring starts")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mix", nargs=2, default=None, metavar=("BASE", "REGEX"))
    ap.add_argument("--ggml", default=P.GGML, required=P.GGML is None)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--json", default=None, help="append the result as one JSON line here")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    t_all = time.time()
    cfg = text_config(a.model)
    T = json.load(open(pathlib.Path(a.model) / "config.json"))["text_config"]
    W = CkWeights(a.model) if a.gguf == "fp8" else Weights(a.gguf, a.ggml, a.threads, tuple(a.mix) if a.mix else None)
    ev = np.load(a.eval).astype(np.int64)
    ref = np.load(a.ref)
    if a.seqs is None:
        seqs = list(range(ev.shape[0]))
    elif "-" in a.seqs:
        s0, s1 = (int(v) for v in a.seqs.split("-"))
        seqs = list(range(s0, s1 + 1))
    else:
        seqs = [int(v) for v in a.seqs.split(",")]
    S, L = len(seqs), ev.shape[1]
    toks = torch.from_numpy(ev[seqs]).to(dev)
    emb = W.get("token_embd.weight")
    h = emb[torch.from_numpy(ev[seqs])].to(torch.bfloat16).to(dev)   # [S, L, H] bf16, as the reference kept them
    del emb
    Rope = remote(a.model, "BailingMoeV3VLRotaryEmbedding")
    rope = Rope(cfg).to(dev)
    n_layers = int(cfg.num_hidden_layers)
    for il in range(n_layers):
        layer, moe = build(W, cfg, T, a.model, il, dev)
        with torch.no_grad():
            for b0 in range(0, S, a.batch):
                hb = h[b0:b0 + a.batch].float()
                B = hb.shape[0]
                pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
                hb, x = attn_half(layer, hb, causal_mask(B, L, dev), None, pos, rope(hb, pos))
                y = moe(x) if moe is not None else layer.mlp(x)
                h[b0:b0 + a.batch] = (hb + y).to(torch.bfloat16)
        del layer, moe
        torch.cuda.empty_cache()
    norm_w = W.get("output_norm.weight").to(dev)
    lm = W.get("output.weight").to(dev)
    eps = float(cfg.rms_norm_eps)
    kls, sames, nlls, klmax = [], [], [], 0.0
    with torch.no_grad():
        for i, s in enumerate(seqs):
            x = h[i].float()
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_w
            lg = x[a.ctx:L - 1] @ lm.t()
            r_lp = torch.from_numpy(ref["logprob"][s, a.ctx:L - 1]).to(dev).float()
            r_id = torch.from_numpy(ref["ids"][s, a.ctx:L - 1]).to(dev).long()
            kl, same, nll = score(lg, r_lp, r_id, toks[i, a.ctx + 1:L])
            kls.append(kl.cpu())
            sames.append(same.cpu())
            nlls.append(nll.cpu())
    kl, same, nll = torch.cat(kls), torch.cat(sames), torch.cat(nlls)
    res = {"label": a.label or pathlib.Path(a.gguf).name, "mix": a.mix[1] if a.mix else None, "seqs": len(seqs),
           "positions": int(kl.numel()), "kl_mean": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)),
           "kl_max": float(kl.max()), "same_top": float(same.mean()), "ppl": float(nll.mean().exp()),
           "nll": float(nll.mean()), "minutes": (time.time() - t_all) / 60,
           "per_seq": [{"seq": s, "kl": float(k_.mean()), "same": float(s_.mean()), "nll": float(n_.mean())}
                       for s, k_, s_, n_ in zip(seqs, kls, sames, nlls)]}
    print(f"{res['label']}: KL {res['kl_mean']:.5f} (p99 {res['kl_p99']:.4f}, max {res['kl_max']:.3f})  "
          f"same top {100 * res['same_top']:.2f}%  ppl {res['ppl']:.4f}  [{res['positions']} positions, "
          f"{res['minutes']:.1f} min]", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
