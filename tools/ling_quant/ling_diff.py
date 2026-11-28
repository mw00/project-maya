"""tools/ling_quant/ling_diff.py - the reference path (ling_calib: FP8 weights) and the evaluation path (ling_eval: a
GGUF's weights) side by side on a few eval sequences, layer by layer: where do their hidden states part?

    python ling_diff.py --gguf BF16.gguf [--seqs 0,1] [--layers 0-41] [--same-input]

--same-input: each layer of the GGUF path takes the reference path's input (so a layer's own difference shows, not
the sum of the ones before it).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ling_calib import (Checkpoint, StatMoE, attn_half, build_layer, causal_mask, is_moe, remote,  # noqa: E402
                        text_config)
from ling_eval import Weights, build  # noqa: E402


def rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", default=P.FP8, required=P.FP8 is None)
    ap.add_argument("--eval", default=P.EVAL)
    ap.add_argument("--seqs", default="0,1")
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--layers", default="0-41")
    ap.add_argument("--same-input", action="store_true")
    ap.add_argument("--ggml", default=P.GGML, required=P.GGML is None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = text_config(a.model)
    T = json.load(open(pathlib.Path(a.model) / "config.json"))["text_config"]
    ck = Checkpoint(a.model)
    W = Weights(a.gguf, a.ggml, 64)
    seqs = [int(s) for s in a.seqs.split(",")]
    ev = np.load(a.eval)[seqs, :a.len].astype(np.int64)
    B, L = ev.shape
    l0, l1 = (int(v) for v in a.layers.split("-"))
    emb_r = ck.get("model.word_embeddings.weight")
    emb_g = W.get("token_embd.weight")
    print(f"embeddings: rel diff {rel(emb_g, emb_r):.2e}", flush=True)
    hr = emb_r[torch.from_numpy(ev)].to(torch.bfloat16).to(dev)
    hg = emb_g[torch.from_numpy(ev)].to(torch.bfloat16).to(dev)
    rope = remote(a.model, "BailingMoeV3VLRotaryEmbedding")(cfg).to(dev)
    pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
    mask = causal_mask(B, L, dev)
    for il in range(l0, l1 + 1):
        lr = build_layer(ck, cfg, a.model, il, dev)
        mr = StatMoE(ck, cfg, il, dev) if is_moe(cfg, il) else None
        if mr is not None:
            mr.collect = False
        lg, mg = build(W, cfg, T, a.model, il, dev)
        # parameter differences
        pr = dict(lr.named_parameters())
        worst = max(((n, rel(p, pr[n])) for n, p in lg.named_parameters()), key=lambda x: x[1])
        with torch.no_grad():
            if a.same_input:
                hg = hr.clone()
            xr_in, xg_in = hr.float(), hg.float()
            h1r, xr = attn_half(lr, xr_in, mask, None, pos, rope(xr_in, pos))
            h1g, xg = attn_half(lg, xg_in, mask, None, pos, rope(xg_in, pos))
            yr = mr(xr) if mr is not None else lr.mlp(xr)
            yg = mg(xg) if mg is not None else lg.mlp(xg)
            if mr is not None:
                from ling_calib import route
                ir, _ = route(xr.reshape(-1, xr.shape[-1]), mr.router, mr.bias, mr.k, mr.n_group, mr.topk_group,
                              mr.scale, mr.norm)
                ig, _ = route(xg.reshape(-1, xg.shape[-1]), mg.router, mg.bias, mg.k, mg.n_group, mg.topk_group,
                              mg.scale, mg.norm)
                same_route = float((ir.sort(-1)[0] == ig.sort(-1)[0]).all(-1).float().mean())
                moe_w = max(rel(mg.wgu, mr.wgu), rel(mg.wd, mr.wd), rel(mg.router, mr.router), rel(mg.bias, mr.bias))
            else:
                same_route, moe_w = float("nan"), 0.0
            hr = (h1r + yr).to(torch.bfloat16)
            hg = (h1g + yg).to(torch.bfloat16)
        kind = getattr(lr, "attention_layer_type", "?")
        print(f"layer {il:2d} {kind:16s} attn out {rel(h1g - xg_in, h1r - xr_in):.2e}  ffn out {rel(yg, yr):.2e}  "
              f"layer out {rel(hg, hr):.2e}  same experts {100 * same_route:6.2f}%  "
              f"worst param {worst[0]} {worst[1]:.1e}  moe weights {moe_w:.1e}", flush=True)
        del lr, mr, lg, mg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
