"""tools/ling_quant/ling_gptq_test.py - error-feedback rounding (tools/maya_quant/gptq.py) against llama.cpp's imatrix
rounding on one Ling MoE layer: each expert's output error on held-out calibration tokens (the last 20% of the
sequences), relative to its BF16 output, for a sample of hot, middling and cold experts.

    python ling_gptq_test.py --bf16 BF16.gguf --stats DIR --layer 20 [--types IQ2_XXS,IQ2_XXS,IQ2_S] [--experts 48]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ling_quantize import GQ, Ggml, LingLayer, QT, Source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--fp8", default=P.FP8, required=P.FP8 is None)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--types", default="IQ2_XXS,IQ2_XXS,IQ2_S")
    ap.add_argument("--experts", type=int, default=48)
    ap.add_argument("--prior", default="16384")
    ap.add_argument("--damp", default="0.1")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--ggml", default=P.GGML, required=P.GGML is None)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    G = Ggml(a.ggml)
    codecs = [GQ.Codec(G.lib, int(QT[t])) for t in a.types.split(",")]
    cfg = json.load(open(pathlib.Path(a.fp8) / "config.json"))["text_config"]
    dev = torch.device("cuda")
    t0 = time.time()
    layer = LingLayer(Source(a.bf16), cfg, pathlib.Path(a.stats), a.layer, dev)
    N = layer.X.shape[0]
    print(f"layer {a.layer}: {N} tokens, routing in {time.time() - t0:.1f} s", flush=True)
    sel_tr = torch.zeros(N, dtype=torch.bool, device=dev)
    sel_tr[: int(N * 0.8) // 2048 * 2048] = True
    sel_ho = ~sel_tr
    cnt = np.array(layer.counts)
    hot = list(np.argsort(-cnt)[: a.experts // 3])
    cold = [e for e in np.argsort(cnt) if cnt[e] > 64][: a.experts // 3]
    rng = np.random.default_rng(0)
    mid = list(rng.choice([e for e in range(layer.E) if e not in hot + cold], a.experts - len(hot) - len(cold),
                          replace=False))
    experts = [int(e) for e in hot + mid + cold]
    print(f"tokens per expert: hot {cnt[hot].min()}-{cnt[hot].max()}, cold {cnt[cold].min()}-{cnt[cold].max()}, "
          f"unused {int((cnt == 0).sum())} of {layer.E}", flush=True)
    pool = cf.ThreadPoolExecutor(a.threads)
    rt = GQ.rtn_experts(layer, experts, codecs, pool)
    r_err = {e: GQ.expert_err(layer, e, rt[e], sel_ho) for e in experts}
    rt = {e: [t.cpu() for t in v] for e, v in rt.items()}
    rmean = np.mean(list(r_err.values()))
    for prior in (float(p) for p in a.prior.split(",")):
        for damp in (float(d) for d in a.damp.split(",")):
            t0 = time.time()
            g_err, gu_err = {}, {}
            for c0 in range(0, len(experts), a.chunk):
                part = GQ.round_experts(layer, experts[c0:c0 + a.chunk], codecs, pool, prior, damp, sel_tr, 0.0)
                for e, v in part.items():
                    r = [t.to(dev) for t in rt[e]]
                    g_err[e] = GQ.expert_err(layer, e, v[3], sel_ho)
                    gu_err[e] = GQ.expert_err(layer, e, (v[3][0], v[3][1], r[2]), sel_ho)
                del part
                torch.cuda.empty_cache()
            rel = np.array([gu_err[e] / r_err[e] for e in experts])
            print(f"prior {prior:6.0f} damp {damp:.3f}: mean rel err imatrix {rmean:.5f} | gptq gate/up + imatrix down "
                  f"{np.mean(list(gu_err.values())):.5f} ({100 * (1 - np.mean(list(gu_err.values())) / rmean):+.1f}%) | "
                  f"gptq all {np.mean(list(g_err.values())):.5f} | per expert gptq/imatrix median {np.median(rel):.3f} "
                  f"worst {rel.max():.3f} | {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
