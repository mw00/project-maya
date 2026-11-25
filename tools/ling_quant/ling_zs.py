"""tools/ling_quant/ling_zs.py - zero-shot accuracy (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA) of a Ling
GGUF or of the FP8 reference, the requests built as tools/maya_quant/zs_tasks.py builds them (lm-evaluation-harness's
form: one (context, continuation) per choice; acc = the highest summed log-likelihood, acc_norm = per byte), each
request opened with Ling's <|startoftext|>.  The forward pass is ling_eval.py's (Ling's own layer code, the weights
from the GGUF dequantised by ggml, or the FP8 weights in fp32 with --gguf fp8), layer by layer over every request,
the requests sorted by length into right-padded batches (causal: the padding changes none of a request's tokens).

    python ling_zs.py --build --out zs.jsonl [--n 400]                  # the requests (once)
    python ling_zs.py --reqs zs.jsonl --gguf Q.gguf --out logp.jsonl     # their log-likelihoods
    python ling_zs.py --reqs zs.jsonl --score logp.jsonl [--vs ref_logp.jsonl] [--json results.jsonl --label NAME]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "maya_quant"))
sys.path.insert(0, str(HERE))
BOS = 156891


def build(a):
    import zs_tasks as Z
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]   # noqa: E731
    rng = random.Random(a.seed)
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for task, items in Z.load(pathlib.Path(a.data)).items():
            idx = list(range(len(items)))
            rng.shuffle(idx)
            for k in sorted(idx[: a.n]):
                it = items[k]
                for c, (ctx, cont) in enumerate(zip(it["ctx"], it["cont"])):
                    cids = [BOS] + enc(ctx)
                    whole = [BOS] + enc(ctx + cont)
                    if whole[: len(cids)] != cids:   # merged across the boundary: split on the text
                        whole = cids + enc(cont)
                    fo.write(json.dumps({"task": task, "item": k, "choice": c, "gold": it["gold"], "n_ctx": len(cids),
                                         "ids": whole, "cont_bytes": len(cont.encode("utf-8"))}) + "\n")
                    n += 1
    print(f"{a.out}: {n} requests")


def batches(reqs, budget):
    order = sorted(range(len(reqs)), key=lambda i: (len(reqs[i]["ids"]), i))
    out, cur, tb = [], [], 0
    for i in order:
        t = len(reqs[i]["ids"])
        if cur and (len(cur) + 1) * max(tb, t) > budget:
            out.append((cur, tb))
            cur, tb = [], 0
        cur.append(i)
        tb = max(tb, t)
    if cur:
        out.append((cur, tb))
    return out


def run(a):
    from ling_calib import attn_half, causal_mask, remote, text_config
    from ling_eval import CkWeights, Weights, build as build_layer
    dev = torch.device("cuda")
    t_all = time.time()
    cfg = text_config(a.model)
    T = json.load(open(pathlib.Path(a.model) / "config.json"))["text_config"]
    W = CkWeights(a.model) if a.gguf == "fp8" else Weights(a.gguf, a.ggml, a.threads)
    reqs = [json.loads(l) for l in open(a.reqs, encoding="utf-8")]
    bs = batches(reqs, a.tokens)
    emb = W.get("token_embd.weight")
    hs = []
    for idx, Lb in bs:
        ids = torch.zeros(len(idx), Lb, dtype=torch.long)
        for j, i in enumerate(idx):
            r = reqs[i]["ids"]
            ids[j, :len(r)] = torch.tensor(r)
        hs.append(emb[ids].to(torch.bfloat16).to(dev))
    del emb
    rope = remote(a.model, "BailingMoeV3VLRotaryEmbedding")(cfg).to(dev)
    for il in range(int(cfg.num_hidden_layers)):
        layer, moe = build_layer(W, cfg, T, a.model, il, dev)
        with torch.no_grad():
            for b, (idx, Lb) in enumerate(bs):
                h = hs[b].float()
                B = h.shape[0]
                pos = torch.arange(Lb, device=dev).unsqueeze(0).expand(B, Lb)
                h, x = attn_half(layer, h, causal_mask(B, Lb, dev), None, pos, rope(h, pos))
                y = moe(x) if moe is not None else layer.mlp(x)
                hs[b] = (h + y).to(torch.bfloat16)
        del layer, moe
        torch.cuda.empty_cache()
        print(f"layer {il:2d} {time.time() - t_all:6.0f} s", flush=True)
    norm_w = W.get("output_norm.weight").to(dev)
    lm = W.get("output.weight").to(dev)
    eps = float(cfg.rms_norm_eps)
    with open(a.out, "w") as fo, torch.no_grad():
        for b, (idx, Lb) in enumerate(bs):
            x = hs[b].float()
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_w
            for j, i in enumerate(idx):
                r = reqs[i]
                n0, n1 = r["n_ctx"], len(r["ids"])
                lp = torch.log_softmax(x[j, n0 - 1:n1 - 1] @ lm.t(), -1)
                tgt = torch.tensor(r["ids"][n0:n1], device=dev)
                fo.write(json.dumps({"task": r["task"], "item": r["item"], "choice": r["choice"],
                                     "logp": float(lp.gather(1, tgt[:, None]).sum())}) + "\n")
    print(f"{len(reqs)} requests in {(time.time() - t_all) / 60:.1f} min -> {a.out}")


def score(a):
    reqs = [json.loads(l) for l in open(a.reqs, encoding="utf-8")]

    def answers(path):
        lp = {}
        for l in open(path):
            r = json.loads(l)
            lp[(r["task"], r["item"], r["choice"])] = r["logp"]
        per = {}
        for r in reqs:
            k = (r["task"], r["item"])
            v = per.setdefault(k, {"gold": r["gold"], "lp": {}, "by": {}})
            v["lp"][r["choice"]] = lp[(r["task"], r["item"], r["choice"])]
            v["by"][r["choice"]] = r["cont_bytes"]
        out = {}
        for k, v in per.items():
            out[k] = (max(v["lp"], key=v["lp"].get), max(v["lp"], key=lambda c: v["lp"][c] / max(1, v["by"][c])),
                      v["gold"])
        return out

    mine = answers(a.score)
    ref = answers(a.vs) if a.vs else None
    tasks = sorted({k[0] for k in mine})
    res = {}
    for t in tasks:
        ks = [k for k in mine if k[0] == t]
        acc = np.mean([mine[k][0] == mine[k][2] for k in ks])
        norm = np.mean([mine[k][1] == mine[k][2] for k in ks])
        row = {"acc": round(100 * acc, 2), "acc_norm": round(100 * norm, 2), "n": len(ks)}
        if ref:
            row["same_answer"] = round(100 * np.mean([mine[k][1] == ref[k][1] for k in ks]), 2)
        res[t] = row
    main_metric = {t: res[t]["acc_norm"] if t in ("arc_challenge", "arc_easy", "hellaswag", "piqa") else res[t]["acc"]
                   for t in tasks}
    res["mean"] = round(float(np.mean(list(main_metric.values()))), 2)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"label": a.label or a.score, **res}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--data", default=P.EVALDATA, required=P.EVALDATA is None)
    ap.add_argument("--model", default=P.FP8, required=P.FP8 is None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reqs")
    ap.add_argument("--gguf")
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=16384, help="padded tokens a batch")
    ap.add_argument("--ggml", default=P.GGML, required=P.GGML is None)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--score")
    ap.add_argument("--vs", default=None, help="the reference's logp file: the share of answers that are the same")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    if a.build:
        build(a)
    elif a.score:
        score(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
