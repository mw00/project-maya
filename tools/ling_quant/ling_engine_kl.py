"""tools/ling_quant/ling_engine_kl.py - a GGUF scored IN maya-ling (PR #1224's engine) against the reference: each eval
sequence goes through the engine's prompt path with LING_LOGITS_OUT set (every position's logits), and is compared
with ling_calib.py's reference top-64 exactly as ling_eval.py compares the weights-only path - so the two numbers side
by side separate what the quant's weights cost from what the engine's arithmetic adds.

    python ling_engine_kl.py --gguf Q.gguf [--seqs 0,4,8,12,16,20,24,28] [--exe build/maya-ling] [-- engine args]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ling_eval import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--exe", default=P.MAYA_LING)
    ap.add_argument("--eval", default=P.EVAL)
    ap.add_argument("--ref", default=P.REF)
    ap.add_argument("--seqs", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--vocab", type=int, default=157184)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("engine", nargs=argparse.REMAINDER, help="-- then extra maya-ling arguments")
    a = ap.parse_args()
    extra = [x for x in a.engine if x != "--"]
    ev = np.load(a.eval)
    ref = np.load(a.ref)
    seqs = [int(s) for s in a.seqs.split(",")]
    kls, sames, nlls = [], [], []
    t_all = time.time()
    with tempfile.TemporaryDirectory() as td:
        for s in seqs:
            ids = ev[s].astype(np.int64)
            L = len(ids)
            fid = pathlib.Path(td) / "ids.txt"
            fid.write_text(",".join(map(str, ids.tolist())))
            flog = pathlib.Path(td) / "logits.f32"
            env = dict(os.environ, LING_LOGITS_OUT=str(flog))
            cmd = [a.exe, "--model", a.gguf, "--prompt-ids", str(fid), "--n-predict", "1", "--temp", "0",
                   "--ctx", str(L + 64)] + extra
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode != 0 or not flog.exists():
                raise SystemExit(f"seq {s}: maya-ling failed\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
            lg = np.fromfile(flog, dtype=np.float32)
            n = lg.size // a.vocab
            if n < L:
                raise SystemExit(f"seq {s}: {n} positions of logits, expected {L}")
            lg = torch.from_numpy(lg[: L * a.vocab].reshape(L, a.vocab)[: L - 1].copy())
            kl, same, nll = score(lg.float(), torch.from_numpy(ref["logprob"][s, : L - 1]).float(),
                                  torch.from_numpy(ref["ids"][s, : L - 1]).long(), torch.from_numpy(ids[1:L]))
            kls.append(kl)
            sames.append(same)
            nlls.append(nll)
            pf = [l for l in r.stderr.splitlines() if l.startswith("prompt:")]
            print(f"seq {s:2d}: KL {float(kl.mean()):.5f}  same top {100 * float(same.mean()):.2f}%  "
                  f"nll {float(nll.mean()):.4f}  ({time.time() - t0:.0f} s; {pf[0] if pf else ''})", flush=True)
            flog.unlink()
    kl, same, nll = torch.cat(kls), torch.cat(sames), torch.cat(nlls)
    res = {"label": a.label or ("maya-ling: " + pathlib.Path(a.gguf).name), "engine": "maya-ling", "seqs": len(seqs),
           "positions": int(kl.numel()), "kl_mean": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)),
           "kl_max": float(kl.max()), "same_top": float(same.mean()), "ppl": float(nll.mean().exp()),
           "nll": float(nll.mean()), "minutes": (time.time() - t_all) / 60, "args": extra}
    print(f"{res['label']}: KL {res['kl_mean']:.5f} (p99 {res['kl_p99']:.4f}, max {res['kl_max']:.3f})  "
          f"same top {100 * res['same_top']:.2f}%  ppl {res['ppl']:.4f}  [{res['positions']} positions]", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
