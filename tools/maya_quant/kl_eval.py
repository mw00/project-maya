"""tools/maya_quant/kl_eval.py - a pack against the FP8 reference: KL, same-top-token rate, NLL on the eval set.

calib.py's reference() keeps the FP8 model's top-k log-probs of every eval position (ref_topk.npz).  Each chosen
sequence is scored by the engine (STRATA_GLM_SCORE, the one-token decode path) after a context of --ctx tokens, with
the matching rows of the reference in the engine's compact form (STRATA_GLM_SCORE_TOPK).

    python kl_eval.py --pack PACK --eval eval.npy --ref ref_topk.npz [--seqs 0,5,10] [--ctx 64] [--len 1024]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import tempfile

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seqs", default=None, help="eval sequence indices (default: every 3rd)")
    ap.add_argument("--ctx", type=int, default=64, help="tokens of context before scoring starts")
    ap.add_argument("--len", type=int, default=1024, help="tokens per sequence used")
    ap.add_argument("--exe", default=str(pathlib.Path(__file__).resolve().parents[2] / "build" / "maya"))
    ap.add_argument("--max-context", type=int, default=8192)
    a = ap.parse_args()
    ev = np.load(a.eval)
    ref = np.load(a.ref)
    lp, ids = ref["logprob"], ref["ids"]
    seqs = [int(s) for s in a.seqs.split(",")] if a.seqs else list(range(0, ev.shape[0], 3))
    L, c = min(a.len, ev.shape[1]), a.ctx
    tot = {"n": 0, "nll": 0.0, "top1": 0.0, "kl": 0.0, "same": 0.0, "kl_max": 0.0}
    for s in seqs:
        toks = ev[s, :L]
        with tempfile.NamedTemporaryFile(suffix=".topk", delete=False) as f:
            f.write(np.int32(ids.shape[2]).tobytes())
            for i in range(c, L - 1):   # the reference row i is the distribution after token i
                f.write(ids[s, i].astype(np.int32).tobytes())
                f.write(lp[s, i].astype(np.float32).tobytes())
            path = f.name
        env = dict(os.environ, STRATA_GLM_SCORE=str(c), STRATA_GLM_SCORE_TOPK=path, STRATA_GLM_USAGE="0")
        r = subprocess.run([a.exe, "--glm-pack", a.pack, "--max-context", str(a.max_context), "--greedy",
                            "--tokens", ",".join(map(str, toks.tolist()))], capture_output=True, text=True, env=env,
                           cwd=pathlib.Path(a.exe).parents[1])
        os.unlink(path)
        m = re.search(r"SCORE (\d+) tokens: nll ([\d.]+) ppl [\d.]+ top1 ([\d.]+)% \| vs ref: KL ([\d.]+) "
                      r"\(max ([\d.]+)\), same top token ([\d.]+)%", r.stdout)
        if not m:
            raise SystemExit(f"seq {s}: no score\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        n = int(m.group(1))
        nll, top1, kl, klm, same = (float(m.group(i)) for i in range(2, 7))
        print(f"seq {s:2d}: {n} tok  KL {kl:.5f} (max {klm:.3f})  same top {same:.2f}%  nll {nll:.4f}", flush=True)
        tot["n"] += n
        tot["nll"] += nll * n
        tot["top1"] += top1 * n
        tot["kl"] += kl * n
        tot["same"] += same * n
        tot["kl_max"] = max(tot["kl_max"], klm)
    n = max(1, tot["n"])
    print(f"ALL {tot['n']} tok: KL {tot['kl'] / n:.5f} (max {tot['kl_max']:.3f})  same top {tot['same'] / n:.2f}%  "
          f"nll {tot['nll'] / n:.4f} ppl {np.exp(tot['nll'] / n):.3f}  top1 {tot['top1'] / n:.2f}%")


if __name__ == "__main__":
    main()
