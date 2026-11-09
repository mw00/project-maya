"""tools/strata/glm53_long_gate.py - the long-context quality gate of the glm-dsa (GLM-5.3) port, Maya only (no
llama.cpp reference needed): the SAME tokens scored twice,

    long   after the whole file read through the prompt path (STRATA_GLM_SCORE=c, STRATA_GLM_SCORE_PREFILL=1): the
           DSA indexer selects 2,048 of up to N cells for every scored row (the production read)
    short  with only CONTEXT tokens of the file before them (plus the prompt head), at most 2,048 cells: no selection
           at all, every cell attended (where day 1 matched llama.cpp, KL 0.017)

and compared by the mean NLL of the true next token and the top-1 hits.  A long read should predict its own last
tokens as well as a short context does (it holds the same local context and more); a selection that loses the local
context shows as a long NLL above the short one.  PASS when long - short <= --tol nats.

    python tools/strata/glm53_long_gate.py --pack PACK --tokens FILE [--rows 400] [--context 1200] [--head 20]
        [--tol 0.25] [--exe build/maya] [--gpus 0,1,2] [--max-context N] [--env K=V]... [--out DIR] [--keep]

FILE: comma-separated token ids (a chat prompt over a long document; --head = the template tokens before the
document, kept at the start of the short file).  Each run loads the model (~3 min on 3 V100s) and writes a score dump
of --rows rows (rows x 620 KB, deleted after unless --keep).  Measured 2026-11-07 (UD-IQ2_M, 3x V100): 8K docs -0.06 (PASS), 16K docs+code +0.14, 32K docs+code +1.38 (FAIL; GLM-5.3-Flash on the same file +0.22).
A PASS is not the whole story: the same 8K rows read with the attention sink + the last 2,044 cells instead of the
indexer's choice (~/glm53-work/l15/dsa-experiments.patch, STRATA_GLMD_SINK=4 STRATA_GLMD_LOCAL=2044) score 2.45
against the indexer's 3.35 - see the notes before trusting a long read.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

V = 154880


def read_tokens(path):
    raw = pathlib.Path(path).read_text().replace("\n", ",").split(",")
    return [int(x) for x in raw if x.strip()]


def score(a, name, ids, c, out):
    """one score run: ids through the prompt path to c - 1, rows c.. on the token path; returns (nll, hit) per row"""
    tok = out / f"{name}.tok"
    tok.write_text(",".join(map(str, ids)) + "\n")
    dump = out / f"{name}.f32"
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=a.gpus, STRATA_GLM_SCORE=str(c),
               STRATA_GLM_SCORE_PREFILL="1", STRATA_GLM_SCORE_DUMP=str(dump))
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    ctx = a.max_context or (len(ids) + 512 + 1023) // 1024 * 1024
    t0 = time.time()
    r = subprocess.run([a.exe, "--glm-pack", a.pack, "--tokens-file", str(tok), "--greedy", "--max-new", "1",
                        "--max-context", str(ctx), "--prefill", "auto"], env=env, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("SCORE")), "")
    print(f"  {name}: {len(ids)} tokens, rows {c}..{len(ids) - 2}, {time.time() - t0:.0f} s | {line[:150]}", flush=True)
    if r.returncode != 0 or not dump.exists():
        print("  engine: " + (r.stdout + r.stderr)[-600:].replace("\n", " | "))
        return None
    d = np.memmap(dump, dtype=np.float32, mode="r")
    d = d.reshape(d.size // V, V)
    nll = np.array([-float(d[j][ids[c + 1 + j]]) for j in range(d.shape[0])])
    hit = np.array([int(int(np.argmax(d[j])) == ids[c + 1 + j]) for j in range(d.shape[0])])
    del d
    if not a.keep:
        dump.unlink()
        tok.unlink()
    return nll, hit


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--exe", default=str(pathlib.Path(__file__).resolve().parents[2] / "build" / "maya"))
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--context", type=int, default=1200)
    ap.add_argument("--head", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--max-context", type=int, default=0)
    ap.add_argument("--env", action="append", default=[], help="K=V for both runs (repeatable)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    ids = read_tokens(a.tokens)
    n = len(ids)
    c = n - 1 - a.rows                                  # rows c .. n-2: a.rows of them
    if c - a.context < a.head or a.head + a.context + a.rows > 2048:
        sys.exit(f"{n} tokens: need head + context < c and head + context + rows <= 2048 (no selection in the short run)")
    out = pathlib.Path(a.out or f"/tmp/glm53_long_gate_{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"glm53 long gate: {a.tokens} ({n} tokens), the last {a.rows} rows; short = {a.head} head + {a.context} context")
    lg = score(a, "long", ids, c, out)
    sh = score(a, "short", ids[:a.head] + ids[c - a.context:], a.head + a.context, out)
    if lg is None or sh is None:
        print("FAIL long gate: an engine run failed")
        return 1
    dl, ds = lg[0].mean(), sh[0].mean()
    msg = (f"long NLL {dl:.4f} top-1 {lg[1].mean() * 100:.1f}% | short NLL {ds:.4f} top-1 {sh[1].mean() * 100:.1f}% | "
           f"long - short {dl - ds:+.4f} nats (tol {a.tol})")
    ok = dl - ds <= a.tol
    print(("PASS" if ok else "FAIL") + " long gate: " + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
