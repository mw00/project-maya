#!/usr/bin/env python3
"""tools/strata/pack_test_gate.py - B7.1 / B13.1: run glm_pack_test on a real pack and gate its logits.

    pack_test_gate.py --bin build/glm_pack_test --pack PACK --tokens 64 --logits out.f32 \\
                      --ids ids.txt --ref-logits ref.bin [--min-top1 0.90] [--max-kl 0.05]
    pack_test_gate.py --bin build/glm_pack_test --pack PACK --tokens 8 --logits out.f32 --informational

glm_pack_test (Maya's tool) only prints each position's argmax and returns 0 whenever the forward returns true, so
NaN or garbage logits pass it. This wrapper runs it (GLM_TEST_IDS=--ids when given), then checks the logits file:

* always: one row per token, every value finite, every row with a nonzero spread (max - min > 1e-3);
* with --ref-logits (llm_logits_dump on the same ids: n_rows x n_vocab float32, row i = the logits after id i):
  top-1 agreement >= --min-top1 and mean KL(ref || engine) over the common vocabulary <= --max-kl (defaults are
  the M1a numbers, BUILD-PLAN section 6; the M1a gate itself uses top-64 + rest KL through MAYA_GLM_SCORE);
* without a reference the run FAILS unless --informational is given (then it is a smoke run: finiteness only).

Exit: 0 pass, 1 fail, 2 usage, 77 glm_pack_test returned 77.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

import numpy as np


def read_ids(path: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", pathlib.Path(path).read_text())]


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def check_logits(got: np.ndarray) -> list[str]:
    """Sanity of the engine's rows: finite, nonzero spread."""
    bad = []
    nonfinite = ~np.isfinite(got)
    if nonfinite.any():
        rows = sorted(set(np.nonzero(nonfinite)[0].tolist()))
        bad.append(f"{int(nonfinite.sum())} non-finite logits in rows {rows[:8]}")
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(got, axis=1) - np.nanmin(got, axis=1)
    flat = np.nonzero(~(spread > 1e-3))[0].tolist()
    if flat:
        bad.append(f"rows without spread (max - min <= 1e-3): {flat[:8]}")
    return bad


def compare(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(top-1 agreement, mean KL(ref || got)) over the common vocabulary of the first len(got) rows."""
    v = min(got.shape[1], ref.shape[1])
    g, r = got[:, :v], ref[: got.shape[0], :v]
    top1 = float((g.argmax(axis=1) == r.argmax(axis=1)).mean())
    lr, lg = log_softmax(r), log_softmax(g)
    kl = float((np.exp(lr) * (lr - lg)).sum(axis=1).mean())
    return top1, kl


def gate(logits: str, n_tok: int, ref: str | None, ref_vocab: int | None, min_top1: float, max_kl: float,
         informational: bool) -> tuple[bool, list[str]]:
    size = os.path.getsize(logits)
    if n_tok <= 0 or size == 0 or size % (4 * n_tok):
        return False, [f"{logits}: {size} bytes is not {n_tok} float32 rows"]
    got = np.fromfile(logits, dtype=np.float32).reshape(n_tok, -1)
    msgs = [f"engine logits: {n_tok} rows x {got.shape[1]} vocab"]
    bad = check_logits(got)
    if ref is None:
        if informational:
            msgs.append("no reference: informational smoke run (finiteness and spread only)")
            return not bad, msgs + bad
        return False, msgs + bad + ["no reference: pass --ref-logits (llm_logits_dump on the same --ids) or "
                                    "--informational for a smoke run"]
    rsize = os.path.getsize(ref)
    rv = ref_vocab or got.shape[1]
    if rsize % (4 * rv):
        return False, msgs + bad + [f"{ref}: {rsize} bytes is not whole rows of {rv} floats (pass --ref-vocab)"]
    refm = np.fromfile(ref, dtype=np.float32).reshape(-1, rv)
    if refm.shape[0] < n_tok:
        return False, msgs + bad + [f"{ref}: {refm.shape[0]} rows < {n_tok} tokens"]
    if bad:
        return False, msgs + bad
    top1, kl = compare(got, refm)
    msgs.append(f"vs reference: top-1 {100 * top1:.2f}% (gate >= {100 * min_top1:.0f}%), "
                f"mean KL {kl:.4f} (gate <= {max_kl})")
    ok = top1 >= min_top1 and kl <= max_kl
    return ok, msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", help="glm_pack_test executable (omit to gate an existing --logits file)")
    ap.add_argument("--pack", help="pack directory")
    ap.add_argument("--tokens", type=int, required=True, help="positions to run / rows in --logits")
    ap.add_argument("--logits", required=True, help="the engine's logits file (written by glm_pack_test)")
    ap.add_argument("--ids", help="teacher-forced ids (GLM_TEST_IDS); at least --tokens of them")
    ap.add_argument("--ref-logits", help="llm_logits_dump output on the same ids")
    ap.add_argument("--ref-vocab", type=int, help="the reference's vocabulary (default: the engine's)")
    ap.add_argument("--min-top1", type=float, default=0.90)
    ap.add_argument("--max-kl", type=float, default=0.05)
    ap.add_argument("--informational", action="store_true", help="allow a run without a reference (smoke only)")
    a = ap.parse_args(argv)
    if a.ref_logits and not a.ids:
        print("pack_test_gate: --ref-logits needs --ids (the reference must score the same tokens)", file=sys.stderr)
        return 2
    if a.ids and len(read_ids(a.ids)) < a.tokens:
        print(f"pack_test_gate: {a.ids} holds fewer than {a.tokens} ids", file=sys.stderr)
        return 2
    if a.bin:
        if not a.pack:
            print("pack_test_gate: --bin needs --pack", file=sys.stderr)
            return 2
        env = dict(os.environ)
        if a.ids:
            env["GLM_TEST_IDS"] = a.ids
        r = subprocess.run([a.bin, a.pack, str(a.tokens), a.logits], env=env)
        if r.returncode == 77:
            return 77
        if r.returncode != 0:
            print(f"pack_test_gate: FAIL glm_pack_test exit {r.returncode}")
            return 1
    try:
        ok, msgs = gate(a.logits, a.tokens, a.ref_logits, a.ref_vocab, a.min_top1, a.max_kl, a.informational)
    except (OSError, ValueError) as e:
        print(f"pack_test_gate: FAIL {e}")
        return 1
    for m in msgs:
        print(f"pack_test_gate: {m}")
    print("pack_test_gate: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
