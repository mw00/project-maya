#!/usr/bin/env python3
"""tools/strata/llama_to_score_ref.py - llama.cpp logits dumps -> the engine's teacher-forced score references (B7.3).

`tools/llm_logits_dump` (built against llama.cpp aa94f208) writes the raw logits of every position of a token list:
n_tokens rows of n_vocab float32, row i = the logits after token i (predicting token i + 1).  The engine scores the
same token list with `MAYA_GLM_SCORE=<c>` (src/program/generate.cpp): positions i = c .. n-2, each against token
i + 1, and compares against one of two reference files:

* `MAYA_GLM_SCORE_TOPK=<file>` (`--format topk`, default): int32 k, then per scored position k int32 token ids
  (best first) and their k float32 log-probabilities.  The engine's KL is over those k tokens plus one bucket for
  the rest of the vocabulary: the "top-64 + rest" KL every BUILD-PLAN gate uses (section 6).
* `MAYA_GLM_SCORE_REF=<file>` (`--format full`): per scored position n_vocab float32 log-softmax values (the
  engine's own `MAYA_GLM_SCORE_DUMP` layout).  `--engine-vocab` pads llama.cpp's vocabulary to the engine's with
  -inf (probability 0) when the two differ.

    llm_logits_dump --ngl 999 --ot exps=CPU --ubatch 512 $M ref.bin @tokens.txt
    python3 tools/strata/llama_to_score_ref.py --dump ref.bin --tokens tokens.txt --context 2048 --out ref.topk
    MAYA_GLM_SCORE=2048 MAYA_GLM_SCORE_TOPK=ref.topk build/strata --glm-pack <pack> --tokens-file tokens.txt

It also prints the reference's own NLL, perplexity and top-1 agreement with the teacher tokens over the scored
positions (100% top-1 on a greedy continuation the reference itself generated).
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

DEFAULT_VOCAB = 154880


def read_tokens(path: str) -> list[int]:
    text = pathlib.Path(path).read_text().replace(",", " ").replace("[", " ").replace("]", " ")
    return [int(x) for x in text.split()]


def log_softmax(row: np.ndarray) -> np.ndarray:
    x = row.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def topk(ls: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, best first (ties: lower id first)."""
    idx = np.argpartition(-ls, k - 1)[:k] if k < ls.size else np.arange(ls.size)
    return idx[np.lexsort((idx, -ls[idx]))]


def convert(dump: str, n_vocab: int, tokens: list[int] | None, context: int, out: str, fmt: str, k: int,
            engine_vocab: int | None) -> dict:
    size = pathlib.Path(dump).stat().st_size
    if size % (4 * n_vocab):
        raise ValueError("%s: %d bytes is not a whole number of %d-float rows (wrong --n-vocab?)" % (dump, size, n_vocab))
    n_rows = size // (4 * n_vocab)
    if tokens is not None and len(tokens) != n_rows:
        raise ValueError("%s has %d rows but the token list has %d tokens" % (dump, n_rows, len(tokens)))
    if not 0 <= context <= n_rows - 2:
        raise ValueError("context %d leaves nothing to score in %d positions (scored: context .. n-2)"
                         % (context, n_rows))
    ev = engine_vocab or n_vocab
    if ev < n_vocab:
        raise ValueError("--engine-vocab %d is smaller than the dump's vocabulary %d" % (ev, n_vocab))
    if fmt == "topk" and not 0 < k <= n_vocab:
        raise ValueError("--topk %d outside 1..%d" % (k, n_vocab))
    rows = np.memmap(dump, dtype=np.float32, mode="r", shape=(n_rows, n_vocab))
    nll, top1, n = 0.0, 0, 0
    pad = np.full(ev - n_vocab, -np.inf, dtype=np.float32) if ev > n_vocab else None
    with open(out, "wb") as f:
        if fmt == "topk":
            f.write(struct.pack("<i", k))
        for i in range(context, n_rows - 1):
            ls = log_softmax(np.asarray(rows[i]))
            if fmt == "topk":
                ids = topk(ls, k)
                f.write(ids.astype("<i4").tobytes())
                f.write(ls[ids].astype("<f4").tobytes())
            else:
                f.write(ls.astype("<f4").tobytes())
                if pad is not None:
                    f.write(pad.astype("<f4").tobytes())
            if tokens is not None:
                nx = tokens[i + 1]
                nll -= float(ls[nx]) if nx < n_vocab else float("inf")
                top1 += int(np.argmax(ls)) == nx
            n += 1
    return {"positions": n, "rows": n_rows, "nll": nll / n if tokens is not None else None,
            "top1": 100.0 * top1 / n if tokens is not None else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="llm_logits_dump output (n_tokens x n_vocab float32)")
    ap.add_argument("--n-vocab", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--tokens", help="the token ids given to llm_logits_dump (whitespace or JSON list)")
    ap.add_argument("--context", type=int, required=True, help="MAYA_GLM_SCORE value: positions context..n-2 scored")
    ap.add_argument("--format", choices=["topk", "full"], default="topk")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--engine-vocab", type=int, help="pad full-format rows to the engine's vocabulary")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        toks = read_tokens(a.tokens) if a.tokens else None
        r = convert(a.dump, a.n_vocab, toks, a.context, a.out, a.format, a.topk, a.engine_vocab)
    except (OSError, ValueError) as e:
        print("llama_to_score_ref: %s" % e, file=sys.stderr)
        return 2
    msg = "wrote %s: %d scored positions (%s) from %d rows" % (
        a.out, r["positions"], "top-%d" % a.topk if a.format == "topk" else "full log-softmax", r["rows"])
    if r["nll"] is not None:
        msg += "; reference nll %.5f ppl %.4f top1 %.2f%% vs the teacher tokens" % (r["nll"], np.exp(r["nll"]), r["top1"])
    print(msg)
    env = "MAYA_GLM_SCORE_TOPK" if a.format == "topk" else "MAYA_GLM_SCORE_REF"
    print("engine: MAYA_GLM_SCORE=%d %s=%s strata --glm-pack <pack> --tokens-file <the same ids>" % (a.context, env, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
