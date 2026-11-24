#!/usr/bin/env python3
"""tools/glm_dsa_mtp_probe.py - the NextN (MTP) draft of a glm-dsa GGUF from a dumped trunk hidden state (B7.5, B11.1).

The glm-dsa successor of Maya's tools/glm_mtp_probe.py (Flash, blk.45).  The draft block reads, for row t, the
trunk's post-output-norm hidden h_t (llama.cpp's t_h_nextn) and the embedding of token t+1 at position pos0 + t, and
predicts token t+2: [enorm(emb) ; hnorm(h)] -> eh_proj -> one full decoder block -> shared_head_norm (fallback
output_norm) -> head.  The block is ref/glm_dsa.py's GlmDsa.mtp (exact gguf-py dequant, lazy per-expert reads), so it
runs on the real 744B GGUF: only the NextN block, the embedding rows of the tokens and the head are read.

    # the trunk hidden from llama.cpp (LLM_CB_DIR of tools/llm_logits_dump: h_nextn.f32, T rows of n_embd)
    python3 tools/glm_dsa_mtp_probe.py $M --tokens p.tokens --hidden cb/h_nextn.f32 \
        --compare engine=draft.f32@0 --positions 0:16 --report probe.json
    # small models: the oracle computes the trunk itself
    python3 tools/glm_dsa_mtp_probe.py small.gguf --tokens 1,2,3,4,5 --oracle-trunk --draft-out d.f32

Hidden sources (one): --hidden H, the post-norm hidden (.npy (n_embd, T) / (T, n_embd), or raw float32 rows);
--trunk-out H, the trunk's last l_out (pre-norm; output_norm is applied here); --oracle-trunk, the oracle's own forward
(small GGUFs only).

Comparisons: --compare NAME=FILE[@FIRST] (repeatable): FILE holds float32 draft logits rows of n_vocab, row 0 being
the draft made at position FIRST (default 0; e.g. ref/glm_dsa.py --mtp-out writes rows 0 .. T-2).  --compare-ids
NAME=a,b,c[@FIRST] compares draft token ids instead (Glm5Model::mtp_draft).  Per position: KL(oracle || NAME) over the
oracle's top-64 tokens plus the rest, the top-1 match, and the near-ties (oracle top-2 gap < 1e-3); the B11.1 gate is a
mean KL < 0.01 over the positions (--gate).  The draft's acceptance against the text (draft at t == token t+2) is
reported too.  Exit 1 when a gate fails, 2 on a usage or read error.

Head-vs-trunk agreement (B11.3 / B15.4, llama.cpp #28351 examples/mtp-agree): teacher-forced, the draft at row t
(predicting token t+2) against the TRUNK's own greedy token for position t+1 - what an MTP draft must equal to be
accepted under greedy decoding, independent of the text.  The trunk's greedy ids come from --oracle-trunk (the
oracle's logits), --trunk-logits FILE[@FIRST] (float32 rows of n_vocab, row 0 = position FIRST, e.g. llama.cpp's
logits dump or the engine's) or --trunk-ids a,b,c[@FIRST] (the engine's argmax series).  Reported for the oracle
draft and for every --compare / --compare-ids draft (an IQ2 head vs a Q4_K head through MAYA_GLM_MTP_GGUF);
--min-agree X fails (exit 1) when a source agrees on fewer than X of the rows.  Upstream: Q4_0 0.718, bf16 0.724,
IQ2_XS collapses; the M4 gate is 0.6 live first-position acceptance (the engine's MAYA_GLM_SPEC_LOG counters).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref"))

from glm_dsa import (GlmDsa, Options, kl_top_rest, load_hidden, parse_positions, rms_norm)  # noqa: E402

GATE_KL = 0.01          # B11.1: draft logits vs the slice oracle, mean KL < 0.01 at 16 positions
NEAR_TIE = 1e-3


def read_tokens(spec: str) -> list[int]:
    p = pathlib.Path(spec)
    text = p.read_text() if p.exists() else spec
    text = text.replace(",", " ").replace("[", " ").replace("]", " ")
    return [int(x) for x in text.split()]


def _split_first(v: str) -> tuple[str, int]:
    if "@" in v:
        a, b = v.rsplit("@", 1)
        return a, int(b)
    return v, 0


def parse_compare(items: list[str]) -> list[tuple[str, str, int]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--compare {it!r}: expected NAME=FILE[@FIRST]")
        name, rest = it.split("=", 1)
        path, first = _split_first(rest)
        out.append((name, path, first))
    return out


def draft_rows(path: str, n_vocab: int, first: int, rows_abs: np.ndarray) -> np.ndarray:
    """(n_vocab, len(rows_abs)) float64 from a float32 (rows, V) file whose row 0 is draft row `first`."""
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0 or raw.size % n_vocab:
        raise ValueError(f"{path}: {raw.size} floats is not a whole number of {n_vocab}-float rows")
    rows = raw.reshape(-1, n_vocab)
    idx = np.asarray(rows_abs, dtype=np.int64) - first
    if idx.size and (idx.min() < 0 or idx.max() >= rows.shape[0]):
        raise ValueError(f"{path}: rows cover drafts {first} .. {first + rows.shape[0] - 1}, not "
                         f"{int(rows_abs.min())} .. {int(rows_abs.max())}")
    return rows[idx].T.astype(np.float64)


def probe(m: GlmDsa, tokens: list[int], h_post: np.ndarray, pos0: int = 0) -> np.ndarray:
    """Draft logits (vocab, T-1): row t from h_post[:, t] and tokens[t+1]."""
    T = len(tokens)
    if h_post.shape[1] < T - 1:
        raise ValueError(f"the hidden state has {h_post.shape[1]} positions, the drafts need {T - 1}")
    return m.mtp(h_post[:, : T - 1], tokens[1:T], pos0)


def compare(draft: np.ndarray, other: np.ndarray) -> dict:
    """Oracle draft vs another draft, both (vocab, P)."""
    k = kl_top_rest(draft, other)
    a, b = draft.argmax(axis=0), other.argmax(axis=0)
    srt = -np.sort(-draft, axis=0)
    gap = srt[0] - srt[1]
    ties = np.nonzero(gap < NEAR_TIE)[0]
    miss = np.nonzero(a != b)[0]
    return {"kl_mean": float(k.mean()), "kl_max": float(k.max()), "kl": [float(v) for v in k],
            "top1_same": int((a == b).sum()), "n": int(k.size),
            "top1_miss_outside_ties": [int(i) for i in miss if i not in set(ties.tolist())],
            "near_ties": [int(i) for i in ties]}


def trunk_greedy(T: int, n_vocab: int, oracle_logits, logits_spec: str | None, ids_spec: str | None) -> np.ndarray:
    """The trunk's greedy token per position 0..T-1 (-1 where unknown): the oracle's logits (vocab, T), a float32
    logits file (rows of n_vocab from position FIRST) or an id list (from FIRST)."""
    out = np.full(T, -1, dtype=np.int64)
    if oracle_logits is not None:
        out[: oracle_logits.shape[1]] = oracle_logits.argmax(axis=0)[:T]
    if logits_spec:
        path, first = _split_first(logits_spec)
        raw = np.fromfile(path, dtype="<f4")
        if raw.size == 0 or raw.size % n_vocab:
            raise ValueError(f"{path}: {raw.size} floats is not a whole number of {n_vocab}-float rows")
        ids = raw.reshape(-1, n_vocab).argmax(axis=1)
        n = max(0, min(len(ids), T - first))
        out[first:first + n] = ids[:n]
    if ids_spec:
        ids_s, first = _split_first(ids_spec)
        ids = [int(x) for x in ids_s.split(",") if x]
        n = max(0, min(len(ids), T - first))
        out[first:first + n] = ids[:n]
    return out


def agreement(draft_ids: np.ndarray, rows: np.ndarray, trunk_arg: np.ndarray) -> dict:
    """Teacher-forced head-vs-trunk agreement: the draft at row t (for token t+2) vs the trunk's greedy token at
    position t+1, over the rows where that is known."""
    same = n = 0
    for k, t in enumerate(rows):
        if t + 1 < trunk_arg.size and trunk_arg[t + 1] >= 0:
            n += 1
            same += int(draft_ids[k] == trunk_arg[t + 1])
    return {"same": same, "n": n, "rate": (same / n) if n else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="a glm-dsa GGUF with a NextN block (any shard of a split file)")
    ap.add_argument("--tokens", required=True, help="token ids: a comma list or a file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hidden", help="post-output-norm trunk hidden (llama.cpp h_nextn / result_norm)")
    src.add_argument("--trunk-out", help="the trunk's last l_out (pre-norm): output_norm is applied here")
    src.add_argument("--oracle-trunk", action="store_true", help="compute the trunk with the oracle (small models)")
    ap.add_argument("--mode", choices=("hf", "llama"), default="hf")
    ap.add_argument("--pos0", type=int, default=0)
    ap.add_argument("--positions", default="0:16", help="draft rows to report (a:b, a,b,c, -N; default 0:16)")
    ap.add_argument("--compare", action="append", default=[], help="NAME=FILE[@FIRST] draft logits rows")
    ap.add_argument("--compare-ids", action="append", default=[], help="NAME=a,b,c[@FIRST] draft token ids")
    ap.add_argument("--gate", type=float, default=GATE_KL, help="mean KL gate per --compare (default 0.01)")
    ap.add_argument("--draft-out", help="write the oracle's draft logits rows (T-1, vocab) float32")
    ap.add_argument("--trunk-logits", help="FILE[@FIRST]: the trunk's logits rows (float32, n_vocab) for --agree")
    ap.add_argument("--trunk-ids", help="a,b,c[@FIRST]: the trunk's greedy ids per position for --agree")
    ap.add_argument("--agree", action="store_true", help="report head-vs-trunk greedy agreement (B11.3)")
    ap.add_argument("--min-agree", type=float, default=None, help="fail when an agreement rate is below this")
    ap.add_argument("--report", help="write the results as JSON")
    a = ap.parse_args(argv)
    try:
        tokens = read_tokens(a.tokens)
        if len(tokens) < 2:
            raise ValueError("need at least 2 tokens")
        m = GlmDsa(a.model, Options(mode=a.mode))
        c = m.cfg
        if c.n_nextn < 1:
            raise ValueError("this GGUF has no NextN block")
        trunk_logits = None
        if a.oracle_trunk:
            trunk_logits = m.forward(tokens, a.pos0)
            h = m.h_nextn
        elif a.hidden:
            h = load_hidden(a.hidden, c.n_embd)
        else:
            h = rms_norm(load_hidden(a.trunk_out, c.n_embd), m.W("output_norm.weight"), c.rms_eps)
        draft_all = probe(m, tokens, h, a.pos0)                         # (vocab, T-1)
        cols = parse_positions(a.positions, draft_all.shape[1], 0)
        draft = draft_all[:, cols]
        comps = parse_compare(a.compare)
        id_comps = []
        for it in a.compare_ids:
            name, rest = it.split("=", 1)
            ids, first = _split_first(rest)
            id_comps.append((name, [int(x) for x in ids.split(",") if x], first))
        trunk_arg = trunk_greedy(len(tokens), c.n_vocab, trunk_logits, a.trunk_logits, a.trunk_ids)
        if (a.agree or a.min_agree is not None) and not (trunk_arg >= 0).any():
            raise ValueError("--agree needs the trunk's greedy ids: --oracle-trunk, --trunk-logits or --trunk-ids")
    except (OSError, ValueError, KeyError) as e:
        print(f"glm_dsa_mtp_probe: {e}", file=sys.stderr)
        return 2
    if a.draft_out:
        np.ascontiguousarray(draft_all.T.astype(np.float32)).tofile(a.draft_out)
    arg = draft.argmax(axis=0)
    truth = np.asarray([tokens[i + 2] if i + 2 < len(tokens) else -1 for i in cols])
    have = truth >= 0
    acc = int((arg[have] == truth[have]).sum())
    print(f"{c.summary()}")
    print(f"oracle drafts at rows {cols[0]}..{cols[-1]} ({cols.size}): argmax {[int(v) for v in arg[:16]]}; "
          f"matches the text (token t+2) {acc}/{int(have.sum())}")
    rep: dict = {"mode": a.mode, "rows": [int(i) for i in cols], "draft_argmax": [int(v) for v in arg],
                 "text_match": acc, "text_n": int(have.sum()), "compare": {}}
    status = 0
    agree_on = a.agree or a.min_agree is not None

    def agree_line(name: str, ids: np.ndarray) -> None:
        nonlocal status
        r = agreement(ids, cols, trunk_arg)
        if a.min_agree is not None:
            r["gate"] = a.min_agree
            r["pass"] = r["n"] > 0 and r["rate"] >= a.min_agree
            status = status or (0 if r["pass"] else 1)
        rep.setdefault("agree", {})[name] = r
        print(f"head-vs-trunk agreement ({name}): draft == the trunk's greedy token at {r['same']}/{r['n']} rows "
              f"({r['rate']:.3f})" + ("" if a.min_agree is None else
                                       f" -> {'PASS' if r['pass'] else 'FAIL'} (gate {a.min_agree:g})"))

    if agree_on:
        agree_line("oracle", arg)
    try:
        for name, path, first in comps:
            other = draft_rows(path, c.n_vocab, first, cols)
            r = compare(draft, other)
            r["gate"] = a.gate
            r["pass"] = r["kl_mean"] < a.gate
            rep["compare"][name] = r
            status = status or (0 if r["pass"] else 1)
            print(f"{name}: KL(oracle || {name}) mean {r['kl_mean']:.6g} max {r['kl_max']:.4g}, same top-1 "
                  f"{r['top1_same']}/{r['n']} (near-ties {len(r['near_ties'])}) -> "
                  f"{'PASS' if r['pass'] else 'FAIL'} (gate mean KL < {a.gate:g})")
            if agree_on:
                agree_line(name, other.argmax(axis=0))
        for name, ids, first in id_comps:
            idx = cols - first
            if idx.min() < 0 or idx.max() >= len(ids):
                raise ValueError(f"--compare-ids {name}: {len(ids)} ids from row {first} do not cover the rows")
            got = np.asarray(ids)[idx]
            same = int((got == arg).sum())
            rep["compare"][name] = {"ids": [int(v) for v in got], "top1_same": same, "n": int(cols.size)}
            print(f"{name}: draft ids equal the oracle's argmax at {same}/{cols.size} rows")
            if agree_on:
                agree_line(name, got)
    except (OSError, ValueError) as e:
        print(f"glm_dsa_mtp_probe: {e}", file=sys.stderr)
        return 2
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
