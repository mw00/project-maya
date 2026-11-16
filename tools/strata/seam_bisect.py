#!/usr/bin/env python3
"""tools/strata/seam_bisect.py - compare two seam-dump directories: the engine's GLM_CB_DIR, llama.cpp's LLM_CB_DIR
(tools/llm_logits_dump) and the oracle's --dump-dir (ref/glm_dsa.py).  The B7.4b / M2 failure triage, criterion (e)
and the B7.6 selection overlap (BUILD-PLAN section 6).

  bisect   the first seam, in forward order, where two runs part:
             python3 tools/strata/seam_bisect.py bisect --a eng_cb --b llama_cb --b-tokens 300 --json bisect.json
           Each side's files are <seam>-<layer>.(f32|i32|bin|npy) (result_norm / result_output without a layer).  A
           raw file holds TOKENS rows (--a-tokens / --b-tokens; default 1, or "auto": the file size over the other
           side's per-token size); the compared row is --token (default -1: the last).  The engine without
           GLM_CB_APPEND keeps the last token only (1 row); llm_logits_dump keeps its last decode call (--ubatch rows)
           or, with LLM_CB_APPEND=1, every call; an oracle .npy has T on its last axis.  Float seams: rel L2
           |a - b| / |b| against --rel (default 2e-2); id seams (top_k, ffn_moe_topk; int32 bytes even in a .f32
           file): the overlap of the two id sets (-1 padding dropped) against --overlap (default 0.99).
           --skip lists seams to ignore (default indexer_q,indexer_k: llama.cpp cb's them after its Hadamard
           rotation, which leaves the scores unchanged).  Exit 0 when nothing diverges, 1 at a divergence.
  overlap  B7.6 / M2 (f): per position, |A cells & B cells| / |B cells| of the top-k selections (top_k-<layer>)
           appended per token (engine GLM_CB_APPEND=1 GLM_CB_ONLY=top_k, llama LLM_CB_APPEND=1 LLM_CB_ONLY=top_k):
             python3 tools/strata/seam_bisect.py overlap --a eng_cb --b llama_cb --layers 0,26,54,74 \
                 --positions 2049:8192 --top-k 2048 --json overlap.json
           Row r of a file is position FIRST + r (--a-first / --b-first).  --ties FILE excludes near-tie positions
           ({"<layer>": [positions]} JSON).  Pass: every layer's overlap >= --gate (default 99.5%) at every position
           outside the ties.  The same command compares F32 vs FP16 index keys (two engine runs).
  moe      M2 (e): the routed expert ids (ffn_moe_topk-<layer>, appended per token) equal as sets at --layers for
           --positions (default 3,26,54,77 and 0:8), outside --ties.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

ORDER = ["attn_norm", "qr", "indexer_q", "indexer_k", "indexer_weights", "indexer_kq", "indexer_score", "top_k", "q",
         "q_nope", "q_pe", "kv_cmpr_pe", "kv_cmpr", "k_pe", "Kcur", "Qcur", "q_nope_absorbed", "kqv_out", "attn_out",
         "ffn_inp", "ffn_norm", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights",
         "ffn_moe_weights_scaled", "ffn_moe_out", "ffn_shexp", "ffn_out", "l_out"]
TAIL = ["result_norm", "result_output"]
ID_SEAMS = {"top_k", "ffn_moe_topk"}
DEFAULT_SKIP = "indexer_q,indexer_k"
_NAME = re.compile(r"^(?P<seam>.+?)(?:-(?P<layer>\d+))?\.(?P<ext>f32|i32|bin|npy)$")


# ------------------------------------------------------------------------------------------------ reading
def scan(d: pathlib.Path) -> dict[str, pathlib.Path]:
    """dump name ("l_out-3", "result_norm") -> file."""
    out = {}
    for p in sorted(d.iterdir()):
        m = _NAME.match(p.name)
        if m:
            key = m.group("seam") + (f"-{m.group('layer')}" if m.group("layer") is not None else "")
            out.setdefault(key, p)
    return out


def split_name(key: str) -> tuple[str, int]:
    m = re.match(r"^(.+)-(\d+)$", key)
    return (m.group(1), int(m.group(2))) if m else (key, 10 ** 9)


def order_key(key: str) -> tuple:
    seam, layer = split_name(key)
    if layer == 10 ** 9:
        return (layer, TAIL.index(seam) if seam in TAIL else len(TAIL), seam)
    return (layer, ORDER.index(seam) if seam in ORDER else len(ORDER), seam)


def raw(p: pathlib.Path, ids: bool) -> np.ndarray:
    if p.suffix == ".npy":
        return np.load(p)
    return np.fromfile(p, dtype="<i4" if ids else "<f4")


def token_row(arr: np.ndarray, p: pathlib.Path, seam: str, n_tok: int, token: int) -> np.ndarray:
    """One token's values.  .npy: T on the last axis (a (T, S) mask/score for top_k / indexer_score)."""
    if p.suffix == ".npy":
        if seam == "top_k" and arr.dtype == bool:          # the oracle's (T, T) selection mask
            return np.nonzero(arr[token])[0].astype(np.int64)
        if seam == "indexer_score" and arr.ndim == 2:
            return np.asarray(arr[token], dtype=np.float64)
        return np.asarray(arr[..., token]).ravel(order="F")
    if n_tok <= 0 or arr.size % n_tok:
        raise ValueError(f"{p.name}: {arr.size} values are not {n_tok} equal token rows")
    return arr.reshape(n_tok, -1)[token]


def ids_overlap(a: np.ndarray, b: np.ndarray) -> float:
    sa = {int(x) for x in np.asarray(a).ravel() if x >= 0}
    sb = {int(x) for x in np.asarray(b).ravel() if x >= 0}
    if not sb:
        return 1.0 if not sa else 0.0
    return len(sa & sb) / max(len(sb), len(sa))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        return math.inf
    fin = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return math.inf
    n = float(np.linalg.norm(b[fin]))
    e = float(np.linalg.norm(a[fin] - b[fin]))
    return e / n if n > 0 else (0.0 if e == 0 else math.inf)


def _tokens(spec: str, own: int, other: int | None) -> int:
    if spec == "auto":
        if other is None or other <= 0 or own % other:
            return 1
        return own // other
    return int(spec)


# ------------------------------------------------------------------------------------------------ bisect
def bisect(a_dir: pathlib.Path, b_dir: pathlib.Path, a_tokens: str = "1", b_tokens: str = "auto", token: int = -1,
           rel: float = 2e-2, overlap: float = 0.99, skip: set[str] | None = None,
           layers: tuple[int, int] | None = None) -> dict:
    A, B = scan(a_dir), scan(b_dir)
    skip = skip or set()
    common = sorted((k for k in A if k in B and split_name(k)[0] not in skip), key=order_key)
    if layers:
        common = [k for k in common if split_name(k)[1] == 10 ** 9 or layers[0] <= split_name(k)[1] < layers[1]]
    rows, first = [], None
    for key in common:
        seam, layer = split_name(key)
        ids = seam in ID_SEAMS
        try:
            ra, rb = raw(A[key], ids), raw(B[key], ids)
            # per-token sizes: a side with a known token count fixes the other's "auto"
            na = _tokens(a_tokens, ra.size, None) if a_tokens != "auto" else None
            nb = _tokens(b_tokens, rb.size, None) if b_tokens != "auto" else None
            if na is None and nb is None:
                na = nb = 1
            if na is None:
                na = _tokens("auto", ra.size, rb.size // nb if nb else None)
            if nb is None:
                nb = _tokens("auto", rb.size, ra.size // na if na else None)
            va = token_row(ra, A[key], seam, na, token)
            vb = token_row(rb, B[key], seam, nb, token)
        except ValueError as e:
            rows.append({"seam": key, "metric": "error", "value": None, "ok": False, "detail": str(e)})
            first = first or key
            continue
        if ids:
            v = ids_overlap(va, vb)
            ok = v >= overlap
            rows.append({"seam": key, "metric": "overlap", "value": v, "ok": ok, "n_a": int((va >= 0).sum()),
                         "n_b": int((vb >= 0).sum())})
        else:
            v = rel_l2(va, vb)
            ok = v <= rel
            rows.append({"seam": key, "metric": "rel_l2", "value": v, "ok": ok, "n": int(va.size)})
        if not ok and first is None:
            first = key
    only_a = sorted(k for k in A if k not in B)
    only_b = sorted(k for k in B if k not in A)
    return {"a": str(a_dir), "b": str(b_dir), "compared": len(rows), "first_divergence": first, "rows": rows,
            "only_a": only_a, "only_b": only_b, "rel": rel, "overlap": overlap}


# ------------------------------------------------------------------------------------------------ overlap / moe
def appended_rows(p: pathlib.Path, width: int) -> np.ndarray:
    a = np.fromfile(p, dtype="<i4")
    if a.size % width:
        raise ValueError(f"{p.name}: {a.size} ids are not rows of {width}")
    return a.reshape(-1, width)


def _find(d: pathlib.Path, name: str) -> pathlib.Path:
    for ext in (".f32", ".i32", ".bin"):
        p = d / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"{d}/{name}.(f32|i32|bin) not found")


def _ties(path: str | None) -> dict[int, set[int]]:
    if not path:
        return {}
    doc = json.loads(pathlib.Path(path).read_text())
    return {int(k): {int(x) for x in v} for k, v in doc.items()}


def set_compare(a_dir, b_dir, seam: str, layers: list[int], positions: tuple[int, int], width: int, a_first: int,
                b_first: int, gate: float, ties: dict[int, set[int]]) -> dict:
    out: dict = {"seam": seam, "positions": list(positions), "width": width, "gate": gate, "layers": {}}
    all_ok = True
    for il in layers:
        ra = appended_rows(_find(pathlib.Path(a_dir), f"{seam}-{il}"), width)
        rb = appended_rows(_find(pathlib.Path(b_dir), f"{seam}-{il}"), width)
        lo = max(positions[0], a_first, b_first)
        hi = min(positions[1], a_first + ra.shape[0], b_first + rb.shape[0])
        if hi <= lo:
            raise ValueError(f"{seam}-{il}: no common positions in {positions[0]}..{positions[1] - 1} "
                             f"(A rows {a_first}..{a_first + ra.shape[0] - 1}, B rows {b_first}..{b_first + rb.shape[0] - 1})")
        tl = ties.get(il, set())
        vals, below, excluded = [], [], 0
        for pos in range(lo, hi):
            v = ids_overlap(ra[pos - a_first], rb[pos - b_first])
            if pos in tl:
                excluded += 1
                continue
            vals.append(v)
            if v * 100.0 < gate:
                below.append(pos)
        ok = not below
        all_ok = all_ok and ok
        out["layers"][str(il)] = {"first": lo, "last": hi - 1, "n": len(vals), "excluded_ties": excluded,
                                  "mean_pct": 100.0 * float(np.mean(vals)) if vals else None,
                                  "min_pct": 100.0 * float(np.min(vals)) if vals else None,
                                  "below_gate": below[:200], "n_below": len(below), "pass": ok}
    out["pass"] = all_ok
    return out


# ------------------------------------------------------------------------------------------------ CLI
def _range(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


def _layers(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bisect")
    b.add_argument("--a", required=True)
    b.add_argument("--b", required=True)
    b.add_argument("--a-tokens", default="1")
    b.add_argument("--b-tokens", default="auto")
    b.add_argument("--token", type=int, default=-1)
    b.add_argument("--rel", type=float, default=2e-2)
    b.add_argument("--overlap", type=float, default=0.99)
    b.add_argument("--skip", default=DEFAULT_SKIP)
    b.add_argument("--layers", help="a:b only these layers")
    b.add_argument("--json")
    for name in ("overlap", "moe"):
        o = sub.add_parser(name)
        o.add_argument("--a", required=True)
        o.add_argument("--b", required=True)
        o.add_argument("--layers", default="0,26,54,74" if name == "overlap" else "3,26,54,77")
        o.add_argument("--positions", default="2049:8192" if name == "overlap" else "0:8")
        o.add_argument("--a-first", type=int, default=0)
        o.add_argument("--b-first", type=int, default=0)
        o.add_argument("--ties", help="near-tie positions to exclude: {\"<layer>\": [positions]}")
        o.add_argument("--json")
        if name == "overlap":
            o.add_argument("--top-k", type=int, default=2048)
            o.add_argument("--gate", type=float, default=99.5, help="overlap percent")
        else:
            o.add_argument("--n-used", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "bisect":
            r = bisect(pathlib.Path(a.a), pathlib.Path(a.b), a.a_tokens, a.b_tokens, a.token, a.rel, a.overlap,
                       {s for s in a.skip.split(",") if s}, _range(a.layers) if a.layers else None)
            for row in r["rows"]:
                v = row["value"]
                vs = "-" if v is None else (f"{v:.4g}")
                print(f"  {'ok  ' if row['ok'] else 'DIFF'} {row['seam']:<28} {row['metric']:<8} {vs}"
                      + (f"  {row.get('detail', '')}" if row["metric"] == "error" else ""))
            print(f"seam_bisect: {r['compared']} seams compared; first divergence: {r['first_divergence'] or 'none'}"
                  + (f"; only in A: {len(r['only_a'])}, only in B: {len(r['only_b'])}" if r["only_a"] or r["only_b"] else ""))
            status = 1 if r["first_divergence"] else 0
        else:
            seam = "top_k" if a.cmd == "overlap" else "ffn_moe_topk"
            width = a.top_k if a.cmd == "overlap" else a.n_used
            gate = a.gate if a.cmd == "overlap" else 100.0
            r = set_compare(a.a, a.b, seam, _layers(a.layers), _range(a.positions), width, a.a_first, a.b_first, gate,
                            _ties(a.ties))
            for il, L in r["layers"].items():
                print(f"  layer {il}: positions {L['first']}..{L['last']} ({L['n']}, {L['excluded_ties']} ties "
                      f"excluded): mean {L['mean_pct']:.3f}% min {L['min_pct']:.3f}%, {L['n_below']} below "
                      f"{gate:g}% -> {'pass' if L['pass'] else 'FAIL'}")
            print(f"seam_bisect {a.cmd}: {'PASS' if r['pass'] else 'FAIL'}")
            status = 0 if r["pass"] else 1
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(r, indent=1) + "\n")
        return status
    except (OSError, ValueError) as e:
        print(f"seam_bisect: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
