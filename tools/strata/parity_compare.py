#!/usr/bin/env python3
"""tools/strata/parity_compare.py - teacher-forced parity numbers and the B7.4a / M2 gates (BUILD-PLAN section 6).

Two subcommands:

  compare  one reference against one test over the scored positions:
             python3 tools/strata/parity_compare.py compare --ref ref.topk@2048 --test engine.lsm@2048 \
                 --name chat --out chat.a_vs_r.json
           Row files (SPEC = PATH[@FIRST], FIRST = the absolute position of the file's row 0):
             *.topk   tools/strata/llama_to_score_ref.py --format topk (STRATA_GLM_SCORE_TOPK): int32 k, then per
                      position k int32 ids and k float32 log-probs
             others   float32 rows of n_vocab: the engine's STRATA_GLM_SCORE_DUMP (log-softmax rows, positions
                      c .. n-2), llm_logits_dump's raw logits (positions 0 .. n-1) or llama_to_score_ref.py
                      --format full; every row is log-softmaxed here, so logits and log-probs mix freely
           KL(ref || test) per position over the reference's top-k tokens (--topk, default 64; a topk file's own k)
           plus one bucket for the rest of the vocabulary: the KL of every BUILD-PLAN gate (and of the engine's
           STRATA_GLM_SCORE line).  Reports mean, p50, p99, max, the top-1 agreement (reference best token vs the
           test's argmax), the near-ties (reference top-2 log-prob gap < 1e-3) and the misses outside them.
           --positions a:b limits the range (absolute positions, b exclusive), e.g. criterion (d)'s first 512.
  gate     the B7.4a / M2 / M2-alt verdict from a manifest of compare results:
             {"gate": "B7.4a" | "M2" | "M2-alt",
              "floor":          [compare JSONs: R vs R2, i.e. R_cpu vs R_cuda (M2) or R_cuda vs the pin's (M2-alt)],
              "a_vs_r":         [compare JSONs: R vs A (A_slow for B7.4a, A_fast strict for M2)],
              "slow_vs_fast":   [compare JSONs: A_slow vs A_fast, 4 short prompts x first 512 positions]  (d),
              "prod_vs_strict": [compare JSONs: A strict (eps 0) vs A production (eps 1e-6)]            (g),
              "extra": {"e": moe.json, "f": overlap.json, "h": slice.json, "i": tokenizer.json}}
           Paths are relative to the manifest.  An "extra" JSON passes when it has "pass": true (or
           "gate": {"pass": true}).  Writes markdown (stdout or --md) and JSON (--json); exit 0 = PASS, 1 = FAIL
           or incomplete (an M2 criterion with no input), 2 = usage error.

Pooled statistics: the mean and p99 are over every scored position of every prompt (not a mean of means).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys

import numpy as np

DEFAULT_VOCAB = 154880
NEAR_TIE = 1e-3


# ------------------------------------------------------------------------------------------------ row sources
def split_spec(spec: str) -> tuple[str, int]:
    if "@" in spec:
        p, f = spec.rsplit("@", 1)
        if f.lstrip("-").isdigit():
            return p, int(f)
    return spec, 0


class Rows:
    """Positions first .. first + n - 1 of a full or top-k row file."""

    def __init__(self, spec: str, n_vocab: int):
        self.path, self.first = split_spec(spec)
        p = pathlib.Path(self.path)
        size = p.stat().st_size
        self.topk = p.suffix == ".topk"
        if self.topk:
            with open(p, "rb") as f:
                self.k = struct.unpack("<i", f.read(4))[0]
            rec = 8 * self.k
            if self.k <= 0 or (size - 4) % rec:
                raise ValueError(f"{p}: not a top-k reference (k {self.k}, {size} bytes)")
            self.n = (size - 4) // rec
            self._mm = np.memmap(p, dtype=np.uint8, mode="r", offset=4, shape=(self.n, rec))
        else:
            if size == 0 or size % (4 * n_vocab):
                raise ValueError(f"{p}: {size} bytes is not a whole number of {n_vocab}-float rows (--n-vocab?)")
            self.k = 0
            self.n = size // (4 * n_vocab)
            self._mm = np.memmap(p, dtype="<f4", mode="r", shape=(self.n, n_vocab))
        self.n_vocab = n_vocab

    @property
    def last(self) -> int:
        return self.first + self.n - 1

    def full(self, pos: int) -> np.ndarray:
        """Log-softmax row (float64) at absolute position pos."""
        x = np.asarray(self._mm[pos - self.first], dtype=np.float64)
        m = np.max(x)
        return x - (m + math.log(float(np.exp(x - m).sum())))

    def top(self, pos: int) -> tuple[np.ndarray, np.ndarray]:
        rec = np.asarray(self._mm[pos - self.first])
        ids = rec[: 4 * self.k].view("<i4").astype(np.int64)
        lp = rec[4 * self.k:].view("<f4").astype(np.float64)
        return ids, lp


def topk_ids(ls: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-ls, k - 1)[:k] if k < ls.size else np.arange(ls.size)
    return idx[np.lexsort((idx, -ls[idx]))]


def kl_top_rest(ref_ids: np.ndarray, ref_lp: np.ndarray, test_ls: np.ndarray) -> float:
    """generate.cpp's STRATA_GLM_SCORE_TOPK KL: over the reference's ids plus the rest bucket."""
    p = np.exp(ref_lp)
    k = float((p * (ref_lp - test_ls[ref_ids])).sum())
    pr, pt = float(p.sum()), float(np.exp(test_ls[ref_ids]).sum())
    rr, rt = max(1.0 - pr, 1e-12), max(1.0 - pt, 1e-12)
    if pr < 1.0:
        k += rr * (math.log(rr) - math.log(rt))
    return max(k, 0.0)


def pctl(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))]


def compare(ref: Rows, test: Rows, k: int = 64, positions: tuple[int, int] | None = None, name: str = "") -> dict:
    if test.topk:
        raise ValueError(f"{test.path}: the test side needs full rows (a top-k file can only be the reference)")
    lo, hi = max(ref.first, test.first), min(ref.last, test.last) + 1
    if positions:
        lo, hi = max(lo, positions[0]), min(hi, positions[1])
    if hi <= lo:
        raise ValueError(f"no common positions: ref {ref.first}..{ref.last}, test {test.first}..{test.last}"
                         + (f", asked {positions[0]}..{positions[1] - 1}" if positions else ""))
    kls, same, ties, miss = [], 0, [], []
    for pos in range(lo, hi):
        t = test.full(pos)
        if ref.topk:
            ids, lp = ref.top(pos)
        else:
            r = ref.full(pos)
            ids = topk_ids(r, min(k, r.size))
            lp = r[ids]
        V = t.size
        ok = ids < V
        if not ok.all():   # a reference id past the test's vocabulary: probability 0 on the test side
            tt = np.full(int(ids.max()) + 1, -np.inf)
            tt[:V] = t
            t = tt
        kls.append(kl_top_rest(ids, lp, t))
        top = int(np.argmax(t))
        hit = int(ids[0]) == top
        same += hit
        tie = lp.size > 1 and lp[0] - lp[1] < NEAR_TIE
        if tie:
            ties.append(pos)
        elif not hit:
            miss.append(pos)
    s = sorted(kls)
    n = len(kls)
    return {"name": name, "ref": ref.path, "test": test.path, "first": lo, "last": hi - 1, "n": n,
            "k": ref.k if ref.topk else k, "kl": kls, "kl_mean": float(sum(kls) / n), "kl_p50": pctl(s, 0.5),
            "kl_p99": pctl(s, 0.99), "kl_max": s[-1], "top1_same": same, "top1_pct": 100.0 * same / n,
            "near_ties": ties, "top1_miss_outside_ties": miss}


# ------------------------------------------------------------------------------------------------ gate
def _load(base: pathlib.Path, paths) -> list[dict]:
    out = []
    for p in paths or []:
        q = pathlib.Path(p)
        out.append(json.loads((q if q.is_absolute() else base / q).read_text()))
    return out


def pooled(results: list[dict]) -> dict | None:
    if not results:
        return None
    kl = [v for r in results for v in r["kl"]]
    n = len(kl)
    same = sum(r["top1_same"] for r in results)
    s = sorted(kl)
    return {"n": n, "kl_mean": sum(kl) / n, "kl_p99": pctl(s, 0.99), "kl_max": s[-1], "top1_pct": 100.0 * same / n,
            "prompts": [r.get("name", "") for r in results]}


def _extra_pass(doc: dict) -> bool:
    if isinstance(doc.get("pass"), bool):
        return doc["pass"]
    g = doc.get("gate")
    return isinstance(g, dict) and bool(g.get("pass"))


def gate(manifest: dict, base: pathlib.Path) -> dict:
    kind = manifest.get("gate", "M2")
    if kind not in ("B7.4a", "M2", "M2-alt"):
        raise ValueError(f"gate {kind!r}: expected B7.4a, M2 or M2-alt")
    floor = pooled(_load(base, manifest.get("floor")))
    a = pooled(_load(base, manifest.get("a_vs_r")))
    crit: list[dict] = []

    def add(cid: str, what: str, ok: bool | None, detail: str):
        crit.append({"id": cid, "what": what, "pass": ok, "detail": detail})

    F = floor["kl_mean"] if floor else None
    ref_name = "R_cuda" if kind == "M2-alt" else "R"
    if a is None:
        add("a", f"mean KL({ref_name} || A) <= max(2F, 0.01)", None, "no a_vs_r results")
    else:
        lim = max(2 * F, 0.01) if F is not None else 0.01
        add("a", f"mean KL({ref_name} || A) <= max(2F, 0.01)", a["kl_mean"] <= lim,
            f"{a['kl_mean']:.5f} vs {lim:.5f} (F = {'%.5f' % F if F is not None else 'not measured: 0.01 used'}), "
            f"{a['n']} positions")
    if kind == "B7.4a":
        if a is not None:
            add("top1", "top-1 agreement >= 95%", a["top1_pct"] >= 95.0, f"{a['top1_pct']:.2f}%")
    else:
        if a is not None:
            add("b", "p99 KL <= 0.15", a["kl_p99"] <= 0.15, f"{a['kl_p99']:.4f}")
            want = 95.0
            det = f"{a['top1_pct']:.2f}% (>= 95%"
            if floor is not None:
                want = max(want, floor["top1_pct"] - 1.0)
                det += f", >= floor agreement {floor['top1_pct']:.2f}% - 1.0"
            add("c", "top-1 agreement", a["top1_pct"] >= want, det + ")")
        else:
            add("b", "p99 KL <= 0.15", None, "no a_vs_r results")
            add("c", "top-1 agreement", None, "no a_vs_r results")
        sf = pooled(_load(base, manifest.get("slow_vs_fast")))
        add("d", "mean KL(A_slow || A_fast) <= 0.002 (short prompts, first 512 positions)",
            None if sf is None else sf["kl_mean"] <= 0.002,
            "no slow_vs_fast results" if sf is None else f"{sf['kl_mean']:.5f} over {sf['n']} positions")
        ps = pooled(_load(base, manifest.get("prod_vs_strict")))
        add("g", "mean KL(strict || production eps) <= 0.005", None if ps is None else ps["kl_mean"] <= 0.005,
            "no prod_vs_strict results" if ps is None else f"{ps['kl_mean']:.5f} over {ps['n']} positions")
        extra = manifest.get("extra") or {}
        for cid, what in (("e", "top-8 expert ids equal at layers 3, 26, 54, 77 (first 8 positions, outside near-ties)"),
                          ("f", "top-2048 overlap >= 99.5% beyond 2048 on the 8K prompts (B7.6)"),
                          ("h", "slice oracle B7.5"), ("i", "tokenizer parity B12.8")):
            if cid in extra:
                doc = _load(base, [extra[cid]])[0]
                add(cid, what, _extra_pass(doc), str(extra[cid]))
            else:
                add(cid, what, None, "not run")
    failed = [c["id"] for c in crit if c["pass"] is False]
    missing = [c["id"] for c in crit if c["pass"] is None]
    verdict = "FAIL" if failed else ("INCOMPLETE" if missing else "PASS")
    return {"gate": kind, "verdict": verdict, "failed": failed, "missing": missing, "floor": floor, "a_vs_r": a,
            "criteria": crit}


def gate_markdown(g: dict) -> str:
    lines = [f"### {g['gate']} gate: **{g['verdict']}**", ""]
    if g["floor"]:
        f = g["floor"]
        lines.append(f"Floor F = {f['kl_mean']:.5f} (mean KL of the two references over {f['n']} positions; "
                     f"their top-1 agreement {f['top1_pct']:.2f}%).")
        lines.append("")
    lines += ["| Criterion | Check | Result | Detail |", "|---|---|---|---|"]
    for c in g["criteria"]:
        res = "not run" if c["pass"] is None else ("pass" if c["pass"] else "FAIL")
        lines.append(f"| ({c['id']}) | {c['what']} | {res} | {c['detail']} |")
    return "\n".join(lines) + "\n"


def compare_markdown(r: dict) -> str:
    return (f"| {r['name']} | {r['first']}..{r['last']} | {r['n']} | {r['kl_mean']:.5f} | {r['kl_p99']:.4f} | "
            f"{r['kl_max']:.4f} | {r['top1_pct']:.2f}% | {len(r['near_ties'])} | {len(r['top1_miss_outside_ties'])} |")


# ------------------------------------------------------------------------------------------------ CLI
def _range(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    a, b = s.split(":")
    return int(a), int(b)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="KL(ref || test) over the scored positions")
    c.add_argument("--ref", required=True, help="PATH[@FIRST]: .topk or float32 rows")
    c.add_argument("--test", required=True, help="PATH[@FIRST]: float32 rows (engine STRATA_GLM_SCORE_DUMP, ...)")
    c.add_argument("--n-vocab", type=int, default=DEFAULT_VOCAB, help="row width of both files")
    c.add_argument("--ref-vocab", type=int, help="row width of the reference when it differs")
    c.add_argument("--test-vocab", type=int, help="row width of the test when it differs")
    c.add_argument("--topk", type=int, default=64, help="reference tokens in the KL for a full reference")
    c.add_argument("--positions", help="a:b absolute positions (b exclusive)")
    c.add_argument("--name", default="")
    c.add_argument("--out", help="write the JSON result")
    g = sub.add_parser("gate", help="the B7.4a / M2 / M2-alt verdict from a manifest")
    g.add_argument("manifest")
    g.add_argument("--json", help="write the verdict as JSON")
    g.add_argument("--md", help="write the markdown table here instead of stdout")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "compare":
            ref = Rows(a.ref, a.ref_vocab or a.n_vocab)
            test = Rows(a.test, a.test_vocab or a.n_vocab)
            r = compare(ref, test, a.topk, _range(a.positions), a.name or pathlib.Path(test.path).stem)
            if a.out:
                pathlib.Path(a.out).write_text(json.dumps(r) + "\n")
            print(f"{r['name']}: positions {r['first']}..{r['last']} ({r['n']}): KL mean {r['kl_mean']:.6f} p99 "
                  f"{r['kl_p99']:.4f} max {r['kl_max']:.4f}, top-1 {r['top1_pct']:.2f}% "
                  f"({len(r['near_ties'])} near-ties, {len(r['top1_miss_outside_ties'])} misses outside them)")
            return 0
        mpath = pathlib.Path(a.manifest)
        res = gate(json.loads(mpath.read_text()), mpath.parent)
        md = gate_markdown(res)
        if a.md:
            pathlib.Path(a.md).write_text(md)
        else:
            print(md)
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
        return 0 if res["verdict"] == "PASS" else 1
    except (OSError, ValueError, KeyError) as e:
        print(f"parity_compare: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
