#!/usr/bin/env python3
"""Compare two llama.cpp reference dumps of the same prompt (tools/ik2/ref/run_dump.sh), e.g. F16 KV + flash
attention against F32 KV without it: the reference's own noise floor, to read maya-ik2's --parity numbers against.

  compare_dumps.py <dump A> <dump B> [--tensor l_out] [--worst N]

Per layer: the minimum and mean cosine over positions (prompt batch and decode steps), the share of router picks
that agree (as sets), and the positions of the N worst cosines.
"""
import argparse
import json
import os

import numpy as np


def load_index(d):
    idx = {}
    with open(os.path.join(d, "index.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            idx[(e["step"], e["name"])] = e
    return idx


def read(d, e):
    a = np.fromfile(os.path.join(d, e["file"]), dtype=np.int32 if e["type"] == "i32" else np.float32)
    return a.reshape(-1, e["ne"][0])  # rows = positions


def cos_rows(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tensor", default="l_out")
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args()
    ia, ib = load_index(args.a), load_index(args.b)
    steps = sorted({s for s, _ in ia} & {s for s, _ in ib})
    layers = sorted({int(n.rsplit("-", 1)[1]) for s, n in ia if n.startswith(args.tensor + "-")})
    print(f"{args.tensor}: layer  cos min   cos mean  | picks same | worst positions (cos)")
    for L in layers:
        cs, pos = [], []
        same = total = 0
        base = 0
        for s in steps:
            k = (s, f"{args.tensor}-{L}")
            if k not in ia or k not in ib:
                continue
            ra, rb = read(args.a, ia[k]), read(args.b, ib[k])
            c = cos_rows(ra, rb)
            cs.append(c)
            pos.append(np.arange(base, base + len(c)))
            base += len(c)
            kt = (s, f"ffn_moe_topk-{L}")
            if kt in ia and kt in ib:
                ta, tb = read(args.a, ia[kt]), read(args.b, ib[kt])
                for x, y in zip(ta, tb):
                    same += len(set(x.tolist()) & set(y.tolist()))
                    total += len(x)
        if not cs:
            continue
        c = np.concatenate(cs)
        p = np.concatenate(pos)
        o = np.argsort(c)[: args.worst]
        worst = " ".join(f"{p[i]}({c[i]:.4f})" for i in o)
        ps = f"{100.0 * same / total:7.3f}%" if total else "    -   "
        print(f"  {L:5d}  {c.min():.6f}  {c.mean():.6f} | {ps} | {worst}")


if __name__ == "__main__":
    main()
