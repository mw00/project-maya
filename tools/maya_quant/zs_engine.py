"""tools/maya_quant/zs_engine.py - the zero-shot requests (zs_tasks.py) scored by the engine: one LOGP line each to
`strata --serve` (the choices of a question share their context: the engine restores it instead of reading it again),
one JSON line out per request with its summed log-probability.  Resumable: requests already in --out are skipped.

    python zs_engine.py --reqs zs.jsonl --out zs_maya.jsonl -- <strata args: --glm-pack PACK --max-context 4096 ...>
    python zs_tasks.py --score zs.jsonl --logp zs_maya.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default=str(pathlib.Path(__file__).resolve().parents[2] / "build" / "maya"))
    ap.add_argument("engine", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = [x for x in a.engine if x != "--"]
    reqs = [json.loads(l) for l in open(a.reqs, encoding="utf-8")]
    done = set()
    if pathlib.Path(a.out).exists():
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["task"], r["item"], r["choice"]))
    todo = [r for r in reqs if (r["task"], r["item"], r["choice"]) not in done]
    print(f"{len(reqs)} requests, {len(todo)} to score", flush=True)
    if not todo:
        return
    p = subprocess.Popen([a.exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         bufsize=1, cwd=pathlib.Path(a.exe).parents[1])
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit("the engine did not start")
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for k, r in enumerate(todo):
            p.stdin.write(f"LOGP {r['n_ctx']} {','.join(map(str, r['ids']))}\n")
            p.stdin.flush()
            for line in p.stdout:
                if line.startswith("LP "):
                    _, lp, n, g = line.split()
                    fo.write(json.dumps({"task": r["task"], "item": r["item"], "choice": r["choice"],
                                         "logp": float(lp), "n": int(n), "greedy": int(g)}) + "\n")
                    break
                if line.startswith("ERR"):
                    raise SystemExit(f"request {k}: {line.strip()}")
            else:
                raise SystemExit("the engine stopped")
            if (k + 1) % 200 == 0:
                fo.flush()
                el = time.time() - t0
                print(f"{k + 1}/{len(todo)} in {el:.0f} s ({el / (k + 1):.2f} s each)", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=120)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
