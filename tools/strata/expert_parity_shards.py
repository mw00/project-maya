#!/usr/bin/env python3
"""tools/strata/expert_parity_shards.py - B5.3 / B13.1: native_expert_parity over a sharded GLM-5.3 GGUF.

    expert_parity_shards.py --bin build/native_expert_parity --layers "3 26 54 77 78" SHARD [SHARD ...]
    expert_parity_shards.py --plan --layers "3;26;54;77;78" SHARD [SHARD ...]

native_expert_parity takes one GGUF file and counts every requested layer whose ffn_{gate,up,down}_exps tensors are
not in that file as a failure. The real GLM-5.3 file is split: shard 1 holds metadata only and the 78 + 1 layers are
spread over the others, so no shard holds layers 3, 26, 54, 77 and 78 together. This wrapper reads each shard's
header (no tensor data, no numpy), gives each shard only the requested layers it holds, and fails when a requested
layer is in no shard or has its gate/up/down in different shards (the tool needs the three in one file).

Exit: 0 every run passed; 1 a run failed or a layer is not testable; 2 usage / unreadable shard; 77 every run
returned 77 (no device).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_audit  # noqa: E402  (pure-struct GGUF header reader)

ROLES = ("gate", "up", "down")
_EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def shard_layers(path: pathlib.Path) -> dict[int, set[str]]:
    """layer -> the expert roles (gate/up/down) this shard holds."""
    with open(path, "rb") as f:
        sh = gguf_audit.read_header(f, path.name, path.stat().st_size, 0)
    out: dict[int, set[str]] = {}
    for t in sh.tensors:
        m = _EXPS.match(t.name)
        if m:
            out.setdefault(int(m.group(1)), set()).add(m.group(2))
    return out


def plan(shards: list[pathlib.Path], layers: list[int]) -> tuple[list[tuple[pathlib.Path, list[int]]], list[str]]:
    """(shard, its layers) for every shard that holds a requested layer whole, and the problems."""
    held = [(p, shard_layers(p)) for p in shards]
    runs: dict[pathlib.Path, list[int]] = {}
    problems = []
    for l in layers:
        whole = [p for p, h in held if h.get(l, set()) >= set(ROLES)]
        if whole:
            runs.setdefault(whole[0], []).append(l)
            continue
        parts = {p.name: sorted(h[l]) for p, h in held if l in h}
        if parts:
            problems.append(f"layer {l}: gate/up/down split over shards {parts}; native_expert_parity needs one file")
        else:
            problems.append(f"layer {l}: no shard holds its ffn_*_exps tensors (shards: "
                            f"{', '.join(p.name for p in shards)})")
    return [(p, runs[p]) for p in shards if p in runs], problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", help="native_expert_parity executable")
    ap.add_argument("--layers", required=True, help="layers to check: one argument, separated by spaces, commas or "
                                                    "';' (a CMake list)")
    ap.add_argument("--plan", action="store_true", help="print the shard -> layers plan and exit")
    ap.add_argument("shards", nargs="+", help="GGUF shards (any order; all of the file's shards is fine)")
    a = ap.parse_args(argv)
    shards = [pathlib.Path(s) for s in a.shards]
    try:
        layers = [int(x) for x in re.split(r"[\s,;]+", a.layers.strip()) if x]
    except ValueError:
        print(f"expert_parity_shards: bad --layers {a.layers!r}", file=sys.stderr)
        return 2
    try:
        runs, problems = plan(shards, layers)
    except (OSError, ValueError, EOFError) as e:
        print(f"expert_parity_shards: {e}", file=sys.stderr)
        return 2
    for p, ls in runs:
        print(f"expert_parity_shards: {p.name}: layers {' '.join(map(str, ls))}")
    for msg in problems:
        print(f"expert_parity_shards: FAIL {msg}")
    if a.plan:
        return 1 if problems else 0
    if not a.bin:
        print("expert_parity_shards: --bin is required unless --plan", file=sys.stderr)
        return 2
    codes = []
    for p, ls in runs:
        r = subprocess.run([a.bin, str(p), *map(str, ls)])
        print(f"expert_parity_shards: {p.name} -> exit {r.returncode}")
        codes.append(r.returncode)
    if codes and all(c == 77 for c in codes) and not problems:
        return 77
    bad = problems or any(c not in (0, 77) for c in codes) or not codes
    print("expert_parity_shards: " + ("FAIL" if bad else "OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
