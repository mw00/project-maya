"""tools/mm3/kv_gate.py - the quality gate of maya-mm3's compact K / V caches (--kv-cache): which is the smallest format
that keeps MiniMax-M3.1's answers.

Two checks, both through the engine's --score mode (src/mm3/mm3_main.cpp: KL over a reference's top 64 log-probs plus
the rest of the vocabulary as one bucket, the same top token, the target's NLL - on the assistant tokens and on all):

  1. against the exact reference (calib_mm31.py's ref_topk.npz: the unquantized model's held-out top 64, rows of
     eval_mm31.npy, 2048 tokens each): BF16 K / V first, then each format - a format may add little to the quant's
     own KL;
  2. long prompts against the engine's own BF16 K / V run: the held-out rows joined end to end to 4K, 16K and 64K
     tokens, the BF16 run writing its top 64 (--write-ref), each format scored against it - the cache's rounding
     alone, at the depths where attention reads the most rows.

    python tools/mm3/kv_gate.py make --ref ~/mm31/ref/ref_topk.npz --rows ~/mm31/calib/eval_mm31.npy --out DIR
    python tools/mm3/kv_gate.py run --exe build/maya-mm3 --pack <model>/maya-mm3 --dir DIR \\
        [--formats bf16,int8,fp8,fp4,int8,fp4 ...] [--lengths 4096,16384,65536] [--rows 8] [-- engine args]

`run` prints a table and the recommendation: the smallest format (bytes a position) whose long-prompt KL against BF16
is at most --max-kl (0.01) with the same top token on at least --min-same (98%) of the assistant tokens, and whose KL
against the reference adds at most --max-add (10%) to BF16's.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import subprocess
import sys

import numpy as np

# the engine's formats: name -> bytes a row of 128 (codes and scales; mm3_kernels.cuh's kv_row_bytes)
ROW = {"bf16": 256, "int8": 132, "fp8": 132, "fp4": 72}


def write_ref(path: pathlib.Path, ids: np.ndarray, rid: np.ndarray | None, rlp: np.ndarray | None,
              mask: np.ndarray) -> None:
    """The engine's MM3R file: rows of ids [T], the reference's top-K ids / log-probs [T][K] (K 0: none), mask [T-1]."""
    rows, T = ids.shape
    K = 0 if rid is None else rid.shape[2]
    with open(path, "wb") as f:
        f.write(b"MM3R")
        f.write(struct.pack("<3i", rows, T, K))
        for r in range(rows):
            f.write(ids[r].astype("<i4").tobytes())
            if K:
                f.write(rid[r].astype("<i4").tobytes())
                f.write(rlp[r].astype("<f4").tobytes())
            f.write(mask[r].astype(np.uint8).tobytes())


def make(a) -> int:
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ref = np.load(pathlib.Path(a.ref).expanduser())
    rows = np.load(pathlib.Path(a.rows).expanduser()).astype(np.int32)
    rid, rlp, asst = ref["ids"], ref["logprob"], ref["asst"].astype(bool)
    n, T = rows.shape
    if rid.shape[:2] != (n, T) or asst.shape != (n, T - 1):
        sys.exit(f"the reference's rows {rid.shape[:2]} / masks {asst.shape} do not match the rows {rows.shape}")
    write_ref(out / "ref.bin", rows, rid, rlp, asst)
    print(f"{out / 'ref.bin'}: {n} rows of {T} tokens, the reference's top {rid.shape[2]}, "
          f"{int(asst.sum())} assistant tokens")
    # the long prompts: the rows end to end (a row's first token predicted from the row before is no assistant token)
    flat = rows.reshape(-1)
    m = np.concatenate([np.append(asst[r], False) for r in range(n)])[:-1]
    for L in [int(x) for x in a.lengths.split(",")]:
        if L > flat.size:
            print(f"(no {L}-token prompt: the rows hold {flat.size} tokens)")
            continue
        write_ref(out / f"long_{L}.bin", flat[None, :L], None, None, m[None, :L - 1])
        print(f"{out / f'long_{L}.bin'}: 1 prompt of {L} tokens, {int(m[:L - 1].sum())} assistant tokens")
    return 0


SCORE = re.compile(r"^SCORE all (.*)$")


def score(exe: str, pack: str, ref: pathlib.Path, fmt: str, ctx: int, extra: list[str], write: pathlib.Path | None):
    """One engine run: {key: value} of its SCORE all line."""
    cmd = [exe, "--pack", pack, "--score", str(ref), "--kv-cache", fmt, "--max-context", str(ctx)] + extra
    if write is not None:
        cmd += ["--write-ref", str(write)]
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("SCORE row"):
            print("    " + line, flush=True)
        if (mt := SCORE.match(line)):
            return {k: float(v) for k, v in (kv.split("=") for kv in mt.group(1).split())}
    sys.stderr.write(p.stderr[-4000:])
    raise SystemExit(f"{fmt}: the engine gave no score (exit {p.returncode})")


def row_bytes(fmt: str) -> int:
    k, _, v = fmt.partition(",")
    return ROW[k] + ROW[v or k]


def run(a) -> int:
    d = pathlib.Path(a.dir).expanduser()
    fmts = [f for f in a.formats.split(";")] if ";" in a.formats else a.formats.split(" ")
    extra = a.engine
    res: dict = {"ref": {}, "long": {}}
    # 1. against the exact reference (the first --rows rows)
    ref = d / "ref.bin"
    if a.rows > 0 and ref.exists():
        sub = d / f"ref_{a.rows}.bin"
        with open(ref, "rb") as f, open(sub, "wb") as g:
            magic = f.read(4)
            rows, T, K = struct.unpack("<3i", f.read(12))
            n = min(rows, a.rows)
            per = 4 * T + 8 * T * K + (T - 1)
            g.write(magic + struct.pack("<3i", n, T, K) + f.read(per * n))
        for fmt in fmts:
            res["ref"][fmt] = score(a.exe, a.pack, sub, fmt, max(T, 4096), extra, None)
    # 2. long prompts against the BF16 run
    for L in [int(x) for x in a.lengths.split(",")]:
        lp = d / f"long_{L}.bin"
        if not lp.exists():
            continue
        own = d / f"long_{L}_bf16.bin"
        score(a.exe, a.pack, lp, "bf16", L, extra, own)
        res["long"][L] = {}
        for fmt in fmts:
            if fmt == "bf16":
                continue
            res["long"][L][fmt] = score(a.exe, a.pack, own, fmt, L, extra, None)
    (d / "kv_gate.json").write_text(json.dumps(res, indent=1))
    # the table and the pick
    print("\nformat      bytes/pos-layer  ref asst KL  ref same   " +
          "  ".join(f"{L // 1024}K KL vs bf16 / same" for L in res["long"]))
    ok = []
    base = res["ref"].get("bf16", {}).get("asst_kl")
    for fmt in fmts:
        r = res["ref"].get(fmt, {})
        cells = []
        good = True
        for L in res["long"]:
            s = res["long"][L].get(fmt)
            if s is None:
                cells.append("      -        ")
                continue
            cells.append(f"{s['asst_kl']:.4f} / {100 * s['asst_same']:.1f}%")
            good &= s["asst_kl"] <= a.max_kl and s["asst_same"] >= a.min_same
        if base is not None and r:
            good &= r["asst_kl"] <= base * (1 + a.max_add)
        print(f"{fmt:10s}  {4 * row_bytes(fmt):6d}           "
              f"{r.get('asst_kl', float('nan')):.4f}      {100 * r.get('asst_same', float('nan')):.1f}%     " +
              "   ".join(cells))
        if good and fmt != "bf16":
            ok.append(fmt)
    if ok:
        pick = min(ok, key=row_bytes)
        print(f"\nsmallest format within the gate: {pick} ({4 * row_bytes(pick)} bytes a position and layer, BF16 "
              f"{4 * row_bytes('bf16')})")
    else:
        print("\nno compact format within the gate: keep bf16 (or loosen --max-kl / --min-same)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--ref", required=True)
    m.add_argument("--rows", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--lengths", default="4096,16384,65536")
    r = sp.add_parser("run")
    r.add_argument("--exe", required=True)
    r.add_argument("--pack", required=True)
    r.add_argument("--dir", required=True)
    r.add_argument("--formats", default="bf16 int8 fp8 fp4 int8,fp4 fp8,fp4")
    r.add_argument("--lengths", default="4096,16384,65536")
    r.add_argument("--rows", type=int, default=8)
    r.add_argument("--max-kl", type=float, default=0.01)
    r.add_argument("--min-same", type=float, default=0.98)
    r.add_argument("--max-add", type=float, default=0.10)
    r.add_argument("engine", nargs=argparse.REMAINDER, help="-- then the engine's own arguments (--gpus, --tp, ...)")
    a = ap.parse_args()
    if a.cmd == "run":
        a.engine = [x for x in a.engine if x != "--"]
        return run(a)
    return make(a)


if __name__ == "__main__":
    sys.exit(main())
