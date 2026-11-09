#!/usr/bin/env python3
"""tools/strata/llama_ot.py - llama.cpp `-ot` (tensor override) lists that put chosen expert layers on each GPU.

BUILD-PLAN B0.10 run L7: instead of `--fit`, pin explicit per-GPU expert layers.  With `-sm layer -ngl 999` the
attention and dense weights of every layer go to the GPU that owns the layer; these overrides keep the routed
experts (`ffn_{gate,up,down}_exps`) of the listed layers on that GPU and send every other expert tensor to the CPU.
llama.cpp takes the FIRST override whose regex matches a tensor name, so the per-GPU patterns come before
`exps=CPU`.  The tool also prints a `-ts` (tensor split) with the layer counts of each GPU's range, so llama.cpp's
layer split matches the ranges; check the load log (`load_tensors: ... CUDA0 model buffer size`) once.

Three ways to choose the layers:

    # explicit ranges, one per GPU ("-" for none)
    python3 tools/strata/llama_ot.py --layers 3-12,26-35,54-62
    # the first N MoE layers of each GPU's range (split at 26,54 over 79 blocks; dense layers 0-2 skipped)
    python3 tools/strata/llama_ot.py --split 26,54 --count 10,10,9
    # as many layers as fit a per-GPU expert budget, from the B0.7 audit TSV (per-tensor bytes)
    python3 tools/strata/llama_ot.py --split 26,54 --audit-tsv ~/strata-facts/glm53_UD-IQ2_M_audit.tsv --vram-gb 22,22,22

Output (default `--format args`): the arguments ready to paste, one `-ot` per GPU plus `-ot exps=CPU` and `-ts`.
"""
from __future__ import annotations

import argparse
import csv
import re
import shlex
import sys


# ------------------------------------------------------------------ integer range -> regex
def _fill_nines(n: int, k: int) -> int:
    s = str(n)
    return int(s[:-k] + "9" * k) if k < len(s) else int("9" * k)


def _fill_zeros(n: int, k: int) -> int:
    return n - n % 10 ** k


def _split(lo: int, hi: int) -> list[int]:
    stops = {hi}
    k = 1
    stop = _fill_nines(lo, k)
    while lo <= stop < hi:
        stops.add(stop)
        k += 1
        stop = _fill_nines(lo, k)
    k = 1
    stop = _fill_zeros(hi + 1, k) - 1
    while lo < stop <= hi:
        stops.add(stop)
        k += 1
        stop = _fill_zeros(hi + 1, k) - 1
    return sorted(stops)


def _pattern(a: int, b: int) -> str:
    sa, sb = str(a), str(b)
    out, any_digit = "", 0
    for x, y in zip(sa, sb):
        if x == y:
            out += x
        elif x == "0" and y == "9":
            any_digit += 1
        else:
            out += "[%s-%s]" % (x, y) if int(y) - int(x) > 1 else "[%s%s]" % (x, y)
    if any_digit:
        out += r"\d" if any_digit == 1 else r"\d{%d}" % any_digit
    return out


def range_regex(lo: int, hi: int) -> str:
    """Alternation matching exactly the decimal integers lo..hi (no leading zeros): 3-12 -> [3-9]|1[0-2]."""
    if lo > hi or lo < 0:
        raise ValueError("bad range %d-%d" % (lo, hi))
    parts, start = [], lo
    for stop in _split(lo, hi):
        parts.append(_pattern(start, stop))
        start = stop + 1
    return "|".join(parts)


def layers_regex(layers: list[int]) -> str:
    """Alternation over a set of layers, merging consecutive runs."""
    layers = sorted(set(layers))
    runs, i = [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        runs.append(range_regex(layers[i], layers[j]))
        i = j + 1
    return "|".join(runs)


# ------------------------------------------------------------------ layer choice
def gpu_ranges(split: list[int], n_layers: int) -> list[tuple[int, int]]:
    bounds = [0] + split + [n_layers]
    return [(bounds[g], bounds[g + 1] - 1) for g in range(len(bounds) - 1)]


def first_n(rng: tuple[int, int], n: int, first_moe: int) -> list[int]:
    lo = max(rng[0], first_moe)
    return list(range(lo, min(rng[1], lo + n - 1) + 1)) if n > 0 else []


def expert_bytes_from_tsv(path: str) -> dict[int, int]:
    """Per-layer routed expert bytes from a gguf_audit.py TSV."""
    out: dict[int, int] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m = re.match(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", row["name"])
            if m:
                out[int(m.group(1))] = out.get(int(m.group(1)), 0) + int(row["bytes"])
    if not out:
        raise ValueError("no routed expert tensors in %s" % path)
    return out


def fit(rng: tuple[int, int], per_layer: dict[int, int], budget_bytes: float) -> list[int]:
    chosen, used = [], 0
    for i in range(rng[0], rng[1] + 1):
        b = per_layer.get(i)
        if b is None:
            continue
        if used + b > budget_bytes:
            break
        chosen.append(i)
        used += b
    return chosen


def overrides(per_gpu: list[list[int]], device: str = "CUDA") -> list[str]:
    ot = []
    for g, layers in enumerate(per_gpu):
        if layers:
            ot.append(r"blk\.(%s)\.ffn_(up|down|gate)_exps=%s%d" % (layers_regex(layers), device, g))
    ot.append("exps=CPU")
    return ot


def parse_ranges(s: str) -> list[list[int]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part in ("-", ""):
            out.append([])
            continue
        lst: list[int] = []
        for piece in part.split("+"):
            a, _, b = piece.partition("-")
            lst += list(range(int(a), int(b or a) + 1))
        out.append(lst)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layers", help="explicit per-GPU layer ranges, e.g. 3-12,26-35,54-62 ('+' joins, '-' none)")
    ap.add_argument("--split", default="26,54", help="first layer of each later GPU (default 26,54)")
    ap.add_argument("--n-layers", type=int, default=79, help="blocks incl. the MTP block (default 79)")
    ap.add_argument("--first-moe", type=int, default=3, help="first MoE layer (leading dense layers before it)")
    ap.add_argument("--count", help="number of expert layers per GPU, e.g. 10,10,9")
    ap.add_argument("--audit-tsv", help="gguf_audit.py TSV for --vram-gb")
    ap.add_argument("--vram-gb", help="per-GPU expert budget in GB (1e9), e.g. 22,22,22")
    ap.add_argument("--device", default="CUDA", help="buffer type prefix (CUDA0, CUDA1, ...)")
    ap.add_argument("--format", choices=["args", "lines", "json"], default="args")
    a = ap.parse_args(argv)
    split = [int(x) for x in a.split.split(",") if x]
    ranges = gpu_ranges(split, a.n_layers)
    try:
        if a.layers:
            per_gpu = parse_ranges(a.layers)
        elif a.count:
            counts = [int(x) for x in a.count.split(",")]
            if len(counts) != len(ranges):
                raise ValueError("--count has %d entries for %d GPUs" % (len(counts), len(ranges)))
            per_gpu = [first_n(r, n, a.first_moe) for r, n in zip(ranges, counts)]
        elif a.audit_tsv and a.vram_gb:
            per_layer = expert_bytes_from_tsv(a.audit_tsv)
            budgets = [float(x) * 1e9 for x in a.vram_gb.split(",")]
            if len(budgets) != len(ranges):
                raise ValueError("--vram-gb has %d entries for %d GPUs" % (len(budgets), len(ranges)))
            per_gpu = [fit(r, per_layer, b) for r, b in zip(ranges, budgets)]
        else:
            ap.error("give --layers, --count, or --audit-tsv with --vram-gb")
            return 2
    except (ValueError, OSError) as e:
        print("llama_ot: %s" % e, file=sys.stderr)
        return 2
    for g, layers in enumerate(per_gpu):
        if g < len(ranges) and layers and (min(layers) < ranges[g][0] or max(layers) > ranges[g][1]):
            print("llama_ot: warning: GPU%d layers %s fall outside its range %d-%d (llama.cpp then copies the "
                  "experts across GPUs)" % (g, layers, *ranges[g]), file=sys.stderr)
    ot = overrides(per_gpu, a.device)
    ts = ",".join(str(r[1] - r[0] + 1) for r in ranges)
    if a.format == "json":
        import json
        print(json.dumps({"layers": per_gpu, "ot": ot, "ts": ts}))
    elif a.format == "lines":
        for o in ot:
            print(o)
    else:
        print(" ".join("-ot %s" % shlex.quote(o) for o in ot) + " -ts %s" % ts)
    for g, layers in enumerate(per_gpu):
        print("GPU%d: %d expert layers %s" % (g, len(layers), layers_regex(layers) if layers else "-"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
