"""Refresh DS4 raw uint32 expert counts from independent before/after snapshots.

See docs/DS4_PROFILE_REFRESH.md. This is not the STRP ranking format used by
make_profile.py. No engine defaults or live files are changed.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

MAX_COUNT = 2**32 - 1


def read_counts(path, layers, experts):
    data = Path(path).read_bytes()
    if len(data) != layers * experts * 4:
        raise ValueError(f"{path}: expected {layers * experts * 4} bytes, got {len(data)}")
    return list(struct.unpack(f"<{layers * experts}I", data))


def refresh(historical, pairs, experts, recent_weight):
    if not math.isfinite(recent_weight) or not 0 < recent_weight <= 1:
        raise ValueError("recent weight must be finite and in (0, 1]")
    recent = [0] * len(historical)
    if not pairs:
        raise ValueError("at least one independent before/after pair is required")
    for before, after in pairs:
        if len(before) != len(historical) or len(after) != len(historical):
            raise ValueError("snapshot geometry differs")
        for i, (b, a) in enumerate(zip(before, after)):
            if a < b:
                raise ValueError(f"counter decreased at index {i}; reset or overflow is unsupported")
            recent[i] += a - b
    result = []
    for start in range(0, len(historical), experts):
        old = historical[start:start + experts]
        new = recent[start:start + experts]
        total, old_total = sum(new), sum(old)
        if not total:
            raise ValueError(f"no recent observations in layer {start // experts}")
        # Keep the recent count scale; normalize the prior separately per layer.
        scale = total * ((1 - recent_weight) / recent_weight)
        for h, n in zip(old, new):
            value = round(n + (h / old_total * scale if old_total else 0))
            if not 0 <= value <= MAX_COUNT:
                raise ValueError("blended count exceeds uint32; shorten calibration")
            result.append(value)
    return result, recent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                   metavar=("BEFORE", "AFTER"), help="independent calibration snapshots; repeat to combine runs")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--recent-weight", type=float, default=0.8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    receipt = Path(str(args.out) + ".json")
    try:
        if args.layers <= 0 or args.experts <= 0:
            raise ValueError("geometry must be positive")
        if args.out.exists() or receipt.exists():
            raise ValueError("output or receipt already exists; choose a fresh output path")
        paths = [args.historical] + [x for pair in args.pair for x in pair]
        counts = [read_counts(x, args.layers, args.experts) for x in paths]
        result, recent = refresh(counts[0], list(zip(counts[1::2], counts[2::2])),
                                 args.experts, args.recent_weight)
        blob = struct.pack(f"<{len(result)}I", *result)
        proof = dict(format="DS4 raw little-endian uint32 counts", layers=args.layers,
                     experts=args.experts, recent_weight=args.recent_weight,
                     historical_total=sum(counts[0]), recent_total=sum(recent),
                     output_total=sum(result), inputs=[dict(path=str(x),
                     sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths],
                     output_sha256=hashlib.sha256(blob).hexdigest())
        # Exclusive creation also protects against accidental overwrite of a live profile.
        with args.out.open("xb") as f:
            f.write(blob)
        with receipt.open("x") as f:
            json.dump(proof, f, indent=2)
            f.write("\n")
    except (ValueError, OSError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
