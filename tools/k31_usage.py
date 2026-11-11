#!/usr/bin/env python3
"""Expert usage counts for maya-k31's placement (--usage), from a REAP report's routing counts.

    k31_usage.py REPORT.json OUT.txt

REPORT.json: {"experts": {"<layer>": {"count": [c0, c1, ...]}}}, the source model's experts in source order.
OUT.txt: one line a MoE layer, "count LAYER c0 c1 ...".  maya-k31 maps a file's experts to source ids through the
file's expert_prune.keep list, so one usage file serves every pruned build of the same source model.
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    rep = json.load(open(sys.argv[1]))
    ex = rep.get("experts")
    if not isinstance(ex, dict) or not ex:
        print(f"k31_usage: {sys.argv[1]} has no experts table", file=sys.stderr)
        return 1
    with open(sys.argv[2], "w") as f:
        f.write(f"# expert routing counts from {sys.argv[1]}\n")
        for layer in sorted(ex, key=int):
            c = ex[layer].get("count")
            if not isinstance(c, list):
                print(f"k31_usage: layer {layer} has no count list", file=sys.stderr)
                return 1
            f.write(f"count {int(layer)} " + " ".join(str(int(v)) for v in c) + "\n")
    print(f"k31_usage: {len(ex)} layers -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
