"""tools/glm_expert_prior.py - the routed-expert frequency prior a GLM pack's fast path warms its tiers with.

    python tools/glm_expert_prior.py [--n-expert N] <pack_dir> <trace|usage> [<trace|usage> ...]

Each trace is the routing record `STRATA_GLM_TRACE=<file>` writes (one "layer expert" line per routed expert); a
usage file is what the fast path saves after every request when `STRATA_GLM_USAGE=<file>` names one ("layer
e:count e:count ...", the routes of every prompt and answer since it started) - a session over a corpus typical of the
model's use, with STRATA_GLM_USAGE pointing at a fresh file, is a profile.

The expert count is the model's: --n-expert, else the "(n_expert N" of the pack's native_experts.txt header
(tools/iq_pack.py writes it: 288 on GLM-5.x-Flash, 256 on GLM-5.3 744B), else the largest id seen + 1.  glm-dsa B8.5:
it used to pad every pack to 288, which on glm-dsa listed 32 experts that do not exist.

Writes into <pack_dir>:
  expert_prior.txt   one line per MoE layer, "layer e0 e1 ... e(N-1)", the layer's experts in descending frequency
                     (ties: lower id first; experts never seen keep their id order at the end).  At load the engine
                     puts each layer's first experts in VRAM and the rest in the pinned RAM tier, so the first request
                     runs warm; the live LFU takes over from there.  The engine reads the file only without an expert
                     profile (--expert-profile, which setup passes: tools/make_profile.py --glm writes one); a pack
                     without either warms in id order.
  expert_counts.txt  "layer c0 c1 ... c(N-1)" by expert id: the engine sizes each layer's VRAM partition from them.
  hit_curve.txt      the routing skew h(f): with each layer holding its most frequent ceil(f N) experts, the share of
                     all routed visits they serve (per-layer static top-k, every layer's visits pooled), for
                     f = 0.00 .. 1.00 in steps of 0.01 - "f h" lines after a "#" header.
                     docs/strata/plan/model/model.py reads it (STRATA_HIT_CURVE=<file>, or the copy at
                     docs/strata/plan/model/hit_curve_volta3.txt) in place of its Flash-calibrated CURVE.
"""
from __future__ import annotations

import argparse
import collections
import math
import re
import sys
from pathlib import Path


def pack_n_expert(pack: Path) -> int | None:
    """The n_expert iq_pack.py recorded in the pack's native_experts.txt header, or None."""
    p = pack / "native_experts.txt"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        head = f.readline()
    m = re.search(r"\(n_expert (\d+)", head)
    return int(m.group(1)) if m else None


def read_counts(paths: list[str]) -> tuple[dict[int, collections.Counter], int]:
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    seen = 0
    for tr in paths:
        with open(tr, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2 or line.startswith("#"):
                    continue
                layer = int(parts[0])
                if ":" in parts[1]:                     # a usage line: "layer e:count ..."
                    for tok in parts[1:]:
                        e, n = tok.split(":")
                        counts[layer][int(e)] += int(n)
                        seen = max(seen, int(e) + 1)
                    continue
                if len(parts) != 2:
                    continue
                e = int(parts[1])
                counts[layer][e] += 1
                seen = max(seen, e + 1)
    return counts, seen


def hit_curve(counts: dict[int, collections.Counter], n_expert: int, steps: int = 100) -> list[tuple[float, float]]:
    """h(f) for f = 0, 1/steps, ..., 1: the visits served by each layer's top ceil(f * n_expert) experts."""
    total = sum(sum(c.values()) for c in counts.values())
    if total == 0 or n_expert <= 0:
        return [(i / steps, i / steps) for i in range(steps + 1)]
    ranked = {layer: sorted(c.values(), reverse=True) for layer, c in counts.items()}
    prefix = {}
    for layer, r in ranked.items():
        acc, p = 0, [0]
        for v in r:
            acc += v
            p.append(acc)
        prefix[layer] = p
    out = []
    for i in range(steps + 1):
        f = i / steps
        k = min(n_expert, math.ceil(f * n_expert - 1e-9))
        served = sum(p[min(k, len(p) - 1)] for p in prefix.values())
        out.append((f, served / total))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-expert", type=int, default=None, help="experts per layer (default: the pack's, else seen)")
    ap.add_argument("pack")
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args(argv)
    pack = Path(a.pack)
    counts, seen = read_counts(a.traces)
    n_expert = a.n_expert or pack_n_expert(pack) or seen
    if seen > n_expert:
        print(f"error: expert id {seen - 1} in the traces, but the model has {n_expert} experts a layer",
              file=sys.stderr)
        return 1
    out = []
    for layer in sorted(counts):
        c = counts[layer]
        order = sorted(range(n_expert), key=lambda e: (-c.get(e, 0), e))
        out.append(f"{layer} " + " ".join(str(e) for e in order))
    (pack / "expert_prior.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    visits = sum(sum(c.values()) for c in counts.values())
    print(f"wrote {pack / 'expert_prior.txt'}: {len(out)} layers x {n_expert} experts, {visits} visits")
    rows = [f"{layer} " + " ".join(str(counts[layer].get(e, 0)) for e in range(n_expert)) for layer in sorted(counts)]
    (pack / "expert_counts.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {pack / 'expert_counts.txt'}")
    curve = hit_curve(counts, n_expert)
    lines = [f"# hit curve h(f): per-layer static top-ceil(f*{n_expert}) experts, {len(counts)} layers, "
             f"{visits} visits (tools/glm_expert_prior.py)"]
    lines += [f"{f:.2f} {h:.6f}" for f, h in curve]
    (pack / "hit_curve.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pts = ", ".join(f"h({f:.2f}) = {h:.3f}" for f, h in curve if round(f * 100) in (5, 10, 20, 34, 50, 75))
    print(f"wrote {pack / 'hit_curve.txt'}: {pts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
