"""tools/glm_expert_prior.py - the routed-expert frequency prior a GLM pack's fast path warms its tiers with.

    python tools/glm_expert_prior.py <pack_dir> <trace|usage> [<trace|usage> ...]

Each trace is the routing record `STRATA_GLM_TRACE=<file>` writes (one "layer expert" line per routed expert); a
usage file is what the fast path saves after every request when `STRATA_GLM_USAGE=<file>` names one ("layer
e:count e:count ...", the routes of every prompt and answer since it started) - a session over a corpus typical of the
model's use, with STRATA_GLM_USAGE pointing at a fresh file, is a profile.
Writes <pack_dir>/expert_prior.txt: one line per MoE layer, "layer e0 e1 ... e287", the layer's experts in
descending frequency (ties: lower id first; experts never seen keep their id order at the end).  At load the engine
puts each layer's first experts in VRAM and the rest in the pinned RAM tier, so the first request runs warm; the
live LFU takes over from there.  The engine reads the file only without an expert profile (--expert-profile, which
setup passes: tools/make_profile.py --glm writes one); a pack without either warms in id order.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pack = Path(sys.argv[1])
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    n_expert = 0
    for tr in sys.argv[2:]:
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
                        n_expert = max(n_expert, int(e) + 1)
                    continue
                if len(parts) != 2:
                    continue
                e = int(parts[1])
                counts[layer][e] += 1
                n_expert = max(n_expert, e + 1)
    n_expert = max(n_expert, 288)
    out = []
    for layer in sorted(counts):
        c = counts[layer]
        order = sorted(range(n_expert), key=lambda e: (-c.get(e, 0), e))
        out.append(f"{layer} " + " ".join(str(e) for e in order))
    (pack / "expert_prior.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {pack / 'expert_prior.txt'}: {len(out)} layers, {sum(sum(c.values()) for c in counts.values())} visits")
    # the counts themselves ("layer c0 c1 ... c287", by expert id): the engine sizes each layer's VRAM partition from
    # them (a layer whose routing is spread wide gets more slots than one that keeps to a few experts)
    rows = [f"{layer} " + " ".join(str(counts[layer].get(e, 0)) for e in range(n_expert)) for layer in sorted(counts)]
    (pack / "expert_counts.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {pack / 'expert_counts.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
