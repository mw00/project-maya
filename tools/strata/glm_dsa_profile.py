#!/usr/bin/env python3
"""tools/strata/glm_dsa_profile.py - the expert profile (--expert-profile) of a glm-dsa pack (glm-dsa W10).

Maya v1.1.3's expert profile (glm-dsa's STRP file) ranks every (layer, expert) pair; the engine gives each MoE layer
its VRAM slots in that order and warms the tiers by it (fast_setup / fast_warm), in place of the pack's
expert_counts.txt / expert_prior.txt.  The profile Maya ships (data/expert-profile-glm.bin) is GLM-5.3-Flash's
(46 layers x 288 experts); GLM-5.3 744B (glm-dsa) needs its own: the trunk's layers, then the NextN block, by the
routed experts (78 + 1 = 79 layers x 256 experts on the published GGUF).  This tool writes it beside the pack
(<pack>/expert-profile.bin), where setup finds it (tools/strata/maya_sync.py profile_for):

    python3 tools/strata/glm_dsa_profile.py <pack_dir> [INPUT ...] [--out PATH] [--equal] [--base PROFILE]

INPUT: what tools/make_profile.py --glm reads - `--dump-routing` traces (binary), usage files (STRATA_GLM_USAGE,
"layer e:count ..."), expert_counts.txt ("layer c0 c1 ... c255", tools/glm_expert_prior.py) and STRATA_GLM_TRACE
records ("layer expert").  Without any INPUT the pack's own expert_counts.txt is used.  The ranking: the inputs'
pairs, most routed first (--equal: each input weighed the same), then --base's order (a previous profile of the same
layout, e.g. one --expert-profile-save learned), then every pair still missing, interleaved across the layers.  The
dense-lead layers (no routed experts) are ranked last: their pairs serve nothing, the engine skips them.

The layout comes from the pack (index.txt's "# glm-dsa: trunk T, dense lead D, experts N" line); --n-layer /
--n-expert override it for a pack without the line.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # tools/: make_profile
sys.path.insert(0, str(HERE))                 # tools/strata/: maya_sync
import make_profile as MP                      # noqa: E402
import maya_sync                               # noqa: E402


def dense_lead(pack: Path) -> int:
    import re
    try:
        for line in (pack / "index.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("#"):
                break
            m = re.search(r"dense lead\s+(\d+)", line)
            if m and "glm-dsa" in line:
                return int(m.group(1))
    except OSError:
        pass
    return 0


def build(pack: Path, inputs: list[Path], n_layer: int, n_expert: int, lead: int, equal: bool = False,
          base: Path | None = None) -> list[tuple[int, int]]:
    """The ranked (layer, expert) pairs (every pair once)."""
    freq: dict[tuple[int, int], float] = defaultdict(float)
    for t in inputs:
        f = MP.read_text(t, n_expert, n_layer) if MP.is_text(t) else MP.read_trace(t, n_expert, n_layer)
        total = sum(f.values())
        scale = 1.0 / total if equal and total > 0 else 1.0
        for p, c in f.items():
            freq[p] += c * scale
    base_pairs = MP.read_profile(base, n_expert, n_layer) if base is not None else []
    ranked, _ = MP.rank_profile(base_pairs, freq, n_expert, no_base=base is None, reorder=True, n_layer=n_layer)
    # the dense lead's pairs last (routes never name them; the engine has no slots there)
    moe = [p for p in ranked if p[0] >= lead]
    return moe + [p for p in ranked if p[0] < lead]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack", type=Path, help="the glm-dsa pack (tools/strata/glm_dsa_pack.py's output)")
    ap.add_argument("inputs", nargs="*", type=Path, help="traces, usage files, expert_counts.txt, STRATA_GLM_TRACE")
    ap.add_argument("--out", type=Path, default=None, help=f"default <pack>/{maya_sync.PACK_PROFILE}")
    ap.add_argument("--equal", action="store_true", help="weigh every input the same")
    ap.add_argument("--base", type=Path, default=None, help="a previous profile of the same layout, ranked after "
                                                            "the inputs")
    ap.add_argument("--n-layer", type=int, default=None, help="profile layers (default: the pack's trunk + 1)")
    ap.add_argument("--n-expert", type=int, default=None, help="experts per layer (default: the pack's)")
    a = ap.parse_args(argv)
    lay = maya_sync.pack_layout(a.pack)
    nl = a.n_layer or (lay[0] if lay else 0)
    ne = a.n_expert or (lay[1] if lay else 0)
    if nl <= 0 or ne <= 0:
        print(f"{a.pack}: not a glm-dsa pack (no '# glm-dsa: trunk ..' line in index.txt); pass --n-layer and "
              "--n-expert", file=sys.stderr)
        return 2
    inputs = list(a.inputs)
    if not inputs:
        counts = a.pack / "expert_counts.txt"
        if not counts.exists():
            print(f"{a.pack}: no inputs and no expert_counts.txt (tools/glm_expert_prior.py writes one)",
                  file=sys.stderr)
            return 2
        inputs = [counts]
    for t in inputs:
        if not t.exists():
            print(f"{t}: missing", file=sys.stderr)
            return 2
    ranked = build(a.pack, inputs, nl, ne, dense_lead(a.pack), a.equal, a.base)
    out = a.out or a.pack / maya_sync.PACK_PROFILE
    MP.write_profile(out, ranked, ne, nl)
    assert MP.read_profile(out, ne, nl) == ranked, "the profile did not survive the round trip"
    print(f"wrote {out}: {nl} layers x {ne} experts, {len(ranked)} ranked pairs from {len(inputs)} input(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
