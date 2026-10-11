"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order: the base profile's ranking (default: the shipped data/expert-profile.bin), then the pairs your routing
traces used, most frequent first, then every pair still missing, interleaved across the layers.  The base is only
*appended to*, never reordered - and the shipped profile already ranks all 24,576 pairs, so with it a trace is a
no-op.  `--reorder` ranks the traces first and lets the base fill the rest, which is how a workload's trace decides
the top; `--no-base` drops the base entirely.

WATCH THE BASE: `take()` skips a pair it has already ranked, and the shipped base ranks all 24,576, so a plain
`--base` run cannot be moved by any trace - the output is the base again.  `--reorder` (traces first) or
`--no-base` is what makes a trace decide the top.  Which ordering is in use is not cosmetic: the layer-split cost
model reads this RANKING as if it were a frequency curve, and the coverage curve in src/program/generate.cpp
carries the measured hit rates that say how far off that goes (94.9% claimed against 69.2% measured at 5,805 pairs
held, on the IQ3_S 4-way rig).

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--reorder] [--out PATH]
                                 [--n-expert 256]      (a pruned model: the Qwen Coder keeps 256 of 512)

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.

GLM-5.3-Flash (`--glm`: 46 layers - the 45 decoder layers, then the NextN block - of 288 experts; the base and the
output default to data/expert-profile-glm.bin, which setup passes to every GLM config).  Besides `--dump-routing`
traces it reads the GLM engine's text records: a usage file (`STRATA_GLM_USAGE`, a pack's expert_usage.txt:
"layer e:count ..."), a pack's expert_counts.txt ("layer c0 c1 ... c287") and STRATA_GLM_TRACE lines ("layer
expert").  `--equal` weighs every input the same (each one's routes scaled to the same total) instead of summing
their counts, so one long session does not decide the order alone:

    python tools/make_profile.py --glm --no-base --equal TRACE_OR_USAGE ...
"""
import argparse
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
GLM_LAYER, GLM_EXPERT = 46, 288       # GLM-5.3-Flash: 45 decoder layers + the NextN block; 288 routed experts
MAGIC, VERSION = b"STRP", 1


def read_profile(path, n_expert=N_EXPERT, n_layer=N_LAYER):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if ver != VERSION:
        raise SystemExit(f"{path}: a version {ver} profile, this tool writes version {VERSION}")
    if (nl, ne) != (n_layer, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {n_layer}x{n_expert}")
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def read_trace(path, n_expert=N_EXPERT, n_layer=N_LAYER):
    """(layer, k, k expert ids, k weights) records, as `--dump-routing` writes them."""
    blob = Path(path).read_bytes()
    off, freq = 0, defaultdict(int)
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        if k < 0 or off + 8 * k > len(blob):
            break                                       # a record cut short (the engine was stopped mid-write)
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < n_layer and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k                                    # the ids and the weights
    return freq


def is_text(path):
    """A GLM engine text record (usage, counts, STRATA_GLM_TRACE) rather than a binary --dump-routing trace."""
    head = Path(path).read_bytes()[:4096]
    return bool(head) and all(c in b"0123456789 :#\t\r\n.-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()_,/"
                              for c in head)


def read_text(path, n_expert=N_EXPERT, n_layer=N_LAYER):
    """Routes per (layer, expert) from a GLM text record: "layer e:count ..." (a usage file), "layer c0 c1 ..." (a
    pack's expert_counts.txt: one count per expert id) or "layer expert" (STRATA_GLM_TRACE, one route a line)."""
    freq = defaultdict(float)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        layer = int(t[0])
        if not 0 <= layer < n_layer:
            continue
        if any(":" in x for x in t[1:]):
            pairs = ((int(e), float(c)) for e, c in (x.split(":") for x in t[1:]))
        elif len(t) == 2:
            pairs = [(int(t[1]), 1.0)]
        else:
            pairs = enumerate(float(c) for c in t[1:])
        for e, c in pairs:
            if 0 <= e < n_expert and c > 0:
                freq[(layer, e)] += c
    return freq


def write_profile(path, ranked, n_expert=N_EXPERT, n_layer=N_LAYER):
    table = [[-1] * n_expert for _ in range(n_layer)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layer, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layer):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def rank_profile(base_pairs, trace_freq, n_expert=N_EXPERT, no_base=False, reorder=False, n_layer=N_LAYER):
    """The ranked (layer, expert) list and the counts it was built from (`base`, `trace`, `fill`).

    Default: the base's order, then the traces' pairs most frequent first, then the fill.  With `reorder` (or
    `no_base`) the traces' pairs come first, so a base that already ranks every pair no longer hides them; the base
    (unless `no_base`) and then the fill follow.  A base is never reordered against itself - it only ever fills."""
    ranked, seen, counts = [], set(), defaultdict(int)

    def take(pairs, key):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)
                counts[key] += 1

    trace_pairs = [p for p, _ in sorted(trace_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    if no_base or reorder:
        take(trace_pairs, "trace")
    if not no_base:
        take(base_pairs, "base")
    if not (no_base or reorder):
        take(trace_pairs, "trace")
    take(((layer, e) for e in range(n_expert) for layer in range(n_layer)), "fill")
    return ranked, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing (with --glm also the GLM engine's "
                                              "usage files, expert_counts.txt and STRATA_GLM_TRACE records)")
    ap.add_argument("--glm", action="store_true",
                    help=f"GLM-5.3-Flash: {GLM_LAYER} layers of {GLM_EXPERT} experts, data/expert-profile-glm.bin")
    ap.add_argument("--base", default=None, help="ranking to keep first (default: the shipped profile)")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--reorder", action="store_true",
                    help="rank the traces' pairs before the base's (the base, then the fill, follow); needed with a "
                         "base that already ranks every pair, where --base alone cannot change the order")
    ap.add_argument("--equal", action="store_true",
                    help="weigh every input the same (its routes scaled to one total) instead of summing them")
    ap.add_argument("--out", default=None, help="the profile to write (default: the shipped one)")
    ap.add_argument("--n-expert", type=int, default=None, help=f"experts per layer (default {N_EXPERT}; --glm "
                                                               f"{GLM_EXPERT})")
    ap.add_argument("--n-layer", type=int, default=None, help=f"layers (default {N_LAYER}; --glm {GLM_LAYER})")
    a = ap.parse_args()

    shipped = str(ROOT / "data" / ("expert-profile-glm.bin" if a.glm else "expert-profile.bin"))
    ne = a.n_expert or (GLM_EXPERT if a.glm else N_EXPERT)
    nl = a.n_layer or (GLM_LAYER if a.glm else N_LAYER)
    base, out = a.base or shipped, a.out or shipped
    freq = defaultdict(float)
    for t in a.traces:
        f = read_text(t, ne, nl) if a.glm and is_text(t) else read_trace(t, ne, nl)
        total = sum(f.values())
        scale = 1.0 / total if a.equal and total > 0 else 1.0
        for p, c in f.items():
            freq[p] += c * scale
    base_pairs = [] if a.no_base else read_profile(base, ne, nl)
    ranked, counts = rank_profile(base_pairs, freq, ne, no_base=a.no_base, reorder=a.reorder, n_layer=nl)
    write_profile(out, ranked, ne, nl)
    assert read_profile(out, ne, nl) == ranked, "the profile did not survive the round trip"
    print(f"wrote {out}: {len(ranked)} ranked pairs ({counts['base']} from the base, {counts['trace']} from the "
          f"traces, {counts['fill']} filled in)")


if __name__ == "__main__":
    main()
