"""tools/ds4/profile.py - DeepSeek-V4.1's starting expert profile (data/expert-profile-ds4.bin) and checks on profiles.

maya-ds4 keeps its routed-expert hit counts in <pack>/expert-profile.bin (uint32 [layer][expert], the trunk's layers
then the drafter's when it was loaded): read at load to place the most-routed experts in VRAM and read the next ones
ahead, counted on as it runs and saved after each answer.  A pack without one starts with every count at zero - the
first answers then run from an arbitrary placement until the expert cache has moved the used experts in.  The setup
copies the shipped profile into a pack that has none (maya.py ds4_seed_profile).

    make     the shipped profile from engine-saved ones (each an engine run over representative text: maya-ds4
             --profile-run CORPUS.ids --profile-out RUN.bin, a line of token ids each - that run's counts alone, the
             pack's profile left as it was; or a run with --expert-profile <new file>): the trunk's layers, every
             input weighing the same, each layer's counts scaled to --tokens tokens' worth (k picks a token) - small,
             so what a PC routes itself soon outweighs it
    compare  how much two profiles agree: per layer, the share of A's top --top experts also in B's top --top

    maya-ds4 --pack <pack> --profile-run code.ids --profile-out code.bin   (one corpus a run; see docs/SETTINGS.md)
    python tools/ds4/profile.py make --pack <pack> --out data/expert-profile-ds4.bin RUN.bin [RUN.bin ...]
    python tools/ds4/profile.py compare --pack <pack> A.bin B.bin [--top 64]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

TOKENS = 2048          # the shipped profile's weight: this many tokens' worth of picks a layer


def geometry(pack: Path) -> tuple:
    """(trunk layers, routed experts, picks a token) from the pack's ds4.txt."""
    kv = {}
    for line in (pack / "ds4.txt").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition(" ")
        kv[k] = v.strip()
    return int(kv["num_hidden_layers"]), int(kv["n_routed_experts"]), int(kv["num_experts_per_tok"])


def read(path: Path, nl: int, ne: int) -> list:
    """A profile's trunk layers: [layer][expert] counts (a file with the drafter's layers after them too)."""
    blob = Path(path).read_bytes()
    n = len(blob) // 4
    if n < nl * ne or n % ne:
        raise SystemExit(f"{path}: {n} counts, not {nl} layers x {ne} experts (and the drafter's)")
    v = struct.unpack(f"<{nl * ne}I", blob[:nl * ne * 4])
    return [list(v[i * ne:(i + 1) * ne]) for i in range(nl)]


def make(inputs: list, nl: int, ne: int, k: int, tokens: int = TOKENS) -> list:
    """The inputs' layers summed with each input weighing the same (its layer's counts over their total), each
    layer scaled to tokens x k counts; an expert used at all keeps at least 1 (the ranking's tail stays)."""
    out = []
    for il in range(nl):
        acc = [0.0] * ne
        for p in inputs:
            tot = sum(p[il])
            if tot:
                for e in range(ne):
                    acc[e] += p[il][e] / tot
        s = sum(acc)
        row = [0] * ne
        if s > 0:
            for e in range(ne):
                if acc[e] > 0:
                    row[e] = max(1, round(acc[e] / s * tokens * k))
        out.append(row)
    return out


def top(row: list, n: int) -> set:
    return set(sorted(range(len(row)), key=lambda e: (-row[e], e))[:n])


def agree(a: list, b: list, n: int) -> list:
    """Per layer: the share of a's n most-routed experts that are among b's n most-routed."""
    return [len(top(ra, n) & top(rb, n)) / n for ra, rb in zip(a, b)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--pack", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--tokens", type=int, default=TOKENS)
    m.add_argument("inputs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("--pack", required=True)
    c.add_argument("--top", type=int, default=64)
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    nl, ne, k = geometry(Path(a.pack))
    if a.cmd == "make":
        prof = make([read(Path(p), nl, ne) for p in a.inputs], nl, ne, k, a.tokens)
        Path(a.out).write_bytes(struct.pack(f"<{nl * ne}I", *[v for row in prof for v in row]))
        print(f"{a.out}: {nl} layers x {ne} experts, {a.tokens} tokens' worth a layer, from {len(a.inputs)} profile(s)")
        return 0
    s = agree(read(Path(a.a), nl, ne), read(Path(a.b), nl, ne), a.top)
    print(f"top {a.top} of {ne}: mean {sum(s) / len(s):.3f}, min {min(s):.3f} (layer {s.index(min(s))}), "
          f"per layer " + " ".join(f"{x:.2f}" for x in s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
