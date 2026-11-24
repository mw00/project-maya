#!/usr/bin/env python3
"""tools/strata/make_bench_corpus.py - the frozen benchmark prompt corpus (BUILD-PLAN B0.10, B13.2).

Writes data/glm-dsa/bench-corpus/prompt_{4k,8k,32k}.txt: English prose generated deterministically from a fixed seed
and the word lists below (no third-party text, so the corpus is ours to publish; it is dedicated to the public
domain, see the corpus README).  Each file holds about 1.6x its nominal token count (about 4 characters per token for
English with the GLM tokenizer), so tools/strata/bench_http.py can cut an exact token count from it and still start
each run at a different offset (a cold prompt for every run).  MANIFEST.sha256 freezes the bytes; the unit test
regenerates the files and compares.

    python3 tools/strata/make_bench_corpus.py            # (re)write the corpus
    python3 tools/strata/make_bench_corpus.py --check    # exit 1 if the committed files differ
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "glm-dsa" / "bench-corpus"
# nominal tokens -> (seed, characters)
FILES = {"prompt_4k.txt": (4041, 4096 * 4 * 16 // 10), "prompt_8k.txt": (8081, 8192 * 4 * 16 // 10),
         "prompt_32k.txt": (32321, 32768 * 4 * 16 // 10)}

TOPICS = ["the lighthouse keepers of the northern coast", "river barges and the grain trade", "the craft of bookbinding",
          "orchards in a dry summer", "the survey of a mountain pass", "clockmakers and their apprentices",
          "a small observatory on a hill", "the winter market in a harbour town", "salt marshes and their birds",
          "the building of a stone bridge", "letters between two botanists", "a railway timetable in 1890",
          "beekeeping through four seasons", "the repair of an old organ", "mapping the tides of an estuary",
          "a village school and its library", "weavers and the dyes they used", "a long walk along a canal"]
NAMES = ["Ada", "Tomas", "Ingrid", "Moses", "Lena", "Arjun", "Clara", "Pieter", "Noor", "Henrik", "Sofia", "Elias",
         "Maren", "Joaquim", "Ruth", "Kofi", "Yara", "Bram", "Greta", "Ivo"]
PLACES = ["Harrowgate", "Old Ferry", "Kestrel Point", "the Upper Weir", "Marrow Hill", "Saltcote", "the east quay",
          "Linden Row", "Fallowmere", "the north fields", "Brindle Cross", "the long meadow"]
ADJ = ["quiet", "careful", "narrow", "patient", "weathered", "bright", "heavy", "steady", "uneven", "modest",
       "ancient", "practical", "curious", "simple", "sturdy", "faded", "generous", "stubborn", "precise", "hidden"]
NOUNS = ["ledger", "lantern", "workshop", "harvest", "instrument", "tide table", "staircase", "notebook", "cart",
         "window", "measurement", "kettle", "rope", "chart", "season", "doorway", "toolbox", "orchard wall",
         "copper bell", "field report", "wheel", "garden", "inventory", "map", "letter", "pattern", "foundation"]
VERBS = ["recorded", "repaired", "carried", "measured", "described", "rebuilt", "examined", "copied", "polished",
         "mended", "counted", "sketched", "tested", "stored", "explained", "corrected", "compared", "collected"]
ADV = ["slowly", "every morning", "without complaint", "in the late afternoon", "by lamplight", "twice a week",
       "with great care", "before the frost", "when the wind turned", "after the market closed", "in plain words"]
LINKS = ["Afterwards,", "In the same year,", "By contrast,", "For that reason,", "Even so,", "Later,", "At first,",
         "In practice,", "As a rule,", "On most days,", "Meanwhile,", "To be fair,"]
OPINIONS = ["it was slower than anyone had planned", "the result held up better than expected",
            "nobody could agree on the cause", "the numbers did not quite add up", "the work was never finished",
            "the method spread to the neighbouring towns", "the old way turned out to be the better one",
            "the cost was paid over several years", "the records survived only in part"]


class Gen:
    """Deterministic choices built on random.random() only (its sequence is stable across Python versions)."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def pick(self, seq):
        return seq[int(self.r.random() * len(seq))]

    def num(self, lo: int, hi: int) -> int:
        return lo + int(self.r.random() * (hi - lo + 1))

    def sentence(self) -> str:
        k = self.num(0, 7)
        if k == 0:
            return "%s %s the %s %s %s." % (self.pick(LINKS), self.pick(NAMES), self.pick(ADJ), self.pick(NOUNS),
                                              self.pick(ADV))
        if k == 1:
            return "At %s the %s was %s, and %s %s it %s." % (self.pick(PLACES), self.pick(NOUNS), self.pick(ADJ),
                                                              self.pick(NAMES), self.pick(VERBS), self.pick(ADV))
        if k == 2:
            return "In %d, %s %s %d pages of the %s near %s." % (self.num(1820, 1960), self.pick(NAMES), self.pick(VERBS),
                                                    self.num(2, 480), self.pick(NOUNS), self.pick(PLACES))
        if k == 3:
            return "%s wrote that %s, although the %s %s had been %s %s." % (
                self.pick(NAMES), self.pick(OPINIONS), self.pick(ADJ), self.pick(NOUNS), self.pick(VERBS),
                self.pick(ADV))
        if k == 4:
            return "The %s measured %d.%d metres from end to end, which %s %s with a %s %s." % (
                self.pick(NOUNS), self.num(1, 240), self.num(0, 9), self.pick(NAMES), self.pick(VERBS),
                self.pick(ADJ), self.pick(NOUNS))
        if k == 5:
            return "\"Keep the %s %s,\" %s said, \"and the %s will look after itself.\"" % (
                self.pick(NOUNS), self.pick(ADJ), self.pick(NAMES), self.pick(NOUNS))
        if k == 6:
            return "%s, %s and %s %s the %s %s; %s." % (self.pick(NAMES), self.pick(NAMES), self.pick(NAMES),
                                                        self.pick(VERBS), self.pick(NOUNS), self.pick(ADV),
                                                        self.pick(OPINIONS))
        return "Nobody at %s %s the %s %s, yet %s." % (self.pick(PLACES), self.pick(VERBS), self.pick(ADJ),
                                                       self.pick(NOUNS), self.pick(OPINIONS))

    def paragraph(self) -> str:
        return " ".join(self.sentence() for _ in range(self.num(4, 8)))

    def text(self, n_chars: int) -> str:
        parts, total, ch = [], 0, 0
        while total < n_chars:
            ch += 1
            head = "Chapter %d: %s" % (ch, self.pick(TOPICS).capitalize())
            parts.append(head)
            total += len(head) + 2
            for _ in range(self.num(4, 9)):
                p = self.paragraph()
                parts.append(p)
                total += len(p) + 2
        return "\n\n".join(parts) + "\n"


def build() -> dict[str, bytes]:
    return {name: Gen(seed).text(n).encode("utf-8") for name, (seed, n) in FILES.items()}


def manifest(files: dict[str, bytes]) -> str:
    return "".join("%s  %s\n" % (hashlib.sha256(b).hexdigest(), n) for n, b in sorted(files.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    files = build()
    man = manifest(files)
    if a.check:
        bad = [n for n, b in files.items() if not (out / n).exists() or (out / n).read_bytes() != b]
        if not (out / "MANIFEST.sha256").exists() or (out / "MANIFEST.sha256").read_text() != man:
            bad.append("MANIFEST.sha256")
        for n in bad:
            print("differs: %s" % (out / n))
        return 1 if bad else 0
    out.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (out / n).write_bytes(b)
    (out / "MANIFEST.sha256").write_text(man)
    print(man, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
