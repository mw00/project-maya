"""Tests for tools/make_profile.py, without a model: synthetic profiles and routing traces.

    python -m unittest tools.test_make_profile
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402

NE = 4   # a tiny "experts per layer"; the layer count is fixed at MP.N_LAYER


def write_trace(path, records):
    """One record per (layer, [expert ids]); the reader skips k weights after the k ids."""
    with open(path, "wb") as f:
        for layer, experts in records:
            f.write(struct.pack("<ii", layer, len(experts)))
            f.write(struct.pack("<%di" % len(experts), *experts))
            f.write(struct.pack("<%df" % len(experts), *([1.0] * len(experts))))


def full_base():
    """Every pair, in an order that differs from rank_profile's fill (e-major then layer)."""
    return [(layer, e) for layer in range(MP.N_LAYER) for e in range(NE)]


class RankProfile(unittest.TestCase):
    def test_a_complete_base_hides_the_trace(self):
        # the shipped profile's situation: it already ranks every pair, so the trace cannot move anything
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked, base)
        self.assertEqual(counts["trace"], 0)
        self.assertEqual(counts["base"], MP.N_LAYER * NE)

    def test_reorder_puts_the_traces_first(self):
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE, reorder=True)
        self.assertEqual(ranked[:2], [(5, 1), (6, 2)])
        self.assertEqual(set(ranked), set(base))                 # still every pair, once
        self.assertEqual(len(ranked), len(base))
        self.assertEqual(counts["trace"], 2)
        self.assertEqual(counts["base"], MP.N_LAYER * NE - 2)    # the base fills the rest, minus the two

    def test_frequency_and_tie_break_order_the_trace(self):
        freq = defaultdict(int, {(1, 0): 1, (2, 0): 3, (3, 0): 3})
        ranked, _ = MP.rank_profile([], freq, NE, no_base=True)  # (2,0)/(3,0) tie -> the smaller pair first
        self.assertEqual(ranked[:3], [(2, 0), (3, 0), (1, 0)])

    def test_default_appends_only(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(0, 0): 5, (1, 3): 9})
        ranked, _ = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked[:3], [(0, 0), (0, 1), (1, 3)])   # base first; (0,0) already there, not moved

    def test_no_base_uses_the_traces_then_the_fill(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(1, 3): 9, (0, 0): 5})
        ranked, counts = MP.rank_profile(base, freq, NE, no_base=True)
        self.assertEqual(ranked[:2], [(1, 3), (0, 0)])
        self.assertEqual(counts["base"], 0)
        self.assertEqual(len(ranked), MP.N_LAYER * NE)

    def test_reorder_without_traces_is_the_default(self):
        base = full_base()
        self.assertEqual(MP.rank_profile(base, defaultdict(int), NE, reorder=True)[0], base)


class Main(unittest.TestCase):
    def run_main(self, argv):
        old, sys.argv = sys.argv, ["make_profile.py"] + argv
        try:
            MP.main()
        finally:
            sys.argv = old

    def test_end_to_end_reorder_over_a_complete_base(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.bin"
            out = Path(d) / "out.bin"
            trace = Path(d) / "t.bin"
            MP.write_profile(str(base), full_base(), NE)
            write_trace(str(trace), [(5, [1]), (6, [2]), (6, [2]), (6, [2])])
            self.run_main([str(trace), "--base", str(base), "--reorder", "--n-expert", str(NE), "--out", str(out)])
            ranked = MP.read_profile(str(out), NE)
            self.assertEqual(ranked[:2], [(6, 2), (5, 1)])       # most frequent first
            self.assertEqual(set(ranked), set(full_base()))
            # and without --reorder it is the base's order, untouched
            self.run_main([str(trace), "--base", str(base), "--n-expert", str(NE), "--out", str(out)])
            self.assertEqual(MP.read_profile(str(out), NE), full_base())


class Glm(unittest.TestCase):
    """--glm: GLM-5.3-Flash's 46 x 288 layout and the GLM engine's text records."""

    def run_main(self, argv):
        old, sys.argv = sys.argv, ["make_profile.py"] + argv
        try:
            MP.main()
        finally:
            sys.argv = old

    def test_text_records(self):
        with tempfile.TemporaryDirectory() as d:
            usage, counts, trace = Path(d) / "usage.txt", Path(d) / "counts.txt", Path(d) / "trace.txt"
            usage.write_text("# strata expert usage\n3 0:5 7:2\n4 287:9\n")
            counts.write_text("3 1 0 4\n")                    # one count per expert id
            trace.write_text("5 6\n5 6\n44 1\n")            # STRATA_GLM_TRACE: one route a line
            self.assertEqual(dict(MP.read_text(usage, 288, 46)), {(3, 0): 5, (3, 7): 2, (4, 287): 9})
            self.assertEqual(dict(MP.read_text(counts, 288, 46)), {(3, 0): 1, (3, 2): 4})
            self.assertEqual(dict(MP.read_text(trace, 288, 46)), {(5, 6): 2, (44, 1): 1})
            bin_trace = Path(d) / "t.bin"
            write_trace(str(bin_trace), [(3, [1, 2])])
            self.assertTrue(MP.is_text(usage) and MP.is_text(counts) and MP.is_text(trace))
            self.assertFalse(MP.is_text(bin_trace))

    def test_end_to_end_glm(self):
        with tempfile.TemporaryDirectory() as d:
            usage, trace, out = Path(d) / "usage.txt", Path(d) / "t.bin", Path(d) / "glm.bin"
            usage.write_text("10 4:100 5:50\n")
            write_trace(str(trace), [(20, [3])] * 3)
            self.run_main(["--glm", "--no-base", str(usage), str(trace), "--out", str(out)])
            ranked = MP.read_profile(str(out), MP.GLM_EXPERT, MP.GLM_LAYER)
            self.assertEqual(len(ranked), MP.GLM_LAYER * MP.GLM_EXPERT)       # every pair
            self.assertEqual(ranked[:3], [(10, 4), (10, 5), (20, 3)])          # summed counts: the usage leads
            # --equal: each input scaled to the same total - the trace's one pair now outweighs either usage pair
            self.run_main(["--glm", "--no-base", "--equal", str(usage), str(trace), "--out", str(out)])
            self.assertEqual(MP.read_profile(str(out), MP.GLM_EXPERT, MP.GLM_LAYER)[:3], [(20, 3), (10, 4), (10, 5)])
            # the profile the engine reads: a 46 x 288 file is refused as a Qwen-sized one
            with self.assertRaises(SystemExit):
                MP.read_profile(str(out), MP.N_EXPERT)


if __name__ == "__main__":
    unittest.main()
