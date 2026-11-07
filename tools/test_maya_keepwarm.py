"""CPU-only checks: python -m unittest discover -s tools -p test_maya_keepwarm.py"""
import json
from pathlib import Path
import tempfile
import unittest

import maya_keepwarm
from maya_keepwarm import (COARSE, RES_TABLE, _coalesce_gens, _coarse, _merge_generations,
                           _merge_runs, resolve_pack)


class MergeRunsTests(unittest.TestCase):
    def test_sorts_coalesces_and_drops_empty(self):
        self.assertEqual(_merge_runs([[16, 8], [0, 12], [100, 1], [8, 5]]),
                         [[0, 13], [16, 8], [100, 1]])
        self.assertEqual(_merge_runs([[0, 0], [0, -2]]), [])
        self.assertEqual(_merge_runs([]), [])

    def test_coalesce_touching_spans(self):
        self.assertEqual(_merge_runs([[0, 4], [4, 4]]), [[0, 8]])
        self.assertNotEqual(_merge_runs([[0, 4], [5, 4]]), [[0, 9]])


class CoarseTests(unittest.TestCase):
    def test_snaps_down_to_boundary(self):
        self.assertEqual(_coarse(0), 0)
        self.assertEqual(_coarse(4097), 0)
        self.assertEqual(_coarse(3 * 1024 * 1024), 2 * 1024 * 1024)
        self.assertEqual(COARSE % 4096, 0)   # the window math assumes it


class ResTableTests(unittest.TestCase):
    def test_mincore_bit0(self):
        self.assertEqual(RES_TABLE[0], 0)
        self.assertEqual(RES_TABLE[1], 1)
        self.assertEqual(RES_TABLE[2], 0)    # bit1 set, not resident
        self.assertEqual(RES_TABLE[3], 1)
        self.assertEqual(len(RES_TABLE), 256)


class CoalesceGensTests(unittest.TestCase):
    def test_adjacency_only_within_a_generation(self):
        # same-generation spans fuse, overlap or touching
        self.assertEqual(_coalesce_gens([[0, 4, 0], [4, 4, 0]]), [[0, 8, 0]])
        # disjoint same-generation spans stay apart
        self.assertEqual(sorted(map(tuple, _coalesce_gens([[0, 2, 1], [5, 2, 1]]))),
                         [(0, 2, 1), (5, 2, 1)])
        # a later generation's overlapping/touching spans fuse as one covering
        out = _coalesce_gens([[0, 4, 0], [4, 4, 1], [0, 2, 1], [0, 4, 1]])
        self.assertEqual(sorted(map(tuple, out)), [(0, 4, 0), (0, 8, 1)])


class MergeGenerationsTests(unittest.TestCase):
    BIG = 1 << 47

    def test_resident_is_zero_gen(self):
        kept = _merge_generations([[0, 10], [100, 5]], [], 8, self.BIG)
        self.assertEqual(kept, [[0, 10, 0], [100, 5, 0]])

    def test_prev_ages_by_one_and_survives(self):
        kept = _merge_generations([[0, 10]], [[0, 10, 0], [200, 5, 3]], 8, self.BIG)
        self.assertEqual(kept, [[0, 10, 0], [200, 5, 4]])

    def test_unseen_ranges_die_after_gen_life(self):
        now, prev = [], [[0, 10, 8]]
        self.assertEqual(_merge_generations(now, prev, 8, self.BIG), [])
        # a range seen every snapshot never ages
        kept = [[0, 10, 0]]
        for _ in range(10):
            kept = _merge_generations([[0, 10]], kept, 8, self.BIG)
        self.assertEqual(kept, [[0, 10, 0]])

    def test_partial_coverage_splits_uncovered_part(self):
        kept = _merge_generations([[0, 4]], [[0, 10, 2]], 8, self.BIG)
        self.assertEqual(kept, [[0, 4, 0], [4, 6, 3]])

    def test_drift_histories_coalesce_per_generation(self):
        # two earlier learned fragments that neither overlaps now
        kept = _merge_generations([[40, 10]],
                                  [[0, 10, 4], [10, 4, 4], [60, 10, 2]],
                                  8, self.BIG)
        self.assertEqual(_coalesce_gens(kept), kept)
        self.assertEqual(kept[0], [0, 14, 5])   # 0-9 and 10-13 fused, aged
        self.assertEqual(kept[1], [40, 10, 0])
        self.assertEqual(kept[2], [60, 10, 3])

    def test_cap_trims_oldest_generations_first(self):
        now = [[0, 100]]                        # resident never dropped
        prev = [[0, 100, 5], [1000, 100, 1], [2000, 300, 3]]
        kept = _merge_generations(now, prev, 8, 250 * 1024 * 1024)
        self.assertEqual(kept[0][2], 0)         # the resident range survives
        total = sum(sp[1] for sp in kept)
        self.assertLessEqual(total, 250 * 1024 * 1024 + COARSE)   # COARSE slack
        self.assertEqual(min(sp[2] for sp in kept), 0)

    def test_cap_at_zero_keeps_only_resident(self):
        kept = _merge_generations([[0, 100]], [[1000, 100, 1]], 8, 0)
        self.assertEqual(kept, [[0, 100, 0]])


class ResolvePackTests(unittest.TestCase):
    def test_pack_option_wins_without_a_config(self):
        self.assertEqual(resolve_pack(None, "/p"), "/p")

    def test_explicit_config_peeked_for_pack(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d, "maya-ds4.json")
            cfg.write_text(json.dumps({"args": ["--pack", "P", "--tp", "2"]}))
            self.assertEqual(resolve_pack(str(cfg), None), "P")

    def test_discovery_ignores_configs_without_a_pack(self):
        with tempfile.TemporaryDirectory() as d:
            # the dashboard's jsons and this tool's own state carry no --pack
            Path(d, "maya-keepwarm.json").write_text(json.dumps({"saved": "x"}))
            Path(d, "maya-other.json").write_text(json.dumps({"args": []}))
            Path(d, "maya-ds4.json").write_text(
                json.dumps({"args": ["--pack", "P", "--tp", "2"]}))
            old = maya_keepwarm.ROOT
            maya_keepwarm.ROOT = d
            try:
                self.assertEqual(resolve_pack(None, None), "P")
            finally:
                maya_keepwarm.ROOT = old

    def test_several_or_no_pack_configs_are_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            old = maya_keepwarm.ROOT
            maya_keepwarm.ROOT = d
            try:
                self.assertIsNone(resolve_pack(None, None))   # none
                Path(d, "a.json").write_text(json.dumps({"args": ["--pack", "P"]}))
                Path(d, "b.json").write_text(json.dumps({"args": ["--pack", "Q"]}))
                self.assertIsNone(resolve_pack(None, None))   # several
            finally:
                maya_keepwarm.ROOT = old


if __name__ == "__main__":
    unittest.main()
