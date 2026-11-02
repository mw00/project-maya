"""MAYA_METAL_RESTORE_POINTS: where the server asks a Metal engine to keep restore points (its GEN line's marks=
and shared=), computed from the prompt's ids as the GLM and MiniMax templates write them - no GPU, no engine:
the points functions and the settings, on synthetic ids, and the keys they put on the line.  The Service-level
gate (the setting on, an engine that advertises restore points, a GLM-shaped render) is exercised by hand
after the engine side lands; these are the parts that must not move under it.

    python -m unittest serve.test_metal_restore_points -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine, metal_restore_points, metal_restore_point_settings  # noqa: E402
import maya_names  # noqa: E402  (tools/, on the path server.py set)

MON = maya_names.read_as("MAYA_METAL_RESTORE_POINTS")
MTAIL = maya_names.read_as("MAYA_METAL_RESTORE_TAIL")

U, S, O, B = 9001, 9002, 9003, 9004      # <|user|>, <|system|>, <|observation|>, ]~b]


def ids(n, at=None):
    out = [7] * n
    for pos, tok in (at or {}).items():
        out[pos] = tok
    return out


class GlmPoints(unittest.TestCase):
    def test_shared_marks_tail(self):
        # system + tools to 1500, a user turn, an answer, a tool result's turn at 3000, then text to 4000
        pts = metal_restore_points(ids(4000, {1500: U, 2600: U, 3000: O}),
                                   user_id=U, system_id=S, obs_id=O)
        self.assertEqual(pts, {"shared": 1500, "marks": [3000], "tail": 4000 - 512})

    def test_a_late_system_turn_is_a_mark(self):
        pts = metal_restore_points(ids(3000, {1200: U, 2900: S}), user_id=U, system_id=S, obs_id=O)
        self.assertEqual(pts, {"shared": 1200, "marks": [2900]})

    def test_points_under_least_are_left_out(self):
        self.assertEqual(metal_restore_points(ids(300, {10: U}), user_id=U, system_id=S), {})
        # no shared either when the first turn is the prompt's own start: the tail alone, as DeepSeek's
        self.assertEqual(metal_restore_points(ids(4000, {10: U}), user_id=U, system_id=S),
                         {"tail": 4000 - 512})

    def test_no_user_id_no_points(self):
        self.assertEqual(metal_restore_points([], user_id=U), {})
        self.assertEqual(metal_restore_points(ids(4000), user_id=None), {})


class MinimaxPoints(unittest.TestCase):
    def test_turns(self):
        # a BOS, the system turn at 1, the first user turn at 1800, the last turn at 3200, text to 5000
        pts = metal_restore_points(ids(5000, {0: 1, 1: B, 1800: B, 3200: B}), turn_id=B)
        self.assertEqual(pts, {"shared": 1800, "marks": [3200], "tail": 5000 - 512})

    def test_a_single_long_turn_gets_only_a_tail(self):
        self.assertEqual(metal_restore_points(ids(4000, {1: B}), turn_id=B), {"tail": 4000 - 512})

    def test_a_short_prompt(self):
        self.assertEqual(metal_restore_points(ids(300, {1: B, 200: B}), turn_id=B), {})


class MetalSettings(unittest.TestCase):
    def test_off_by_default(self):
        # the DeepSeek setting is on by default (4); the Metal one is not - the engines grow the support
        self.assertEqual(metal_restore_point_settings({}), (False, 512))

    def test_values(self):
        self.assertEqual(metal_restore_point_settings({MON: "0"}), (False, 512))
        self.assertEqual(metal_restore_point_settings({MON: "1"}), (True, 512))
        self.assertEqual(metal_restore_point_settings({MON: "yes", MTAIL: "x"}), (False, 512))
        self.assertEqual(metal_restore_point_settings({MON: "2", MTAIL: "0"}), (True, 0))
        self.assertEqual(metal_restore_point_settings({MON: "2", MTAIL: "2000"}), (True, 2000))


class TheLine(unittest.TestCase):
    def test_restore_keys_serialize_a_metal_dict(self):
        keys = StrataEngine.restore_keys({"_restore": {"marks": [3000], "tail": 3488, "shared": 1500}})
        self.assertEqual(keys, " marks=3000 tail=3488 shared=1500")

    def test_no_points_no_keys(self):
        self.assertEqual(StrataEngine.restore_keys({}), "")
        self.assertEqual(StrataEngine.restore_keys({"_restore": {}}), "")


if __name__ == "__main__":
    unittest.main()
