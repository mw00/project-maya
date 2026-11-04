"""Pascal (compute capability 6.x) in Maya's setup: accepted as experimental, built with CUDA 12 for its own arch like
Volta, and anything older still refused (#212, #214).  No GPU needed."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


def gpu(index, arch, name, vram):
    return {"index": index, "arch": arch, "name": name, "vram_gb": vram, "driver": "580.126.20"}


class PascalSetupTests(unittest.TestCase):
    def setUp(self):
        self.a = SimpleNamespace(gpu=None, gpus=None)

    def test_a_p40_can_be_used(self):
        p40 = gpu(0, "61", "Tesla P40", 24)
        self.assertEqual(maya.choose_gpus(self.a, [p40]), [p40])
        self.a.gpu = 0
        self.assertEqual(maya.choose_gpus(self.a, [p40]), [p40])

    def test_older_than_pascal_is_refused(self):
        m40 = gpu(0, "52", "Tesla M40", 24)
        with self.assertRaises(SystemExit):
            maya.choose_gpus(self.a, [m40])
        self.a.gpu = 0
        with self.assertRaises(SystemExit):
            maya.choose_gpus(self.a, [m40])
        self.a.gpu = 1                                  # a P40 beside it is fine, the M40 is not offered
        p40 = gpu(1, "61", "Tesla P40", 24)
        self.assertEqual(maya.choose_gpus(self.a, [m40, p40]), [p40])

    def test_pascal_builds_like_volta(self):
        self.assertEqual(maya.arch_setting([61]), "61")
        self.assertEqual(maya.nvcc_range([61]), ((12, 0), (13, 0)))
        self.assertEqual(maya.pick_nvcc([("/cuda13/nvcc", (13, 0)), ("/cuda12/nvcc", (12, 8))], [61]),
                         ("/cuda12/nvcc", (12, 8)))
        self.assertIn("Pascal", maya.toolkit_hint([61]))

    def test_volta_and_newer_unchanged(self):
        self.assertEqual(maya.arch_setting([70]), "70")   # (by name: never "native", which takes every card)
        self.assertEqual(maya.arch_setting([86, 89]), "86;89")
        self.assertEqual(maya.nvcc_range([86]), ((12, 0), None))
        self.assertLess(maya.MIN_CC, maya.VALIDATED_CC)
        self.assertEqual(maya.VALIDATED_CC, 70)


if __name__ == "__main__":
    unittest.main()
