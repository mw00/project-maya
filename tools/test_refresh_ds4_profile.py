"""CPU-only checks: python -m unittest discover -s tools -p test_refresh_ds4_profile.py"""
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from refresh_ds4_profile import MAX_COUNT, read_counts, refresh


class RefreshTests(unittest.TestCase):
    def test_independent_runs_and_layer_normalization(self):
        # Old history favors expert zero; recent workload favors expert one.
        old = [900000, 100000, 100, 300]
        before = [10, 20, 30, 40]
        after = [10, 100, 90, 60]
        result, recent = refresh(old, [(before, after), (before, after)], 2, .8)
        self.assertEqual(recent, [0, 160, 120, 40])
        self.assertEqual(result, [36, 164, 130, 70])
        self.assertGreater(result[1], result[0])

    def test_zero_prior_and_recent_only(self):
        self.assertEqual(refresh([0, 0], [([0, 0], [2, 3])], 2, .8)[0], [2, 3])
        self.assertEqual(refresh([100, 0], [([0, 0], [2, 3])], 2, 1)[0], [2, 3])

    def test_invalid_calibration(self):
        for pairs in ([], [([1, 0], [0, 1])], [([0, 0], [0, 0])], [([0], [1])]):
            with self.assertRaises(ValueError):
                refresh([1, 1], pairs, 2, .8)
        for weight in (0, -1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                refresh([1, 1], [([0, 0], [1, 1])], 2, weight)
        with self.assertRaises(ValueError):
            refresh([1, 1], [([0, 0], [MAX_COUNT, MAX_COUNT])], 2, .8)

    def test_binary_geometry_and_endianness(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'profile.bin'
            p.write_bytes(struct.pack('<2I', 1, 256))
            self.assertEqual(read_counts(p, 1, 2), [1, 256])
            with self.assertRaises(ValueError):
                read_counts(p, 2, 2)

    def test_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source, output = d / 'input.bin', d / 'output.bin'
            source.write_bytes(struct.pack('<2I', 1, 2))
            output.write_bytes(b'keep me')
            run = subprocess.run([sys.executable, str(Path(__file__).with_name('refresh_ds4_profile.py')),
                '--layers', '1', '--experts', '2', '--historical', str(source),
                '--pair', str(source), str(source), '--out', str(output)], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(output.read_bytes(), b'keep me')
            self.assertFalse(Path(str(output) + '.json').exists())
