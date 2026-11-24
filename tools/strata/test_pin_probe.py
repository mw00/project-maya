"""tools/strata/test_pin_probe.py - pin_probe.cu builds and its CPU-only variant runs (B0.11).

The CUDA build is compile-checked when nvcc is on PATH (or $CUDA_HOME/bin); the registration itself needs the test PC.
The CPU-only variant (-DPIN_PROBE_NO_CUDA) runs a small probe here: mbind, fault-in, node check, acceptance line.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

SRC = pathlib.Path(__file__).resolve().parent / "pin_probe.cu"
GXX = shutil.which("g++")
NVCC = shutil.which("nvcc") or next((str(p) for p in [pathlib.Path(os.environ.get("CUDA_HOME", "/nonexistent")) / "bin" / "nvcc",
                                                      pathlib.Path("/usr/local/cuda-12.9/bin/nvcc")] if p.exists()), None)


class PinProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.bin = pathlib.Path(cls.td.name) / "pin_probe_cpu"
        if GXX:
            subprocess.run([GXX, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-x", "c++", "-DPIN_PROBE_NO_CUDA",
                            str(SRC), "-o", str(cls.bin), "-lpthread"], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    @unittest.skipUnless(GXX, "g++ not found")
    def test_cpu_probe_passes(self):
        r = subprocess.run([str(self.bin), "--total-gib", "0.25", "--seg-gib", "0.125", "--nodes", "0",
                            "--min-avail-gib", "0.5", "--touch-threads", "2"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = r.stdout.splitlines()
        rows = [l.split() for l in lines if l.strip()[:1].isdigit()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[1] == "0" and row[3] == "0" for row in rows), rows)   # bound node 0, landed on 0
        self.assertIn("PIN_PROBE PASS", r.stdout)

    @unittest.skipUnless(GXX, "g++ not found")
    def test_acceptance_fails_on_low_memavailable(self):
        r = subprocess.run([str(self.bin), "--total-gib", "0.0625", "--seg-gib", "0.0625", "--nodes", "0",
                            "--min-avail-gib", "1000000"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("PIN_PROBE FAIL (MemAvailable too low)", r.stdout)

    @unittest.skipUnless(GXX, "g++ not found")
    def test_bad_arguments(self):
        r = subprocess.run([str(self.bin), "--bogus"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    @unittest.skipUnless(NVCC, "nvcc not found (compile check of the CUDA build)")
    def test_cuda_build_compiles_for_sm70(self):
        out = pathlib.Path(self.td.name) / "pin_probe"
        r = subprocess.run([NVCC, "-O2", "-std=c++17", "-arch=sm_70", "-Wno-deprecated-gpu-targets", str(SRC), "-o",
                            str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
