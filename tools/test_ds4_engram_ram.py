"""Small CPU-only regression tests for the optional Engram RAM reader (no model or GPU)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(sys.platform == "linux" and shutil.which("c++"), "Linux C++ compiler required")
class EngramRam(unittest.TestCase):
    def test_locked_reads_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as d:
            exe = str(Path(d) / "engram-ram-test")
            subprocess.run(["c++", "-std=c++17", "-O2", "-pthread", str(ROOT / "tests/ds4_engram_ram.cpp"),
                            "-o", exe], check=True, timeout=120)
            subprocess.run([exe], check=True, timeout=30)

if __name__ == "__main__":
    unittest.main()
