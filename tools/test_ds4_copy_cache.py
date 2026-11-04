"""CPU-only checks for opt-in expert-copy cache controls; no global cache clearing."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(sys.platform == "linux" and shutil.which("c++"), "Linux C++ compiler required")
class ExpertCopyCache(unittest.TestCase):
    def test_bytes_alignment_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = str(Path(directory) / "ds4-copy-cache-test")
            subprocess.run(["c++", "-std=c++17", "-O2", "-pthread", str(ROOT / "tests/ds4_copy_cache.cpp"), "-o", exe],
                           check=True, timeout=120)
            # DS4_TEST_FILE_DIR may select a real filesystem for optional O_DIRECT/eviction diagnostics.
            subprocess.run([exe, os.environ.get("DS4_TEST_FILE_DIR", directory)], check=True, timeout=30)

if __name__ == "__main__":
    unittest.main()
