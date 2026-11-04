"""Exercise CMake's actual architecture parser without enabling a compiler."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HipArchitectureTests(unittest.TestCase):
    def probe(self, value):
        # Run the same validation/normalization code used before enable_language.
        source = (ROOT / "cmake/hip_backend.cmake").read_text().split("enable_language(HIP)")[0]
        source += '\nmessage(STATUS "compiler=${CMAKE_HIP_ARCHITECTURES};runtime=${STRATA_HIP_ARCHS}")\n'
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "probe.cmake"
            script.write_text("cmake_minimum_required(VERSION 3.24)\n" + source)
            return subprocess.run(["cmake", f"-DCMAKE_HIP_ARCHITECTURES={value}", "-P", str(script)],
                                  capture_output=True, text=True)

    def test_single(self):
        r = self.probe("gfx1201")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("compiler=gfx1201;runtime=gfx1201", r.stdout)

    def test_multi_arch(self):
        r = self.probe("gfx1100;gfx1201;gfx1151")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("compiler=gfx1100;gfx1201;gfx1151;runtime=gfx1100,gfx1201,gfx1151", r.stdout)
        self.assertNotIn("not validated", r.stderr)                 # (Strix Halo is validated)
        r = self.probe("gfx1100;gfx1150")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("not validated", r.stderr)

    def test_spaces_suffixes_and_duplicates(self):
        r = self.probe("gfx1100:xnack-  gfx1201;gfx1201;gfx1151")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("compiler=gfx1100:xnack-;gfx1201;gfx1151;runtime=gfx1100,gfx1201,gfx1151", r.stdout)

    def test_rejects_unsupported_and_prefix_matches(self):
        for value in ("gfx12010", "gfx90a", "gfx1100;gfx1036"):
            with self.subTest(value=value):
                self.assertNotEqual(self.probe(value).returncode, 0)

    def test_rejects_whitespace_only(self):
        r = self.probe("   ")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("empty", r.stderr)


if __name__ == "__main__":
    unittest.main()
