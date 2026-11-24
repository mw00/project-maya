#!/usr/bin/env python3
"""Unit tests for tools/strata/pack_test_gate.py (synthetic logits files, a fake glm_pack_test)."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack_test_gate as G  # noqa: E402

N, V = 6, 50


class Gate(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.t.name)
        rng = np.random.default_rng(3)
        self.ref = (rng.standard_normal((N + 2, V)) * 3).astype(np.float32)
        self.ref_path = self.d / "ref.bin"
        self.ref.tofile(self.ref_path)
        self.ids = self.d / "ids.txt"
        self.ids.write_text(" ".join(str(i + 1) for i in range(N + 2)))

    def tearDown(self):
        self.t.cleanup()

    def write(self, a: np.ndarray) -> str:
        p = self.d / "got.f32"
        a.astype(np.float32).tofile(p)
        return str(p)

    def run_gate(self, got: np.ndarray, *extra: str) -> int:
        return G.main(["--tokens", str(N), "--logits", self.write(got), *extra])

    def test_close_logits_pass(self):
        got = self.ref[:N] + 0.01
        self.assertEqual(self.run_gate(got, "--ids", str(self.ids), "--ref-logits", str(self.ref_path)), 0)

    def test_wrong_logits_fail(self):
        got = -self.ref[:N]
        self.assertEqual(self.run_gate(got, "--ids", str(self.ids), "--ref-logits", str(self.ref_path)), 1)

    def test_nan_and_flat_rows_fail_even_informational(self):
        got = self.ref[:N].copy()
        got[2, 7] = np.nan
        self.assertEqual(self.run_gate(got, "--informational"), 1)
        got = self.ref[:N].copy()
        got[4, :] = 1.0
        self.assertEqual(self.run_gate(got, "--informational"), 1)
        self.assertEqual(self.run_gate(got, "--ids", str(self.ids), "--ref-logits", str(self.ref_path)), 1)

    def test_no_reference_fails_unless_informational(self):
        self.assertEqual(self.run_gate(self.ref[:N]), 1)
        self.assertEqual(self.run_gate(self.ref[:N], "--informational"), 0)

    def test_usage(self):
        self.assertEqual(self.run_gate(self.ref[:N], "--ref-logits", str(self.ref_path)), 2)   # no --ids
        short = self.d / "short.txt"
        short.write_text("1 2")
        self.assertEqual(self.run_gate(self.ref[:N], "--ids", str(short), "--informational"), 2)
        self.assertEqual(G.main(["--tokens", "7", "--logits", self.write(self.ref[:N]), "--informational"]), 1)

    def test_runs_the_binary(self):
        fake = self.d / "fake_pack_test.py"
        src = self.d / "src.f32"
        (self.ref[:N] + 0.01).tofile(src)
        fake.write_text(
            "#!%s\nimport os, shutil, sys\n"
            "assert os.environ.get('GLM_TEST_IDS') == %r\n"
            "shutil.copy(%r, sys.argv[3])\n"
            "sys.exit(int(os.environ.get('FAKE_RC', '0')))\n" % (sys.executable, str(self.ids), str(src)))
        fake.chmod(0o755)
        out = str(self.d / "out.f32")
        args = ["--bin", str(fake), "--pack", str(self.d), "--tokens", str(N), "--logits", out,
                "--ids", str(self.ids), "--ref-logits", str(self.ref_path)]
        self.assertEqual(G.main(args), 0)
        import os
        os.environ["FAKE_RC"] = "77"
        try:
            self.assertEqual(G.main(args), 77)
            os.environ["FAKE_RC"] = "3"
            self.assertEqual(G.main(args), 1)
        finally:
            del os.environ["FAKE_RC"]


if __name__ == "__main__":
    unittest.main()
