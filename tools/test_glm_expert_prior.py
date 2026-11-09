"""glm_expert_prior tests (glm-dsa B8.5); run python3 -m unittest discover -s tools -p test_glm_expert_prior.py."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import glm_expert_prior as gep


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = gep.main(argv)
    return rc, buf.getvalue()


def rows(path):
    return [line.split() for line in Path(path).read_text().splitlines() if line and not line.startswith("#")]


class PriorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def header(self, n):
        (self.d / "native_experts.txt").write_text(
            "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
            f"(n_expert {n}, total 123; absolute offsets in x.gguf)\n3 16 18 0 100 0 0 0\n")

    def test_glm_dsa_256_from_the_pack(self):
        self.header(256)
        tr = self.d / "trace.txt"
        tr.write_text("3 7\n3 7\n3 200\n4 255\n")
        rc, out = run([str(self.d), str(tr)])
        self.assertEqual(rc, 0, out)
        prior = rows(self.d / "expert_prior.txt")
        self.assertEqual([len(r) - 1 for r in prior], [256, 256])   # no 288 padding
        self.assertEqual(prior[0][1:4], ["7", "200", "0"])
        self.assertEqual(sorted(map(int, prior[1][1:])), list(range(256)))
        counts = rows(self.d / "expert_counts.txt")
        self.assertEqual(len(counts[0]) - 1, 256)
        self.assertEqual(int(counts[0][1 + 7]), 2)

    def test_flash_288_and_usage_lines(self):
        self.header(288)
        us = self.d / "usage.txt"
        us.write_text("5 1:4 287:9\n")
        rc, out = run([str(self.d), str(us)])
        self.assertEqual(rc, 0, out)
        prior = rows(self.d / "expert_prior.txt")
        self.assertEqual(len(prior[0]) - 1, 288)
        self.assertEqual(prior[0][1:3], ["287", "1"])

    def test_explicit_and_seen_counts(self):
        tr = self.d / "trace.txt"
        tr.write_text("3 9\n")
        rc, _ = run(["--n-expert", "16", str(self.d), str(tr)])
        self.assertEqual(rc, 0)
        self.assertEqual(len(rows(self.d / "expert_prior.txt")[0]) - 1, 16)
        rc, _ = run([str(self.d), str(tr)])   # no header, no flag: the ids seen
        self.assertEqual(len(rows(self.d / "expert_prior.txt")[0]) - 1, 10)
        rc, out = run(["--n-expert", "8", str(self.d), str(tr)])
        self.assertEqual(rc, 1)
        self.assertIn("expert id 9", out)

    def test_hit_curve(self):
        self.header(256)
        # layer 3: expert e used 256 - e times (a skewed layer); layer 4: uniform
        lines = []
        for e in range(256):
            lines += [f"3 {e}"] * (256 - e) + [f"4 {e}"] * 10
        tr = self.d / "trace.txt"
        tr.write_text("\n".join(lines) + "\n")
        rc, out = run([str(self.d), str(tr)])
        self.assertEqual(rc, 0, out)
        curve = [tuple(map(float, r)) for r in rows(self.d / "hit_curve.txt")]
        self.assertEqual(len(curve), 101)
        self.assertEqual(curve[0], (0.0, 0.0))
        self.assertAlmostEqual(curve[-1][1], 1.0)
        hs = [h for _, h in curve]
        self.assertEqual(hs, sorted(hs))   # monotone
        # f = 0.5: top 128 of layer 3 = sum(129..256) = 24640 visits, layer 4: 1280, of 32896 + 2560
        self.assertAlmostEqual(curve[50][1], (24640 + 1280) / (32896 + 2560), places=6)
        self.assertGreater(curve[20][1], 0.2)   # the skew lifts h above f


if __name__ == "__main__":
    unittest.main()
