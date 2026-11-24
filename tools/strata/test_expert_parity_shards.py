#!/usr/bin/env python3
"""Unit tests for tools/strata/expert_parity_shards.py (header-only GGUF shards, a fake parity binary)."""
from __future__ import annotations

import os
import pathlib
import struct
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert_parity_shards as E  # noqa: E402


def gguf(path: pathlib.Path, names: list[str]) -> pathlib.Path:
    """A header-only GGUF v3 file: no KV pairs, one 2-D Q8_0 tensor info per name."""
    b = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(names), 0)
    for n in names:
        nb = n.encode()
        b += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", 2) + struct.pack("<QQ", 64, 32)
        b += struct.pack("<IQ", 8, 0)
    path.write_bytes(bytes(b))
    return path


def exps(l: int, roles=E.ROLES) -> list[str]:
    return [f"blk.{l}.ffn_{r}_exps.weight" for r in roles]


class Plan(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        d = pathlib.Path(self.t.name)
        self.s1 = gguf(d / "m-00001-of-00003.gguf", ["token_embd.weight"])            # metadata shard
        self.s2 = gguf(d / "m-00002-of-00003.gguf", exps(3) + exps(26) + ["blk.54.ffn_gate_exps.weight"])
        self.s3 = gguf(d / "m-00003-of-00003.gguf", exps(77) + exps(78) + exps(54, ("up", "down")))
        self.d = d

    def tearDown(self):
        self.t.cleanup()

    def test_each_shard_gets_only_its_layers(self):
        runs, problems = E.plan([self.s1, self.s2, self.s3], [3, 26, 77, 78])
        self.assertEqual([(p.name, ls) for p, ls in runs],
                         [(self.s2.name, [3, 26]), (self.s3.name, [77, 78])])
        self.assertEqual(problems, [])

    def test_missing_and_split_layers_fail(self):
        runs, problems = E.plan([self.s1, self.s2, self.s3], [3, 54, 60])
        self.assertEqual([(p.name, ls) for p, ls in runs], [(self.s2.name, [3])])
        self.assertEqual(len(problems), 2)
        self.assertIn("layer 54: gate/up/down split", problems[0])
        self.assertIn("layer 60: no shard holds", problems[1])
        self.assertEqual(E.main(["--plan", "--layers", "3,54", str(self.s2), str(self.s3)]), 1)

    def test_runs_the_binary_per_shard(self):
        log = self.d / "calls.txt"
        fake = self.d / "fake_parity.py"
        fake.write_text("#!%s\nimport sys\nopen(%r, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                        "sys.exit(int(__import__('os').environ.get('FAKE_RC', '0')))\n" % (sys.executable, str(log)))
        fake.chmod(0o755)
        args = ["--bin", str(fake), "--layers", "3 26 77", str(self.s1), str(self.s2), str(self.s3)]
        self.assertEqual(E.main(args), 0)
        self.assertEqual(log.read_text().splitlines(), [f"{self.s2} 3 26", f"{self.s3} 77"])
        os.environ["FAKE_RC"] = "77"
        try:
            self.assertEqual(E.main(args), 77)          # no device anywhere: skipped, not passed
            os.environ["FAKE_RC"] = "1"
            self.assertEqual(E.main(args), 1)
        finally:
            del os.environ["FAKE_RC"]
        # a layer no shard holds fails even when every run passes
        self.assertEqual(E.main(["--bin", str(fake), "--layers", "3;60", str(self.s2)]), 1)

    def test_not_a_gguf_is_a_usage_error(self):
        bad = self.d / "bad.gguf"
        bad.write_bytes(b"nope" * 8)
        self.assertEqual(E.main(["--plan", "--layers", "3", str(bad)]), 2)


if __name__ == "__main__":
    unittest.main()
