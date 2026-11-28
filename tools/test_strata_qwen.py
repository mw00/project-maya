"""tools/test_strata_qwen.py - tools/strata_qwen.py's config writing and checks (no GPU, no network, no build).

    python tools/test_strata_qwen.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_qwen as SQ  # noqa: E402

QWEN = {"exe": "/m/build/maya", "args": ["--pack", "/m/pack", "--native", "/m/a.gguf", "--ple-gguf", "/m/b.gguf",
                                         "--spec", "4", "--mtp", "/m/pack/mtp", "--max-context", "262144"],
        "env": {"MAYA_HC_Q8": "1"}, "installer": {"family": "qwen", "qwen_quant": "ud-iq2_xs"}, "port": 8080}


class WriteConfig(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.src = self.d / "maya-qwen.json"
        self.src.write_text(json.dumps(QWEN))

    def test_engine_and_spread(self):
        out = self.d / "maya-qwen-strata.json"
        SQ.write_config(self.src, out, Path("/x/build/strata-qwen/strata"), None)
        c = json.loads(out.read_text())
        self.assertEqual(c["exe"], "/x/build/strata-qwen/strata")
        self.assertEqual(c["args"][:len(QWEN["args"])], QWEN["args"])      # (the model and its settings as they were)
        self.assertEqual(c["args"].count("--prompt-cache-spread"), 1)
        self.assertNotIn("--draft", c["args"])                              # (DFlash2 only when asked for)
        self.assertEqual(c["env"], QWEN["env"])
        self.assertEqual(c["installer"], QWEN["installer"])                 # (the Models page still knows the files)
        self.assertEqual(c["log"], str(out.with_suffix(".log").resolve()))
        self.assertEqual(json.loads(self.src.read_text()), QWEN)            # (the old config untouched)

    def test_dflash2_and_idempotent(self):
        out = self.d / "o.json"
        SQ.write_config(self.src, out, Path("/e"), Path("/p/df2.pack"))
        SQ.write_config(out, out, Path("/e"), Path("/p/df2b.pack"))         # (again over its own output)
        a = json.loads(out.read_text())["args"]
        self.assertEqual(a.count("--draft"), 1)
        self.assertEqual(a[a.index("--draft") + 1], "dflash2")
        self.assertEqual(a[a.index("--draft-model") + 1], "/p/df2b.pack")
        self.assertEqual(a.count("--prompt-cache-spread"), 1)


class Checks(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "tools" / "strata_qwen.py"), *args],
                              capture_output=True, text=True)

    def test_amd_arch_refused(self):
        r = self.run_tool("--arch", "gfx1100", "--no-build")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("NVIDIA", r.stderr)

    def test_not_qwen_refused_before_build(self):
        d = Path(tempfile.mkdtemp())
        p = d / "glm.json"
        p.write_text(json.dumps({"exe": "x", "args": ["--glm-pack", "/p"]}))
        r = self.run_tool("--arch", "86", "--from", str(p))   # (no build is started: the check comes first)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not a Qwen3.8-Flash-Next config", r.stderr)

    def test_dflash2_needs_mtp(self):
        d = Path(tempfile.mkdtemp())
        p, pack = d / "q.json", d / "df2.pack"
        pack.write_bytes(b"x")
        c = dict(QWEN, args=[a for a in QWEN["args"] if a not in ("--mtp", "/m/pack/mtp")])
        p.write_text(json.dumps(c))
        r = self.run_tool("--arch", "86", "--from", str(p), "--dflash2", str(pack))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--mtp", r.stderr)


if __name__ == "__main__":
    unittest.main()
