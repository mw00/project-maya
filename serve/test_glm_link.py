"""serve/glm_link.py: a GLM config without "mixed_gpus" starts unchanged; with it, the CUDA lead + the HIP helper."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import glm_link as GL


def builds(root: Path) -> Path:
    for d, names in (("build-hip", ("maya", "maya-glm-helper")), ("build-cuda", ("maya",))):
        (root / d).mkdir(parents=True, exist_ok=True)
        for n in names:
            (root / d / n).write_text("")
    return root / "build-hip" / "maya"


def config(exe: Path, **kw) -> dict:
    return {"exe": str(exe), "backend": "hip", "gpu": [0], "args": ["--glm-pack", "/m/pack", "--max-context", "65536"],
            "env": {"MAYA_GLM_RESERVE_MB": "1024", "MAYA_HIPBLASLT_TUNING": "/t.txt", "MAYA_GLM_PREFILL_SUB": "1024"},
            **kw}


class GlmLinkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.exe = builds(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_key_unchanged(self):
        c = config(self.exe)
        before = json.dumps(c, sort_keys=True)
        with mock.patch.object(GL, "nvidia_cards", return_value=[0]):
            out, line = GL.at_start(c)
        self.assertEqual(json.dumps(out, sort_keys=True), before)
        self.assertEqual(line, "")
        out, line = GL.at_start(config(self.exe, mixed_gpus=False))
        self.assertEqual(line, "")
        self.assertEqual(out["exe"], str(self.exe))

    @unittest.skipIf(not Path("/proc").exists(), "Linux only")
    def test_on_is_link(self):
        with mock.patch.object(GL, "nvidia_cards", return_value=[1]):
            out, line = GL.at_start(config(self.exe, mixed_gpus=True))
        self.assertIn("both GPUs ON", line)
        self.assertEqual(out["backend"], "cuda")
        self.assertEqual(Path(out["exe"]), Path(self.tmp.name).resolve() / "build-cuda" / "maya")
        self.assertEqual(out["gpu"], [1])
        e = out["env"]
        self.assertEqual(e["MAYA_GLM_EXPERT_HELPER"], "link")
        self.assertEqual(e["HIP_VISIBLE_DEVICES"], "0")
        self.assertTrue(e["MAYA_GLM_HELPER_EXE"].endswith("maya-glm-helper"))
        self.assertNotIn("MAYA_HIPBLASLT_TUNING", e)
        self.assertEqual(e["MAYA_GLM_RESERVE_MB"], "1024")

    @unittest.skipIf(not Path("/proc").exists(), "Linux only")
    def test_falls_back(self):
        with mock.patch.object(GL, "nvidia_cards", return_value=[]):
            out, line = GL.at_start(config(self.exe, mixed_gpus=True))
        self.assertIn("AMD GPU alone", line)
        self.assertEqual(out["backend"], "hip")
        (Path(self.tmp.name) / "build-hip" / "maya-glm-helper").unlink()
        with mock.patch.object(GL, "nvidia_cards", return_value=[0]):
            self.assertIn("helper", GL.why_not(config(self.exe)))
        self.assertIsNotNone(GL.why_not(config(self.exe, backend="cuda"), [0]))
        self.assertIsNotNone(GL.why_not({"exe": str(self.exe), "backend": "hip", "args": ["--pack", "x"]}, [0]))


if __name__ == "__main__":
    unittest.main()
