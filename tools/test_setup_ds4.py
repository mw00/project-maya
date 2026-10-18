"""tools/test_setup_ds4.py - the setup's two models: GLM-5.3-Flash or DeepSeek-V4.1-Flash-1017 (maya.py) - which one the
options pick, the DeepSeek download's manifest and URLs, its config (what a setup again keeps) and its start command.
No GPU, no network.

    python -m unittest tools.test_setup_ds4 -v
"""
from pathlib import Path
import json
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402


def args(**kw):
    base = dict(model=None, model_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, port=None, host=None,
                api_key=None, env=[], no_vision=False, vram_reserve_mib=None, parallel=None, gpus=None, gpu=None)
    base.update(kw)
    return SimpleNamespace(**base)


class Family(unittest.TestCase):
    def test_options_pick_the_model(self):
        self.assertEqual(maya.choose_family(args(model=maya.DS4), {}), "deepseek")
        self.assertEqual(maya.choose_family(args(model="Maya-L"), {}), "glm")
        self.assertEqual(maya.choose_family(args(model_dir="/x"), {}), "deepseek")
        self.assertEqual(maya.choose_family(args(gguf_dir="/x"), {}), "glm")
        self.assertEqual(maya.choose_family(args(draft="dflash2"), {}), "glm")
        self.assertEqual(maya.choose_family(args(backend="hip"), {}), "glm")   # (DeepSeek: NVIDIA only)

    def test_default_is_the_model_before(self):
        ds4 = {"exe": "/m/build/strata-ds4", "args": ["--pack", "/m/maya-ds4", "--max-context", "65536"]}
        glm = {"exe": "/m/build/strata", "args": ["--glm-pack", "/m/pack"]}
        with patch.object(maya.S, "gpus", return_value=[]), patch.object(maya, "say", lambda *a: None):
            self.assertEqual(maya.choose_family(args(), ds4), "deepseek")
            self.assertEqual(maya.choose_family(args(), glm), "glm")
            self.assertEqual(maya.choose_family(args(), {}), "glm")
        self.assertEqual(maya.config_family(ds4), "deepseek")
        self.assertEqual(maya.config_family(glm), "glm")
        self.assertEqual(maya.config_family({"installer": {"family": "deepseek"}}), "deepseek")


class Manifest(unittest.TestCase):
    def test_every_file_pinned(self):
        m = maya.ds4_model()
        self.assertGreater(len(m["files"]), 45)
        for n in ["config.json", "model.safetensors.index.json", "tokenizer.json"] + \
                 [f"model-{i:05d}-of-00043.safetensors" for i in range(1, 44)] + \
                 ["model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"]:
            self.assertIn(n, m["files"])
        for n, size in m["files"].items():
            self.assertNotIn("/", n)                      # (the downloads are flat: no subfolders)
            self.assertGreater(size, 0)
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
        self.assertAlmostEqual(m["download_gb"], 413.9, delta=0.5)

    def test_urls(self):
        m = maya.ds4_model()
        rev = re.compile(r"/resolve/[0-9a-f]{40}/")
        u = maya.file_url(m, "model-00001-of-00043.safetensors")
        self.assertTrue(u.startswith("https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/"))
        self.assertRegex(u, rev)
        u = maya.file_url(m, "model-00047-of-00048.safetensors")   # the Engram tables: DeepSeek's own repo
        self.assertTrue(u.startswith("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/"))
        self.assertRegex(u, rev)
        self.assertTrue(u.endswith("/model-00047-of-00048.safetensors"))


class Config(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.patches = [patch.object(maya, "ROOT", self.root), patch.object(maya, "say", lambda *a: None),
                        patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def write(self, a, old=None):
        if old is not None:
            (self.root / "maya-ds4.json").write_text(json.dumps(old))
        pc = {"gpus": [{"index": i} for i in range(3)]}
        p = maya.write_config_ds4(a, pc, {"lib_dirs": ["/cuda/lib64"]}, Path("/m/DS/maya-ds4"), 131072,
                                  Path("/models"), Path("/m/DS"))
        return p, json.loads(p.read_text())

    def test_fresh(self):
        p, c = self.write(args())
        self.assertEqual(p.name, "maya-ds4.json")
        self.assertEqual(c["args"], ["--pack", "/m/DS/maya-ds4", "--max-context", "131072", "--tp", "2"])
        self.assertEqual(c["exe"], str(maya.DS4_EXE))
        self.assertEqual(c["tokenizer"], "/m/DS/maya-ds4/tokenizer")
        self.assertEqual((c["sampling"], c["reasoning_effort"]), (maya.DS4_SAMPLING, maya.DS4_EFFORT))
        self.assertEqual(c["gpu"], [0, 1, 2])
        self.assertEqual(c["installer"]["family"], "deepseek")
        self.assertEqual(maya.config_family(c), "deepseek")
        self.assertIn("maya.py", (self.root / "run-maya-ds4.sh").read_text())

    def test_again_keeps_what_it_does_not_ask(self):
        old = {"args": ["--pack", "/old/maya-ds4", "--max-context", "262144", "--tp", "1", "--prefill-chunk", "4096"],
               "reasoning_effort": "max", "sampling": {"temperature": 0.6}, "host": "0.0.0.0", "api_key": "k",
               "env": {"DS4_RING_MB": "800"}}
        _, c = self.write(args(no_vision=True, vram_reserve_mib=900, env=["DS4_CHUNK_MIN=256"]), old)
        a = c["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "131072")    # (asked now)
        self.assertEqual(a[a.index("--tp") + 1], "1")
        self.assertEqual(a[a.index("--prefill-chunk") + 1], "4096")
        self.assertEqual(a[a.index("--reserve-mib") + 1], "900")
        self.assertEqual(a[a.index("--vision") + 1], "off")
        self.assertEqual((c["reasoning_effort"], c["sampling"], c["host"], c["api_key"]),
                         ("max", {"temperature": 0.6}, "0.0.0.0", "k"))
        self.assertEqual(c["env"], {"DS4_RING_MB": "800", "DS4_CHUNK_MIN": "256"})


class Start(unittest.TestCase):
    def test_server_command(self):
        d = Path(tempfile.mkdtemp())
        pack = d / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        exe = d / "strata-ds4"
        exe.write_text("")
        cfg = {"exe": str(exe), "args": ["--pack", str(pack), "--max-context", "65536", "--tp", "2"],
               "tokenizer": str(pack / "tokenizer"), "model_name": maya.DS4_MODEL_NAME, "gpu": [0, 1], "port": 8099}
        p = d / "maya-ds4.json"
        p.write_text(json.dumps(cfg))
        quiet = [patch.object(maya, "say", lambda *a: None), patch.object(maya, "ok", lambda *a: None),
                 patch.object(maya, "refresh_engine", lambda c: None), patch.object(maya.S, "UI", None)]
        for q in quiet:
            q.start()
        try:
            cmd, env, info = maya.server_command(p, args())
        finally:
            for q in quiet:
                q.stop()
        self.assertIn("--config", cmd)
        self.assertEqual(cmd[cmd.index("--port") + 1], "8099")
        self.assertEqual(info["context"], "65536")
        self.assertNotIn("--expert-profile", json.loads(p.read_text())["args"])   # (GLM's profile is not DeepSeek's)


if __name__ == "__main__":
    unittest.main()
