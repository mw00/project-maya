"""tools/test_setup_ds4.py - the setup's two models: GLM-5.3-Flash or DeepSeek-V4.1-Flash-1017 (maya.py; DeepSeek at 2.9 or
3.5 bits) - which one the options pick, the DeepSeek downloads' manifests and URLs, the Engram tables shared between the
quants, its config (what a setup again keeps) and its start command.
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


class Quant35(unittest.TestCase):
    """The 3.5-bit quant: picked by --model or the menu, its own pinned manifest, its own folder, the Engram tables
    linked from the 2.9-bit folder rather than downloaded again."""

    def test_picked(self):
        self.assertEqual(maya.choose_family(args(model=maya.DS4_35), {}), "deepseek")
        a = args()
        with patch.object(maya.S, "gpus", return_value=[]), patch.object(maya, "say", lambda *x: None):
            self.assertEqual(maya.choose_family(a, {"installer": {"family": "deepseek", "quant": maya.DS4_35}}),
                             "deepseek")
        self.assertEqual(a.model, maya.DS4_35)                 # (the setup before: the default row)
        self.assertEqual(maya.ds4_quant(args(), {"installer": {"quant": maya.DS4_35}}), maya.DS4_35)
        self.assertEqual(maya.ds4_quant(args(), {}), maya.DS4)

    def test_every_file_pinned(self):
        m = maya.ds4_model(maya.DS4_35)
        for n in ["config.json", "model.safetensors.index.json", "tokenizer.json"] + \
                 [f"model-{i:05d}-of-00044.safetensors" for i in range(1, 45)] + \
                 ["model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"]:
            self.assertIn(n, m["files"])
        for n, size in m["files"].items():
            self.assertNotIn("/", n)
            self.assertGreater(size, 0)
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
        self.assertAlmostEqual(m["download_gb"], 460.7, delta=0.5)
        self.assertIn("3.5 bits", m["about"])
        rev = re.compile(r"/resolve/[0-9a-f]{40}/")
        u = maya.file_url(m, "model-00001-of-00044.safetensors")
        self.assertTrue(u.startswith("https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw/"))
        self.assertRegex(u, rev)
        u = maya.file_url(m, "model-00048-of-00048.safetensors")
        self.assertTrue(u.startswith("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/"))

    def test_own_folder(self):
        models = Path(tempfile.mkdtemp())
        old = models / "old"
        old.mkdir()
        (old / "config.json").write_text("{}")
        prev = {"args": ["--pack", str(old / "maya-ds4")], "installer": {"quant": maya.DS4}}
        self.assertEqual(maya.ds4_dir(args(), models, prev, maya.DS4), old)          # (the same quant: its folder)
        self.assertEqual(maya.ds4_dir(args(), models, prev, maya.DS4_35), models / maya.DS4_35)

    def test_engram_linked(self):
        models = Path(tempfile.mkdtemp())
        m = maya.ds4_model(maya.DS4_35)
        have = models / maya.DS4
        have.mkdir()
        book = {}
        for n in m["sources"]:
            with open(have / n, "wb") as f:
                f.truncate(m["files"][n])                  # (sparse: the size without the bytes)
            st = (have / n).stat()
            book[n] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        (have / maya.VERIFIED).write_text(json.dumps(book))
        d = models / maya.DS4_35
        with patch.object(maya, "ok", lambda *x: None), patch.object(maya, "configs", return_value=[]):
            maya.ds4_link_engram(d, m, models, {})
        for n in m["sources"]:
            self.assertEqual((d / n).stat().st_ino, (have / n).stat().st_ino)    # (a hard link, not a copy)
            self.assertIsNone(maya.file_incomplete(m, d / n))
        self.assertEqual(json.loads((d / maya.VERIFIED).read_text()), book)


GPUS3 = [{"index": i, "name": "RTX 5060 Ti", "vram_gb": 15.9} for i in range(3)]


def fake_pc(root: Path) -> list:
    """The patches of a PC with GPUS3 and a calibration store of its own (root / calibration.json)."""
    return [patch.object(maya.S, "gpus", return_value=GPUS3), patch.object(maya.S, "cpu_info", return_value=("Xeon",)),
            patch.object(maya.S, "ram_gb", return_value=183.0), patch.object(maya, "CAL_STORE", root / "cal.json")]


class Config(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.patches = [patch.object(maya, "ROOT", self.root), patch.object(maya, "say", lambda *a: None),
                        patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)] + \
            fake_pc(self.root)
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

    def test_quant_recorded(self):
        pc = {"gpus": [{"index": 0}]}
        p = maya.write_config_ds4(args(), pc, {}, Path("/m/D35/maya-ds4"), 65536, Path("/models"), Path("/m/D35"),
                                  maya.DS4_35)
        c = json.loads(p.read_text())
        self.assertEqual(c["installer"]["quant"], maya.DS4_35)
        self.assertEqual(maya.ds4_quant(args(), c), maya.DS4_35)

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

    def tuned(self, tp=8, threads=33):
        key = maya.hardware_key({"gpu": [0, 1, 2], "args": ["--max-context", "131072"], "installer": {"quant": maya.DS4}})
        maya.CAL_STORE.write_text(json.dumps({key: {"args": {"--tp": tp, "--cpu-threads": threads},
                                                    "date": "2026-10-19"}}))

    def test_tuned_before_is_used(self):
        self.tuned()
        _, c = self.write(args())
        self.assertEqual(c["args"][-4:], ["--tp", "8", "--cpu-threads", "33"])
        self.tuned(threads=None)                       # (the engine's own threads: no flag)
        (self.root / "maya-ds4.json").unlink()         # (a fresh setup: no config before)
        _, c = self.write(args())
        self.assertNotIn("--cpu-threads", c["args"])

    def test_the_config_befores_own_wins(self):
        self.tuned()
        _, c = self.write(args(), {"args": ["--pack", "/old/maya-ds4", "--max-context", "131072", "--tp", "4"]})
        a = c["args"]
        self.assertEqual((a[a.index("--tp") + 1], a[a.index("--cpu-threads") + 1]), ("4", "33"))


class Calibrate(unittest.TestCase):
    """./maya.sh --calibrate on a DeepSeek config: tools/calibrate_ds4.py's result into the config and the store."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cfg = self.root / "maya-ds4.json"
        self.cfg.write_text(json.dumps({"exe": str(maya.DS4_EXE), "gpu": [0, 1, 2], "installer": {"family": "deepseek",
                                        "quant": maya.DS4}, "args": ["--pack", "/m/maya-ds4", "--max-context",
                                                                     "65536", "--tp", "2", "--cpu-threads", "40"]}))
        self.patches = [patch.object(maya, "say", lambda *a: None), patch.object(maya, "ok", lambda *a: None),
                        patch.object(maya, "warn", lambda *a: None)] + fake_pc(self.root)
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_result_saved(self):
        res = {"args": {"--tp": 8, "--cpu-threads": None},
               "report": {"tok_s": 151.0, "own_tok_s": 103.0, "prompt_tok_s": 607}}
        with patch.object(maya, "busy_gpus", return_value=[]), \
                patch.object(maya.CAL4, "run", return_value=res) as run:
            self.assertTrue(maya.calibrate_config(self.cfg))   # (a DeepSeek config: calibrate_ds4's tuning)
        self.assertEqual(run.call_args.args[1], [15.9, 15.9, 15.9])   # (the config's GPUs' memory)
        c = json.loads(self.cfg.read_text())
        self.assertEqual(c["args"], ["--pack", "/m/maya-ds4", "--max-context", "65536", "--tp", "8"])
        self.assertEqual(c["installer"]["calibration"]["tok_s"], 151.0)
        store = json.loads(maya.CAL_STORE.read_text())
        self.assertEqual(store[maya.hardware_key(c)]["args"], {"--tp": 8, "--cpu-threads": None})

    def test_busy_gpus(self):
        with patch.object(maya, "busy_gpus", return_value=[(0, 15000)]), patch.object(maya.CAL4, "run") as run:
            self.assertFalse(maya.calibrate_config(self.cfg))
        run.assert_not_called()

    def test_failed_keeps_the_config(self):
        before = self.cfg.read_text()
        with patch.object(maya, "busy_gpus", return_value=[]), \
                patch.object(maya.CAL4, "run", side_effect=RuntimeError("engine exited")):
            self.assertFalse(maya.calibrate_config(self.cfg))
        self.assertEqual(self.cfg.read_text(), before)
        self.assertFalse(maya.CAL_STORE.exists())


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
