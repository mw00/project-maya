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
                api_key=None, env=[], no_vision=False, vram_reserve_mib=None, parallel=None, gpus=None, gpu=None,
                ds4_quant=None)
    base.update(kw)
    return SimpleNamespace(**base)


class Family(unittest.TestCase):
    def test_options_pick_the_model(self):
        self.assertEqual(maya.choose_family(args(model=maya.DS4), {}), "deepseek")
        self.assertEqual(maya.choose_family(args(model="Maya-L"), {}), "glm")
        self.assertEqual(maya.choose_family(args(model_dir="/x"), {}), "deepseek")
        self.assertEqual(maya.choose_family(args(gguf_dir="/x"), {}), "glm")
        self.assertEqual(maya.choose_family(args(draft="dflash2"), {}), "glm")
        self.assertEqual(maya.choose_family(args(backend="hip"), {}), "glm")   # (asked: the default, GLM)

    def test_default_is_the_model_before(self):
        ds4 = {"exe": "/m/build/maya-ds4", "args": ["--pack", "/m/maya-ds4", "--max-context", "65536"]}
        glm = {"exe": "/m/build/maya", "args": ["--glm-pack", "/m/pack"]}
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
    """LibertAIDAI's 3.5-bit quant: its own pinned manifest, the Engram tables linked from the 2.9-bit folder rather
    than downloaded again."""

    def test_every_file_pinned(self):
        m = maya.ds4_model("exl3-3.5")
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
        e29 = maya.ds4_model("exl3")   # (the same Engram tables as the 2.9-bit quant's, pinned alike)
        for n in m["sources"]:
            self.assertEqual((m["files"][n], m["sha256"][n], m["sources"][n]),
                             (e29["files"][n], e29["sha256"][n], e29["sources"][n]))

    def test_offered_only_where_exl3_runs(self):
        """--ds4-quant exl3-3.5, or the menu's row - only on NVIDIA compute capability 7.0+ (not AMD, not a PC whose
        GPUs are all older)."""
        self.assertIn("exl3-3.5", maya.DS4_QUANTS)
        def offered(gpus, backend="cuda"):
            rows = []
            def pick(title, intro, opts, default, yes, *x):
                rows.extend(o[0] for o in opts)
                return default
            with patch.object(maya.S, "gpus", return_value=gpus), patch.object(maya, "choose", pick), \
                    patch.object(maya.S, "UI", None):
                maya.choose_ds4_quant(args(backend=backend), {})
            return [r for r in rows if r.startswith("EXL3")]
        self.assertEqual(len(offered([{"arch": 86}, {"arch": 70}])), 2)
        self.assertEqual(len(offered([{"arch": 70}, {"arch": 70}])), 2)       # (Volta: since v1.1.20)
        self.assertEqual(offered([{"arch": 61}, {"arch": 61}]), [])            # (Pascal: the GGUFs only)
        self.assertEqual(offered([], backend="hip"), [])
        self.assertRegex(offered([{"arch": 89}])[1], r"^EXL3 3\.5 bits \(461 GB\)$")
        self.assertEqual(maya.choose_ds4_quant(args(ds4_quant="exl3-3.5"), {}), "exl3-3.5")

    def test_own_folder_and_config(self):
        models = Path("/models")
        self.assertEqual(maya.ds4_dir(args(), models, {}, "exl3-3.5"), models / "DeepSeek-V4.1-Flash-1017-3.5bpw")
        self.assertEqual(maya.ds4_dir(args(), models, {}, "exl3"), models / "DeepSeek-V4.1-Flash-1017")
        d = Path(tempfile.mkdtemp())
        quiet = [patch.object(maya, "ROOT", d), patch.object(maya, "say", lambda *a: None),
                 patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for q in quiet:
            q.start()
        try:
            p = maya.write_config_ds4(args(), {"gpus": [{"index": 0}]}, {}, d / "maya-ds4", 65536, models, d, "exl3-3.5")
        finally:
            for q in quiet:
                q.stop()
        c = json.loads(p.read_text())
        self.assertEqual(c["installer"]["ds4_quant"], "exl3-3.5")
        self.assertNotIn("--vision", c["args"])                # (its own vision tower, as the 2.9-bit quant's)

    def test_engram_linked(self):
        tmp = Path(tempfile.mkdtemp())
        man = json.loads(maya.DS4_EXL3["exl3-3.5"]["manifest"].read_text(encoding="utf-8"))
        for i, k in enumerate(sorted(man["engram"]["files"])):
            man["engram"]["files"][k][0] = 3 + i               # (tiny stand-ins for the 101 GB tables)
        p = tmp / "download-3.5bpw.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        models = tmp / "models"
        with patch.dict(maya.DS4_EXL3, {"exl3-3.5": dict(maya.DS4_EXL3["exl3-3.5"], manifest=p)}):
            m = maya.ds4_model("exl3-3.5")
            have = models / maya.DS4
            have.mkdir(parents=True)
            book = {}
            for n in m["sources"]:
                (have / n).write_bytes(b"x" * m["files"][n])
                st = (have / n).stat()
                book[n] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            (have / maya.VERIFIED).write_text(json.dumps(book))
            d = models / maya.DS4_35
            with patch.object(maya, "ok", lambda *x: None), patch.object(maya, "configs", return_value=[]):
                maya.ds4_link_engram(d, m, models, {})
            for n in m["sources"]:
                self.assertTrue(Path.samefile(d / n, have / n))   # (a hard link, not a copy)
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
        self.assertEqual(c["args"], ["--pack", str(Path("/m/DS/maya-ds4")), "--max-context", "131072", "--tp", "2"])
        self.assertEqual(c["exe"], str(maya.DS4_EXE))
        self.assertEqual(c["tokenizer"], str(Path("/m/DS/maya-ds4/tokenizer")))
        self.assertEqual((c["sampling"], c["reasoning_effort"]), (maya.DS4_SAMPLING, maya.DS4_EFFORT))
        self.assertEqual(c["gpu"], [0, 1, 2])
        self.assertEqual(c["installer"]["family"], "deepseek")
        self.assertEqual(maya.config_family(c), "deepseek")
        self.assertIn("maya.py", (self.root / ("run-maya-ds4" + (".bat" if maya.WIN else ".sh"))).read_text())

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
        key = maya.hardware_key({"gpu": [0, 1, 2], "args": ["--max-context", "131072"], "installer": {"quant": maya.DS4, "ds4_quant": "exl3"}})
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

    def test_gguf_config(self):
        pc = {"gpus": [{"index": 0}]}
        p = maya.write_config_ds4(args(), pc, {}, Path("/m/G/maya-ds4"), 65536, Path("/models"), Path("/m/G"),
                                  "gguf-q8")
        c = json.loads(p.read_text())
        self.assertEqual(c["args"][c["args"].index("--vision") + 1], "off")   # (no vision tower in a GGUF)
        self.assertEqual(c["installer"]["ds4_quant"], "gguf-q8")
        self.assertEqual(maya.config_family(c), "deepseek")

    def test_hip_reserve(self):
        def reserve(gpu, old=None, **kw):
            (self.root / "maya-ds4-hip.json").unlink(missing_ok=True)   # (a fresh setup, or the one before: `old`)
            if old is not None:
                (self.root / "maya-ds4-hip.json").write_text(json.dumps(old))
            pc = {"backend": "hip", "gpus": [dict(gpu, index=0)]}
            p = maya.write_config_ds4(args(**kw), pc, {}, Path("/m/G/maya-ds4"), 65536, Path("/models"), Path("/m/G"),
                                      "maya-s")
            a = json.loads(p.read_text())["args"]
            return a[a.index("--reserve-mib") + 1] if "--reserve-mib" in a else None
        card = {"arch": "gfx1100", "name": "Radeon RX 7900 XTX", "vram_gb": 24}
        apu = {"arch": "gfx1151", "name": "Radeon 8060S", "vram_gb": 96}
        self.assertEqual(reserve(card), "3072")                       # (the desktop allocates beside the engine)
        with patch.object(maya, "WIN", False):
            self.assertEqual(reserve(apu), "1024")
        with patch.object(maya, "WIN", True):
            self.assertEqual(reserve(apu), "3072")
        self.assertEqual(reserve(card, vram_reserve_mib=500), "500")  # (asked for)
        self.assertEqual(reserve(card, {"args": ["--reserve-mib", "2000"]}), "2000")   # (kept)
        p, c = self.write(args())                                     # (NVIDIA: the engine's own 600)
        self.assertNotIn("--reserve-mib", c["args"])


class Calibrate(unittest.TestCase):
    """./maya.sh --calibrate on a DeepSeek config: tools/calibrate_ds4_tp.py's result into the config and the store."""

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
            self.assertTrue(maya.calibrate_config(self.cfg))   # (a DeepSeek config: calibrate_ds4_tp's tuning)
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


class Gguf(unittest.TestCase):
    def test_manifest_and_urls(self):
        rev = re.compile(r"/resolve/[0-9a-f]{40}/")
        for q, gb in (("gguf-q8", 508.4), ("gguf-q4", 500.7)):
            m = maya.ds4_gguf_model(q)
            self.assertEqual((m["format"], m["folder"]), ("ds4-gguf", maya.DS4_QUANTS[q]))
            self.assertEqual(len(m["files"]), 9)
            for n, size in m["files"].items():
                self.assertNotIn("/", n)
                self.assertTrue(n.endswith(".gguf"))
                self.assertGreater(size, 0)
                self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(m["download_gb"], gb, delta=0.5)
            u = maya.file_url(m, sorted(m["files"])[0])
            self.assertTrue(u.startswith("https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/"))
            self.assertRegex(u, rev)
            self.assertIn("/%s/" % maya.DS4_QUANTS[q], u)

    def test_earlier_layout_pinned(self):
        """unsloth's layout before its 2026-10-19 re-shard: 10 shards at dc788bdb, each with its size and sha256."""
        for q, gb in (("gguf-q8", 508.4), ("gguf-q4", 500.7)):
            m = maya.ds4_gguf_model(q)
            (e,) = m["earlier"]
            self.assertEqual(e["revision"], "dc788bdb683b0af7a6e5315a7d98e9d082faf5e5")
            self.assertEqual(sorted(e["files"]), ["DeepSeek-V4.1-Flash-1017-%s-%05d-of-00010.gguf" %
                                                  (maya.DS4_QUANTS[q], i) for i in range(1, 11)])
            self.assertTrue(all(re.match(r"^[0-9a-f]{64}$", s) for s in e["sha256"].values()))
            self.assertAlmostEqual(e["download_gb"], gb, delta=0.5)
            self.assertEqual((e["folder"], e["engram"]), (maya.DS4_QUANTS[q], []))

    def test_earlier_layout_used_when_whole(self):
        """A folder that holds unsloth's earlier layout whole is used as it is (nothing downloaded again); not whole,
        or the current layout whole: the current one."""
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        q8 = man["quants"]["UD-Q8_K_XL"]
        for files in [q8["files"]] + [e["files"] for e in q8["earlier"]]:
            for i, k in enumerate(sorted(files)):
                files[k][0] = i + 1
        tmp = Path(tempfile.mkdtemp())
        p = tmp / "download_gguf.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        d = tmp / "UD-Q8_K_XL"
        d.mkdir()
        with patch.object(maya, "DS4_GGUF_MANIFEST", p):
            m = maya.ds4_gguf_model("gguf-q8")
            self.assertIsNone(maya.ds4_earlier(m, d))  # (nothing there: the current layout is downloaded)
            old = sorted(m["earlier"][0]["files"])
            for i, n in enumerate(old):
                (d / n).write_bytes(b"x" * (i + 1))
            e = maya.ds4_earlier(m, d)
            self.assertEqual((sorted(e["files"]), e["revision"], e["engram"], e["layout"]),
                             (old, "dc788bdb683b0af7a6e5315a7d98e9d082faf5e5", None, "earlier"))
            (d / old[-1]).write_bytes(b"x")            # (one cut short)
            self.assertIsNone(maya.ds4_earlier(m, d))
            (d / old[-1]).write_bytes(b"x" * len(old))
            for i, n in enumerate(sorted(m["files"])):   # (the current layout too, whole: that one)
                (d / n).write_bytes(b"x" * (i + 1))
            self.assertIsNone(maya.ds4_earlier(m, d))
            warned = []
            with patch.object(maya, "warn", warned.append):
                maya.ds4_unused(m, d)
            self.assertIn("10 files of an earlier layout of the model, not used", warned[0])

    def test_gguf_config_with_the_vision_tower(self):
        pc = {"gpus": [{"index": 0}]}
        d = Path(tempfile.mkdtemp())
        (d / "maya-ds4").mkdir()
        (d / "maya-ds4" / "ds4.txt").write_text("hidden_size 4096\nvision.num_hidden_layers 32\nimage_token_id 1\n")
        quiet = [patch.object(maya, "ROOT", d), patch.object(maya, "say", lambda *a: None),
                 patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for q in quiet:
            q.start()
        try:
            for a, vision, off in ((args(), None, False), (args(), False, True), (args(no_vision=True), True, True)):
                p = maya.write_config_ds4(a, pc, {}, d / "maya-ds4", 65536, Path("/models"), d, "gguf-q8", vision)
                self.assertEqual("--vision" in json.loads(p.read_text())["args"], off, (a, vision))
        finally:
            for q in quiet:
                q.stop()

    def test_choice_and_folder(self):
        with patch.object(maya, "say", lambda *a: None), patch.object(maya.S, "UI", None), \
                patch.object(maya.S, "gpus", return_value=[{"arch": 86}]):   # (EXL3 offered: an RTX 30)
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant="gguf-q4"), {}), "gguf-q4")
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {"installer": {"ds4_quant": "exl3"}}), "exl3")
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {}), "maya-s")   # (published: the default)
        self.assertEqual(maya.ds4_dir(args(), Path("/models"), {}, "gguf-q8"),
                         Path("/models") / "DeepSeek-V4.1-Flash-1017-GGUF" / "UD-Q8_K_XL")
        self.assertEqual(maya.ds4_dir(args(), Path("/models"), {}, "exl3"), Path("/models") / maya.DS4)


class Vision(unittest.TestCase):
    """A GGUF's vision tower (step 7): DeepSeek's shard 1 and config.json pinned in download_gguf.json, the picture
    markers' 30 KB of shard 2 pinned row by row, and the step's paths - skipped, already there, made, failed."""

    def test_manifest(self):
        m = maya.ds4_vision_model()
        self.assertEqual(m["repo"], "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual(m["files"], {"model-00001-of-00048.safetensors": 970533624, "config.json": 3311})
        self.assertEqual(m["sha256"]["model-00001-of-00048.safetensors"],
                         "13bd121075f0b68cb7e8e8021a1011d42a81a83f557a2602fd9c07d52fddb0ec")
        for n in m["files"]:
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
            u = maya.file_url(m, n)
            self.assertTrue(u.startswith("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/"
                                         "1bae0e1088993b9b82de08f59de5be5a34e42170/"), u)
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text())
        self.assertEqual(man["vision"]["revision"], man["drafter"]["revision"])   # (one checkpoint)
        mk = man["vision"]["marks"]
        rows = mk["rows"]
        self.assertEqual(sorted(rows), ["image_end", "image_newline", "image_start"])
        spans = sorted((o, o + nb) for o, nb, _ in rows.values())
        self.assertTrue(all(nb == 5120 * 2 for _, nb, _ in rows.values()))
        self.assertEqual([a for a, _ in spans[1:]], [b for _, b in spans[:-1]])   # (back to back: one range)
        self.assertEqual(spans[-1][1], mk["bytes"])                               # (the file's last 30 KB)
        self.assertLess(spans[-1][1] - spans[0][0], 1 << 20)

    def setUp(self):
        self.models = Path(tempfile.mkdtemp())
        self.d = self.models / "DeepSeek-V4.1-Flash-1017-GGUF" / "UD-Q8_K_XL"
        self.d.mkdir(parents=True)
        self.v = maya.ds4_vision_dir(self.d)
        self.quiet = [patch.object(maya, "say", lambda *a: None), patch.object(maya, "ok", lambda *a: None),
                      patch.object(maya, "warn", lambda *a: None), patch.object(maya, "step", lambda *a: None),
                      patch.object(maya, "mem_gb", lambda: (64.0, 50.0))]
        for q in self.quiet:
            q.start()

    def tearDown(self):
        for q in self.quiet:
            q.stop()

    def test_folder_is_shared_by_the_quants(self):
        self.assertEqual(self.v, self.models / "DeepSeek-V4.1-Flash-1017-GGUF" / "vision")
        self.assertEqual(maya.ds4_vision_dir(self.d.parent / "UD-Q4_K_XL"), self.v)

    def test_no_vision_downloads_nothing(self):
        with patch.object(maya, "offer_download", side_effect=AssertionError("asked")):
            self.assertFalse(maya.ds4_vision_step(args(no_vision=True), {}, self.d))
        self.assertFalse(self.v.exists())

    def test_already_there(self):
        self.v.mkdir()
        (self.v / "vision.bin").write_bytes(b"x")
        (self.v / "vision.json").write_text("{}")
        with patch.object(maya, "offer_download", side_effect=AssertionError("asked")):
            self.assertTrue(maya.ds4_vision_step(args(), {}, self.d))

    def fake_download(self, a, what, d, files, m):
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            f.write_bytes(b"s" * 8)
        return True

    def fake_model(self):
        import hashlib
        return {"files": {"model-00001-of-00048.safetensors": 8, "config.json": 8},
                "sha256": {n: hashlib.sha256(b"s" * 8).hexdigest() for n in ("model-00001-of-00048.safetensors",
                                                                              "config.json")},
                "repo": "r", "revision": "v", "folder": "", "download_gb": 0.0, "about": "x"}

    def test_made_then_the_shard_is_removed(self):
        def pack(cmd, check=True):
            self.assertIn("vision_pack.py", str(cmd[1]))
            self.assertEqual(cmd[cmd.index("--dir") + 1], str(self.v))
            (self.v / "vision.bin").write_bytes(b"v")
            (self.v / "vision.json").write_text("{}")
            return SimpleNamespace(returncode=0)
        with patch.object(maya, "ds4_vision_model", self.fake_model), \
                patch.object(maya, "offer_download", self.fake_download), patch.object(maya, "run", pack):
            self.assertTrue(maya.ds4_vision_step(args(), {}, self.d))
        self.assertFalse((self.v / "model-00001-of-00048.safetensors").exists())
        self.assertTrue((self.v / "vision.bin").exists())

    def test_failed_markers_keep_the_shard_and_read_text(self):
        with patch.object(maya, "ds4_vision_model", self.fake_model), \
                patch.object(maya, "offer_download", self.fake_download), \
                patch.object(maya, "run", lambda cmd, check=True: SimpleNamespace(returncode=1)):
            self.assertFalse(maya.ds4_vision_step(args(), {}, self.d))
        self.assertTrue((self.v / "model-00001-of-00048.safetensors").exists())   # (not downloaded again next time)

    def test_not_downloaded(self):
        with patch.object(maya, "offer_download", lambda *a: False):
            self.assertFalse(maya.ds4_vision_step(args(), {}, self.d))

    def test_pack_rebuilt_when_the_tower_arrives(self):
        pack = self.d / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        (self.d / "m-00001-of-00001.gguf").write_bytes(b"g")
        for f in ("ds4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
            (pack / f).write_text("hidden_size 4096\n")
        maya.stamp_pack(pack, maya.ds4_shards(self.d))
        rec = json.loads((pack / maya.PACK_SOURCES).read_text())
        rec["tools"] = maya.S.source_hash(maya.DS4_GGUF_PACK_TOOLS)
        (pack / maya.PACK_SOURCES).write_text(json.dumps(rec))
        calls = []
        with patch.object(maya, "run", lambda cmd, **k: calls.append(cmd)):
            maya.ds4_pack_step(args(repack=False), self.d, gguf=True)
        self.assertEqual(calls, [])                    # (built, and no tower to add)
        self.assertFalse(maya.ds4_pack_has_vision(pack))
        self.v.mkdir()
        (self.v / "vision.json").write_text("{}")
        with patch.object(maya, "run", lambda cmd, **k: calls.append(cmd)):
            maya.ds4_pack_step(args(repack=False), self.d, gguf=True, announce=False)
        self.assertEqual(len(calls), 1)                # (built again: the tower arrived after it)
        self.assertEqual(calls[0][calls[0].index("--vision") + 1], str(self.v))


class MayaQuants(unittest.TestCase):
    """Maya's quants of DeepSeek: offered once download_gguf.json lists their files, downloaded into
    <...-GGUF>/Maya-S, their Engram tables Maya's own in Q6_K from the same repo into <...-GGUF>/Engram-Q6_K - or
    unsloth's MXFP8 shard of them in <...-GGUF>/Engram, kept while it is whole (a setup before v1.1.17 put it there)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.quiet = [patch.object(maya, n, lambda *a, **k: None) for n in ("say", "ok", "warn", "step")]
        for q in self.quiet:
            q.start()

    def tearDown(self):
        for q in self.quiet:
            q.stop()

    def unpublished(self):
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        man["maya"]["revision"] = ""
        for q in man["maya"]["quants"].values():
            q["files"] = {}
        p = self.tmp / "download_gguf_unpublished.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        return patch.object(maya, "DS4_GGUF_MANIFEST", p)

    def test_pinned_maya_s(self):
        m = maya.ds4_gguf_model("maya-s")
        self.assertEqual((m["folder"], len(m["files"])), ("Maya-S", 6))
        self.assertRegex(m["revision"], r"^[0-9a-f]{40}$")
        self.assertAlmostEqual(m["download_gb"], 339.48, delta=0.01)
        for n in m["files"]:
            self.assertRegex(n, r"^DeepSeek-V4\.1-Flash-1017-Maya-S-IQ2_XXS-0000[1-6]-of-00006\.gguf$")
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
        self.assertFalse(maya.ds4_maya_published("maya-m"))
        d = maya.ds4_maya_drafter_model()
        self.assertEqual((d["revision"], sorted(d["files"])), (m["revision"], ["dspark.bin", "dspark.json"]))

    def published(self):
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        man["maya"]["revision"] = "0123456789abcdef0123456789abcdef01234567"
        man["maya"]["quants"]["Maya-S"]["files"] = {
            "Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00002.gguf": [48_000_000_000, "a" * 64],
            "Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00002-of-00002.gguf": [30_000_000_000, "b" * 64]}
        p = self.tmp / "download_gguf.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        return patch.object(maya, "DS4_GGUF_MANIFEST", p)

    def test_not_offered_before_they_are_published(self):
        with self.unpublished(), patch.object(maya.S, "UI", None):
            self.assertFalse(maya.ds4_maya_published("maya-s"))
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {}), maya.DS4_DEFAULT_QUANT)
            with self.assertRaises(SystemExit):
                maya.ds4_gguf_model("maya-s")

    def test_published(self):
        with self.published(), patch.object(maya.S, "UI", None):
            self.assertTrue(maya.ds4_maya_published("maya-s"))
            self.assertFalse(maya.ds4_maya_published("maya-m"))
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {}), "maya-s")   # (recommended)
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {"installer": {"ds4_quant": "gguf-q8"}}),
                             "gguf-q8")
            m = maya.ds4_gguf_model("maya-s")
            self.assertEqual((m["format"], m["folder"], len(m["files"])), ("ds4-gguf", "Maya-S", 2))
            self.assertAlmostEqual(m["download_gb"], 78.0, delta=0.01)
            u = maya.file_url(m, sorted(m["files"])[0])
            self.assertEqual(u, "https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/resolve/"
                                "0123456789abcdef0123456789abcdef01234567/Maya-S/"
                                "DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00002.gguf")
        self.assertEqual(maya.ds4_dir(args(), Path("/models"), {}, "maya-s"),
                         Path("/models") / "DeepSeek-V4.1-Flash-1017-GGUF" / "Maya-S")

    def test_drafter_from_mayas_repo_once_pinned(self):
        with self.unpublished():
            self.assertIsNone(maya.ds4_maya_drafter_model())   # (no revision pinned: DeepSeek's shards instead)
        with self.published():
            m = maya.ds4_maya_drafter_model()
            self.assertEqual(sorted(m["files"]), ["dspark.bin", "dspark.json"])
            self.assertEqual(m["folder"], "dspark")
            self.assertTrue(maya.file_url(m, "dspark.bin").endswith("/dspark/dspark.bin"))
            d = self.tmp / "dspark"
            with patch.object(maya, "offer_download", return_value=True) as dl, \
                    patch.object(maya, "run", side_effect=AssertionError("converted")):
                self.assertTrue(maya.ds4_drafter_step(args(), d))
            self.assertEqual(dl.call_count, 1)

    def test_maya_s_one_split(self):
        """Maya-S is one GGUF split of 6 shards with its Engram tables (Q6_K, shards 5 and 6), from Maya's repo."""
        m = maya.ds4_gguf_model("maya-s")
        self.assertEqual(sorted(m["files"]), ["DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-%05d-of-00006.gguf" % i
                                              for i in range(1, 7)])
        self.assertAlmostEqual(m["download_gb"], 339.5, delta=0.1)
        self.assertEqual(m["repo"], "peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF")
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        self.assertNotIn("engram", man["maya"])        # (nothing apart: the tables are in the quant's own shards)

    def test_v1116_layout_pinned(self):
        """v1.1.16's Maya-S (4 shards at 26c6aea) and unsloth's MXFP8 Engram shard beside it, in either of its
        layouts, are the earlier layout the setup keeps."""
        (e,) = maya.ds4_gguf_model("maya-s")["earlier"]
        self.assertEqual(e["revision"], "26c6aeaf5ecce88fe47103f780279bcc6f1f0d9b")
        self.assertEqual(sorted(e["files"]), ["DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-%05d-of-00004.gguf" % i
                                              for i in range(1, 5)])
        self.assertAlmostEqual(e["download_gb"], 178.19, delta=0.01)
        self.assertEqual([sorted(s["files"]) for s in e["engram"]],
                         [["DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00009.gguf"],
                          ["DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00010.gguf",
                           "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00010.gguf"]])
        self.assertEqual([round(s["download_gb"], 1) for s in e["engram"]], [202.8, 202.8])

    def tiny(self):
        """download_gguf.json with tiny files (a file's size: its place in its list + 1)."""
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        q = man["maya"]["quants"]["Maya-S"]
        e = q["earlier"][0]
        for files in [q["files"], e["files"]] + e["engram"]:
            for i, k in enumerate(sorted(files)):
                files[k][0] = i + 1
        p = self.tmp / "download_gguf_tiny.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        self.base = self.tmp / "DeepSeek-V4.1-Flash-1017-GGUF"
        self.d = self.base / "Maya-S"
        self.d.mkdir(parents=True)
        return patch.object(maya, "DS4_GGUF_MANIFEST", p)

    def put(self, d: Path, names):
        d.mkdir(parents=True, exist_ok=True)
        for i, n in enumerate(sorted(names)):
            (d / n).write_bytes(b"x" * (i + 1))

    def test_v1116_install_kept(self):
        """v1.1.16's Maya-S with its Engram shard whole: used as it is, nothing downloaded again - whichever layout of
        unsloth's shard it has."""
        with self.tiny():
            m = maya.ds4_gguf_model("maya-s")
            e = m["earlier"][0]
            self.put(self.d, e["files"])
            self.assertIsNone(maya.ds4_earlier(m, self.d))   # (no Engram tables beside it: the 6-shard set)
            for s in e["engram"]:
                eg = self.base / "Engram"
                if eg.exists():
                    for f in eg.iterdir():
                        f.unlink()
                self.put(eg, s["files"])
                old = maya.ds4_earlier(m, self.d)
                self.assertEqual((sorted(old["files"]), old["engram"], old["revision"]),
                                 (sorted(e["files"]), eg, "26c6aeaf5ecce88fe47103f780279bcc6f1f0d9b"))
            first = sorted(e["engram"][1]["files"])[0]
            (eg / first).write_bytes(b"")              # (the Engram shard cut short: the 6-shard set)
            self.assertIsNone(maya.ds4_earlier(m, self.d))
            (eg / first).write_bytes(b"x")
            self.put(self.d, m["files"])               # (the 6-shard set whole too: that one)
            self.assertIsNone(maya.ds4_earlier(m, self.d))
            warned = []
            with patch.object(maya, "warn", warned.append):
                maya.ds4_unused(m, self.d)             # (the earlier files: the user's to delete)
            self.assertEqual(len(warned), 2)
            self.assertTrue(warned[0].startswith("%s: 4 files of an earlier layout" % self.d))
            self.assertTrue(warned[1].startswith("%s: 2 files of an earlier layout" % eg))

    def test_v1116_install_kept_by_the_setup(self):
        """set_up_ds4 with v1.1.16's Maya-S whole: no download, the pack indexes its 4 shards and Engram/ (--shards,
        --engram)."""
        with self.tiny():
            m = maya.ds4_gguf_model("maya-s")
            e = m["earlier"][0]
            self.put(self.d, e["files"])
            self.put(self.base / "Engram", e["engram"][0]["files"])
            packs = []
            with patch.object(maya, "check_pc", return_value={"gpus": [{"index": 0}]}), \
                    patch.object(maya, "choose_context", return_value=65536), \
                    patch.object(maya, "tools_step", return_value=None), patch.object(maya, "build_step", return_value={}), \
                    patch.object(maya, "offer_download", side_effect=AssertionError("asked to download")), \
                    patch.object(maya, "ds4_drafter_step", return_value=False), \
                    patch.object(maya, "ds4_vision_step", return_value=False), \
                    patch.object(maya, "ds4_pack_step", lambda *a, **k: packs.append((a, k)) or self.d / "maya-ds4"), \
                    patch.object(maya, "write_config_ds4", return_value=self.tmp / "maya-ds4.json"), \
                    patch.object(maya.CDS, "has_drafter", return_value=False):
                maya.set_up_ds4(args(ds4_quant="maya-s", model_dir=str(self.d), models_dir=str(self.tmp)), {})
            (a, k), = packs
            self.assertEqual(sorted(f.name for f in k["files"]), sorted(e["files"]))
            self.assertEqual(k["engram"], self.base / "Engram")

    def test_pack_follows_the_engram_folder(self):
        """The pack indexes the files and the Engram folder the setup picked (--shards, --engram), and is built again
        when they change (v1.1.16's 4 shards + Engram/ -> the 6-shard set)."""
        old = ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
        new = ["m-00001-of-00001.gguf"]
        base = self.tmp / "DeepSeek-V4.1-Flash-1017-GGUF"
        d = base / "Maya-S"
        for n in old + new:
            self.put(d, [n])
        self.put(base / "Engram", ["u-00008-of-00009.gguf"])
        pack = d / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        for f in ("ds4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
            (pack / f).write_text("hidden_size 4096\n")
        maya.stamp_pack(pack, [d / n for n in old] + sorted((base / "Engram").glob("*.gguf")))
        rec = json.loads((pack / maya.PACK_SOURCES).read_text())
        rec["tools"] = maya.S.source_hash(maya.DS4_GGUF_PACK_TOOLS)
        (pack / maya.PACK_SOURCES).write_text(json.dumps(rec))
        calls = []
        with patch.object(maya, "run", lambda cmd, **k: calls.append(cmd)):
            maya.ds4_pack_step(args(repack=False), d, gguf=True, engram=base / "Engram", files=[d / n for n in old])
            self.assertEqual(calls, [])                # (built with these files)
            maya.ds4_pack_step(args(repack=False), d, gguf=True, files=[d / n for n in new])
        self.assertEqual(len(calls), 1)                # (built again: the 6-shard set now)
        self.assertNotIn("--engram", calls[0])
        self.assertEqual(calls[0][calls[0].index("--shards") + 1:], new)


class Rename(unittest.TestCase):
    """v1.1.16's program names: an earlier version's programs renamed in place, the stamps and configs follow."""

    def test_programs_and_stamp(self):
        root = Path(tempfile.mkdtemp())
        ext = ".exe" if maya.WIN else ""
        (root / "build").mkdir()
        (root / "build-vision" / "bin").mkdir(parents=True)
        (root / "build" / ("strata" + ext)).write_text("glm")
        (root / "build" / ("strata-ds4" + ext)).write_text("ds4")
        (root / "build-vision" / "bin" / ("strata-vision" + ext)).write_text("vision")
        (root / "build" / "MAYA-BUILD.json").write_text(json.dumps({"src": "s", "targets": ["strata", "strata-ds4"]}))
        with patch.object(maya, "ROOT", root), patch.object(maya, "VBUILD", root / "build-vision"):
            maya.rename_programs()
            maya.rename_programs()   # (again: nothing left to do)
        self.assertEqual((root / "build" / ("maya" + ext)).read_text(), "glm")
        self.assertEqual((root / "build" / ("maya-ds4" + ext)).read_text(), "ds4")
        self.assertEqual((root / "build-vision" / "bin" / ("maya-vision" + ext)).read_text(), "vision")
        self.assertFalse((root / "build" / ("strata" + ext)).exists())
        self.assertEqual(json.loads((root / "build" / "MAYA-BUILD.json").read_text())["targets"], ["maya", "maya-ds4"])

    def test_config(self):
        m = Path(tempfile.mkdtemp())
        (m / "build").mkdir()
        (m / "build-vision" / "bin").mkdir(parents=True)
        for p in ("build/maya.exe", "build/maya-ds4", "build-vision/bin/maya-vision"):
            (m / p).write_text("")
        cfg = {"exe": str(m / "build" / "strata.exe"), "vision": {"exe": str(m / "build-vision/bin/strata-vision")}}
        self.assertTrue(maya.fix_config_names(cfg))
        self.assertEqual(Path(cfg["exe"]).name, "maya.exe")
        self.assertEqual(Path(cfg["vision"]["exe"]).name, "maya-vision")
        self.assertFalse(maya.fix_config_names(cfg))
        ds4 = {"exe": str(m / "build" / "strata-ds4")}
        self.assertEqual(maya.config_family(ds4), "deepseek")   # (read before it is fixed too)
        self.assertTrue(maya.fix_config_names(ds4))
        self.assertEqual(Path(ds4["exe"]).name, "maya-ds4")
        other = {"exe": str(m / "elsewhere" / "strata")}   # (no program under the new name there: kept)
        self.assertFalse(maya.fix_config_names(other))
        self.assertEqual(Path(other["exe"]).name, "strata")


@unittest.skipUnless(maya.shutil.which("curl"), "no curl")
class Download(unittest.TestCase):
    """The model download (fetch_files): files fetched side by side, each resumed from its size on disk, and a file
    that keeps stopping without progress ends the setup (it continues on the next run)."""

    def files(self, n=3, size=300_000):
        src, dst = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        data = {f"m-{i:05d}.gguf": bytes((i * 7 + k) % 251 for k in range(size + i)) for i in range(n)}
        for name, b in data.items():
            (src / name).write_bytes(b)
        m = {"files": {name: len(b) for name, b in data.items()}, "download_gb": 0.001}
        return src, dst, data, m

    def test_resumes_each_file_from_its_size(self):
        src, dst, data, m = self.files()
        first = sorted(data)[0]
        (dst / first).write_bytes(data[first][:100_000])   # (half way: an earlier run stopped there)
        shards = [dst / name for name in sorted(data)]
        urls = {s: (src / s.name).as_uri() for s in shards}
        with patch.object(maya, "hf_name", lambda mm, s: s.name):
            maya.fetch_files(m, shards, shards, urls)
        for s in shards:
            self.assertEqual(s.read_bytes(), data[s.name])

    def test_a_file_that_never_arrives_stops_the_setup(self):
        src, dst, data, m = self.files(n=1)
        s = dst / sorted(data)[0]
        with patch.object(maya, "hf_name", lambda mm, x: x.name), patch.object(maya, "DL_STUCK", 1), \
                self.assertRaises(SystemExit):
            maya.fetch_files(m, [s], [s], {s: (src / "nothing-here.gguf").as_uri()})


class Start(unittest.TestCase):
    def test_server_command(self):
        d = Path(tempfile.mkdtemp())
        pack = d / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        exe = d / "build" / "maya-ds4"
        exe.parent.mkdir()
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


class StartingProfile(unittest.TestCase):
    """DeepSeek's shipped expert profile: copied into a pack that has none (maya.ds4_seed_profile), and how
    tools/ds4/profile.py makes it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pack = self.root / "maya-ds4"
        self.pack.mkdir()
        (self.pack / "ds4.txt").write_text("num_hidden_layers 2\nn_routed_experts 4\nnum_experts_per_tok 2\n")
        self.shipped = self.root / "expert-profile-ds4.bin"
        self.shipped.write_bytes(bytes(range(32)))
        self.p = patch.object(maya, "DS4_PROFILE", self.shipped)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def cfg(self, *extra):
        return {"args": ["--pack", str(self.pack), "--max-context", "65536", *extra]}

    def test_seeds_a_pack_without_one(self):
        self.assertEqual(maya.ds4_seed_profile(self.cfg()), self.pack / "expert-profile.bin")
        self.assertEqual((self.pack / "expert-profile.bin").read_bytes(), bytes(range(32)))

    def test_leaves_a_learned_one_and_an_explicit_one(self):
        (self.pack / "expert-profile.bin").write_bytes(b"\1" * 32)
        self.assertIsNone(maya.ds4_seed_profile(self.cfg()))           # (what this PC learned stays)
        self.assertEqual((self.pack / "expert-profile.bin").read_bytes(), b"\1" * 32)
        (self.pack / "expert-profile.bin").unlink()
        self.assertIsNone(maya.ds4_seed_profile(self.cfg("--expert-profile", "none")))
        self.assertFalse((self.pack / "expert-profile.bin").exists())

    def test_only_the_packs_geometry(self):
        self.shipped.write_bytes(bytes(40))
        self.assertIsNone(maya.ds4_seed_profile(self.cfg()))
        self.assertFalse((self.pack / "expert-profile.bin").exists())

    def test_make_and_compare(self):
        import importlib
        prof = importlib.import_module("ds4.profile")
        a = [[10, 0, 30, 60], [1, 1, 1, 1]]
        b = [[1, 0, 3, 6], [0, 0, 0, 4]]
        out = prof.make([a, b], 2, 4, 2, tokens=100)
        self.assertEqual(out[0], [20, 0, 60, 120])                     # (equal weights: the same shares; k x tokens)
        self.assertEqual(sum(out[1]), 200)
        self.assertEqual(out[1].index(max(out[1])), 3)
        self.assertEqual(prof.agree(a, b, 2), [1.0, 0.5])


class Drafting(unittest.TestCase):
    """The drafter chosen by the user instead of the speed calibration: --drafting, the question, and the config."""
    def setUp(self):
        self.quiet = [patch.object(maya, n, lambda *a, **k: None) for n in ("say", "ok")]
        for p in self.quiet:
            p.start()
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for p in self.quiet:
            p.stop()

    def test_choice(self):
        choice = lambda **kw: maya.drafting_choice(args(**kw), "The DSpark drafter", "10-30")   # noqa: E731
        self.assertEqual(choice(drafting="off"), "off")
        self.assertEqual(choice(drafting="on", no_speed_calibration=False), "on")
        self.assertEqual(choice(no_speed_calibration=True), "on")
        self.assertEqual(choice(), "measure")                             # (--yes: measured)
        for answer, how in (("m", "measure"), ("y", "on"), ("n", "off")):
            with patch.object(maya, "ask", return_value=answer) as asked:
                self.assertEqual(choice(yes=False), how)
            self.assertEqual(asked.call_args[0][2], "m")                  # (measuring is the default answer)

    def write(self, cfg: dict) -> Path:
        p = self.dir / "c.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return p

    def test_deepseek_kept_as_chosen(self):
        pack = self.dir / "maya-ds4"
        pack.mkdir()
        (pack / "tensors.txt").write_text("t mtp.0.attn.wq_a.weight BF16 0 0 2 1\n")
        for on, mode in ((True, "on"), (False, "none")):
            p = self.write({"exe": "/m/build/maya-ds4", "args": ["--pack", str(pack), "--dspark", "none"],
                            "installer": {"family": "deepseek"}})
            self.assertTrue(maya.needs_speed_calibration(json.loads(p.read_text())))
            maya.keep_drafting(p, on)
            c = json.loads(p.read_text())
            self.assertEqual(c["args"][-2:], ["--dspark", mode])
            self.assertEqual(c["args"].count("--dspark"), 1)
            self.assertEqual(c["installer"]["speed_calibration"]["kept"], "dspark" if on else "none")
            self.assertFalse(maya.needs_speed_calibration(c))             # (a start does not measure by itself)

    def test_glm_kept_as_chosen(self):
        cfg = {"exe": "/m/build/maya", "args": ["--glm-pack", "/m/pack", "--dflash", "/m/d.bin"], "gpu": [0, 1]}
        p = self.write(cfg)
        maya.keep_drafting(p, False)
        c = json.loads(p.read_text())
        self.assertNotIn("--dflash", c["args"])
        self.assertEqual((c["installer"]["draft"], c["installer"]["speed_calibration"]["kept"]), ("mtp", "mtp"))
        p = self.write(cfg)
        maya.keep_drafting(p, True)
        c = json.loads(p.read_text())
        self.assertEqual(c["args"][-2:], ["--dflash", "/m/d.bin"])
        self.assertEqual(c["installer"]["draft"], "dflash2")
        self.assertFalse(maya.needs_speed_calibration(c))


if __name__ == "__main__":
    unittest.main()
