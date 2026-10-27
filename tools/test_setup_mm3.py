"""tools/test_setup_mm3.py - the setup's MiniMax-M3.1-Flash (maya.py): where its EXL3 quant is offered, the context it
recommends by the PC's memory (and its warnings), its config - the model card's sampling, the thinking level, pictures
on - and what a setup again keeps.  No GPU, no network.

    python -m unittest tools.test_setup_mm3 -v
"""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402


class Failed(Exception):
    pass


def args(**kw):
    base = dict(model=maya.MM3, model_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, port=None,
                host=None, api_key=None, env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None)
    base.update(kw)
    return SimpleNamespace(**base)


def fail(msg, hint=None):
    raise Failed(msg)


class Gate(unittest.TestCase):
    """NVIDIA, and AMD as GLM's EXL3 quants there (experimental, said); Windows and V100 once the engine runs there
    (MM3_ALL_PLATFORMS, one switch)."""

    def check(self, backend="cuda", win=False, everyone=False, gpus=None):
        with patch.object(maya, "fail", fail), patch.object(maya, "WIN", win), \
                patch.object(maya, "MM3_ALL_PLATFORMS", everyone), patch.object(maya, "step", lambda *a: None), \
                patch.object(maya.S, "gpus", return_value=gpus or []), patch.object(maya, "say", lambda *a: None), \
                patch.object(maya, "warn", lambda *a: None), patch.object(maya.S, "is_wsl", return_value=False):
            maya.check_pc(args(backend=backend), "minimax")

    def test_amd_experimental(self):
        """AMD: the EXL3 quant goes on to the AMD PC check with an experimental warning (as GLM's EXL3 there), the
        GGUFs without one; neither is refused, and MiniMax's HIP check does not say "text only"."""
        for gguf in (False, True):
            said, hip = [], []
            with patch.object(maya, "fail", fail), patch.object(maya, "step", lambda *a: None), \
                    patch.object(maya, "warn", said.append), \
                    patch.object(maya, "check_hip_pc", lambda a, **kw: hip.append(kw) or {"backend": "hip"}):
                self.assertEqual(maya.check_pc(args(backend="hip"), "minimax", gguf=gguf), {"backend": "hip"})
            self.assertEqual(hip, [{"text_only": False}])
            self.assertEqual(len(said), 0 if gguf else 1)
            if not gguf:
                self.assertIn("experimental", said[0])

    def test_amd_quants_offered(self):
        """AMD is offered the EXL3 quant and the published GGUFs, as NVIDIA is (the EXL3 one marked experimental)."""
        seen = {}

        def choose(title, intro, rows, default, auto):
            seen["rows"] = rows
            return default

        for backend, amd in (("hip", True), ("cuda", False)):
            with patch.object(maya, "mm3_gguf_published", return_value=True), patch.object(maya, "choose", choose), \
                    patch.object(maya, "mem_gb", return_value=(64.0, 60.0)), \
                    patch.object(maya.S, "gpus", return_value=[{"vram_gb": 32.0}]), \
                    patch.object(maya.S, "amd_gpus", return_value=[{"vram_gb": 96.0}]), \
                    patch.object(maya, "mm3_models", return_value=({"download_gb": 197.0}, {})):
                maya.choose_mm3_quant(args(backend=backend, mm3_quant=None), {})
            kind = lambda t: "EXL3" if "EXL3" in t else "Maya-XS" if "Maya-XS" in t else "Maya-S" if "Maya-S" in t else t  # noqa: E731
            labels = {kind(r[0]): r for r in seen["rows"]}
            self.assertEqual(sorted(labels), ["EXL3", "Maya-S", "Maya-XS"])
            self.assertTrue(all(r[0].startswith("MiniMax M3.1") for r in seen["rows"]))   # (the model's name first)
            self.assertEqual("experimental on AMD" in labels["EXL3"][2], amd)
            self.assertTrue(all("text only" not in r[2] for r in seen["rows"]))
            # (the recommendation counts the GPUs' memory: 64 + 96 GB on AMD holds Maya-S, 64 + 32 GB Maya-XS)
            rec = [r[0] for r in seen["rows"] if r[1] == "recommended"]
            want = "MiniMax M3.1 · Maya-S (122 GB)" if amd else "MiniMax M3.1 · Maya-XS (93 GB)"
            self.assertEqual(rec, [want])

    def test_amd_companions(self):
        """A GGUF's pictures and drafting on AMD: the EXL3 repo's files, as on NVIDIA (no AMD-only text-only stop)."""
        d = Path(tempfile.mkdtemp()) / "MiniMax-M3.1-Flash-Maya-XS"
        exl3 = d.parent / maya.MM3_DIR
        (exl3 / "dspark").mkdir(parents=True)
        for n in maya.MM3_VISION_FILES + ("dspark/config.json",):
            (exl3 / n).write_text("{}")
        with patch.object(maya, "ok", lambda *a: None):
            self.assertEqual(maya.mm3_companions_step(args(backend="hip"), d, True), exl3)

    def test_windows_waits_for_the_switch(self):
        with self.assertRaisesRegex(Failed, "Linux only for now"):
            self.check(win=True)
        with self.assertRaisesRegex(Failed, "no NVIDIA GPU"):       # (past the gate: the GPU check)
            self.check(win=True, everyone=True)

    def test_min_cc(self):
        with patch.object(maya, "warn", lambda *a: None):
            with patch.object(maya, "MM3_ALL_PLATFORMS", False), patch.dict(maya.os.environ, {}, clear=False):
                maya.os.environ.pop(maya.MM3_TEST_CC_ENV, None)
                self.assertEqual(maya.mm3_min_cc(), 80)
                maya.os.environ[maya.MM3_TEST_CC_ENV] = "70"
                self.assertEqual(maya.mm3_min_cc(), 70)               # (testing only, opt-in)
                maya.os.environ.pop(maya.MM3_TEST_CC_ENV)
            with patch.object(maya, "MM3_ALL_PLATFORMS", True):
                self.assertEqual(maya.mm3_min_cc(), maya.DS4_MIN_CC)   # (DeepSeek's EXL3 reach: V100 on)
        self.assertEqual(maya.MM3_MIN_CC, maya.DS4_MIN_CC)

    def test_exe_by_platform(self):
        self.assertEqual(maya.exe_for_target("maya-mm3"), maya.MM3_EXE)
        maya.select_build_backend("cuda")
        self.assertEqual(maya.MM3_EXE.name, "maya-mm3.exe" if maya.WIN else "maya-mm3")


class Contexts(unittest.TestCase):
    def test_up_to_the_models_own(self):
        self.assertEqual(maya.MM3_CONTEXTS, [32768, 65536, 131072, 262144, 524288, 1048576])

    def test_recommended_by_memory(self):
        """The largest tested size whose KV cache fits beside the model in VRAM + RAM; the smallest when it does not."""
        bf16 = maya.MM3_KV_BYTES["bf16"]
        self.assertAlmostEqual(maya.mm3_kv_gb(1048576, "bf16"), 1048576 * bf16 / 2**30)
        self.assertEqual(maya.mm3_recommended_context(48, 512, "bf16"), 262144)    # (never past the tested 256K)
        self.assertEqual(maya.mm3_recommended_context(48, 256, "bf16"), 262144)    # (54 GB spare: 30 GB at 256K)
        self.assertEqual(maya.mm3_recommended_context(48, 220, "bf16"), 131072)    # (18 GB spare: 15 GB at 128K)
        self.assertEqual(maya.mm3_recommended_context(24, 128, "bf16"), 32768)     # (below the model: the smallest)
        self.assertEqual(maya.mm3_recommended_context(48, 0, "bf16"), maya.MM3_DEFAULT_CONTEXT)   # (RAM unknown)
        self.assertGreater(maya.mm3_recommended_context(48, 214, "int8"), maya.mm3_recommended_context(48, 214, "bf16"))

    def test_kv_format_from_the_config(self):
        self.assertEqual(maya.mm3_kv_format(["--kv-cache", "int8"]), "int8")
        self.assertEqual(maya.mm3_kv_format([]), maya.MM3_KV_FORMAT)

    def test_warnings(self):
        said = []
        with patch.object(maya, "warn", said.append), patch.object(maya, "ok", lambda *a: None):
            maya.mm3_context_notes(262144, 48, 256, "bf16")
            self.assertEqual(said, [])
            maya.mm3_context_notes(1048576, 48, 256, "bf16")
        self.assertEqual(len(said), 2)
        self.assertIn("not been run end to end", said[0])
        self.assertIn("256K fits this PC", said[1])


class Config(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pack = self.root / "MM" / "maya-mm3"
        self.pack.mkdir(parents=True)
        (self.pack / "mm3.txt").write_text("hidden_size 6144\nmax_position_embeddings 1048576\n"
                                           "vision.hidden_size 1024\ndspark.num_hidden_layers 5\n")
        self.patches = [patch.object(maya, "ROOT", self.root), patch.object(maya, "say", lambda *a: None),
                        patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def write(self, a, old=None, gpus=3):
        if old is not None:
            (self.root / "maya-mm3.json").write_text(json.dumps(old))
        pc = {"gpus": [{"index": i} for i in range(gpus)]}
        p = maya.write_config_mm3(a, pc, {"lib_dirs": []}, self.pack, 131072, Path("/models"), self.pack.parent)
        return json.loads(p.read_text())

    def test_fresh(self):
        c = self.write(args())
        a = c["args"]
        self.assertEqual((c["sampling"], c["reasoning_effort"]), ({"temperature": 1.0, "top_p": 0.95}, "auto"))
        self.assertEqual(a[a.index("--vision") + 1], "on")                # (pictures on by default)
        self.assertEqual(a[a.index("--dspark") + 1], "off")             # (until this PC's speed calibration)
        self.assertEqual(a[a.index("--kv-streaming") + 1], "on")
        self.assertEqual(a[a.index("--max-context") + 1], "131072")
        self.assertEqual(c["model_name"], "minimax-m3.1-flash")
        self.assertEqual(maya.config_family(c), "minimax")

    def test_no_vision(self):
        a = self.write(args(no_vision=True))["args"]
        self.assertEqual(a[a.index("--vision") + 1], "off")

    def test_again_keeps_what_it_does_not_ask(self):
        old = {"args": ["--pack", "/old", "--max-context", "262144", "--kv-cache", "int8", "--vision", "off"],
               "reasoning_effort": "high", "sampling": {"temperature": 0.7}, "host": "0.0.0.0"}
        c = self.write(args(), old)
        a = c["args"]
        self.assertEqual(a[a.index("--kv-cache") + 1], "int8")
        self.assertEqual(a[a.index("--vision") + 1], "off")
        self.assertEqual((c["reasoning_effort"], c["sampling"], c["host"]), ("high", {"temperature": 0.7}, "0.0.0.0"))

    def test_tp_follows_the_gpus_chosen_now(self):
        """#436 (DeepSeek's, the same rule here): the config before's --tp only for the same GPUs"""
        nine = {"args": ["--pack", "/old", "--max-context", "65536", "--tp", "4"], "gpu": list(range(9))}
        a = self.write(args(), nine, gpus=9)["args"]
        self.assertEqual(a[a.index("--tp") + 1], "4")                      # (the same GPUs: kept)
        a = self.write(args(), nine, gpus=1)["args"]
        self.assertNotIn("--tp", a)                                        # (one GPU: no tensor parallel)
        a = self.write(args(), nine, gpus=5)["args"]
        self.assertEqual(a[a.index("--tp") + 1], "4")                      # (5 GPUs: 1 + a group of 4, worked out)
        a = self.write(args(), {"args": ["--tp", "4"], "gpu": [0, 1, 2, 3, 4]}, gpus=3)["args"]
        self.assertEqual(a[a.index("--tp") + 1], "2")                      # (3 GPUs: 1 + a group of 2)

    def test_dspark_measured_as_deepseeks(self):
        """The DSpark drafter: off in the config until this PC's speed calibration (DeepSeek's tool) says it pays."""
        import calibrate_ds4 as CDS
        c = self.write(args())
        self.assertFalse(CDS.has_drafter(c))                               # (no dspark.* tensors indexed yet)
        (self.pack / "tensors.txt").write_text("file 0 a.safetensors\nt dspark.fc.weight BF16 0 0 8 2 2\n")
        self.assertTrue(CDS.has_drafter(c))
        self.assertTrue(maya.needs_speed_calibration(c))
        a = CDS.with_dspark(c, "none")["args"]
        self.assertEqual(a[a.index("--dspark") + 1], "off")                # (maya-mm3's word for none)
        self.assertEqual(CDS.profile_copy(c, self.root)["args"], c["args"])  # (it only reads its --profile)

    def test_template_in_the_pack_tools(self):
        """A change to Maya's MiniMax template rebuilds the pack (its tokenizer folder carries the template)."""
        self.assertIn("tools/mm3/chat_template.jinja", maya.MM3_PACK_TOOLS)
        self.assertTrue((Path(maya.__file__).parent / "tools" / "mm3" / "chat_template.jinja").exists())

    def test_gguf(self):
        """A GGUF quant's config: no pictures, no DSpark, no tensor parallel - kept from an EXL3 config before neither -,
        the quant in the installer record."""
        (self.pack / "mm3.txt").write_text("hidden_size 6144\nmax_position_embeddings 1048576\nformat gguf\n")
        old = {"args": ["--pack", "/old", "--max-context", "65536", "--vision", "on", "--tp", "2", "--dspark", "on",
                        "--kv-cache", "int8"]}
        (self.root / "maya-mm3.json").write_text(json.dumps(old))
        pc = {"gpus": [{"index": i} for i in range(3)]}
        p = maya.write_config_mm3(args(), pc, {"lib_dirs": []}, self.pack, 131072, Path("/models"), self.pack.parent,
                                  "gguf-xs")
        c = json.loads(p.read_text())
        for k in ("--vision", "--tp", "--dspark"):
            self.assertNotIn(k, c["args"])
        self.assertEqual(c["args"][c["args"].index("--kv-cache") + 1], "int8")
        self.assertEqual((c["installer"]["mm3_quant"], c["installer"]["quant"]), ("gguf-xs", "MiniMax-M3.1-Flash-Maya-XS"))
        self.assertEqual(p.name, "maya-minimax-maya-xs.json")                # (the naming scheme's config)
        # a GGUF pack with the companions' vision tower and draft: pictures on, the draft off until measured
        (self.pack / "mm3.txt").write_text("format gguf\nvision.hidden_size 1280\ndspark.num_hidden_layers 5\n")
        (self.root / "maya-mm3.json").unlink()
        p = maya.write_config_mm3(args(), pc, {"lib_dirs": []}, self.pack, 131072, Path("/models"), self.pack.parent,
                                  "gguf-s")
        a = json.loads(p.read_text())["args"]
        self.assertEqual((a[a.index("--vision") + 1], a[a.index("--dspark") + 1]), ("on", "off"))
        self.assertNotIn("--tp", a)


class Gguf(unittest.TestCase):
    """Maya's GGUF quants of MiniMax: the manifest (unpublished: no revision, no sha256), the choice, AMD allowed for
    them (and the EXL3 quant, experimental), the folder and the pack's tool."""

    def test_manifest(self):
        for q, name in (("gguf-xs", "MiniMax-M3.1-Flash-Maya-XS"), ("gguf-s", "MiniMax-M3.1-Flash-Maya-S")):
            m = maya.mm3_gguf_entry(q)
            self.assertEqual(m["folder"], name[len("MiniMax-M3.1-Flash-"):])     # (the repo's Maya-XS/, Maya-S/)
            self.assertTrue(any(n.endswith("-maya-indexers.gguf") for n in m["files"]))
            self.assertTrue(all(n.startswith(name) for n in m["files"]))
            self.assertGreater(m["memory_gb"], m["download_gb"])
            self.assertTrue(maya.mm3_gguf_published(q))               # (uploaded: the revision and every sha256)
        self.assertEqual(maya.mm3_memory_gb("exl3"), maya.MM3_MEMORY_GB)

    def test_choice(self):
        with patch.object(maya, "fail", fail):
            self.assertEqual(maya.choose_mm3_quant(args(mm3_quant="gguf-s"), {}), "gguf-s")
            # the one recommended by the PC's VRAM + RAM: the EXL3 quant where it fits (250 GB), else the biggest GGUF
            # that fits, else the smallest
            for ram, vram, want in ((256, 96, "exl3"), (128, 32, "gguf-s"), (96, 24, "gguf-xs"), (32, 8, "gguf-xs")):
                with patch.object(maya, "mem_gb", return_value=(float(ram), float(ram))), \
                        patch.object(maya.S, "gpus", return_value=[{"vram_gb": float(vram)}]):
                    self.assertEqual(maya.choose_mm3_quant(args(mm3_quant=None), {}), want, (ram, vram))
            with patch.object(maya.S, "amd_gpus", return_value=[]), \
                    patch.object(maya, "mem_gb", return_value=(128.0, 128.0)):   # (AMD too: a 128 GB Strix Halo)
                self.assertEqual(maya.choose_mm3_quant(args(mm3_quant=None, backend="hip"), {}), "gguf-xs")

    def test_amd_allowed_for_gguf(self):
        with patch.object(maya, "fail", fail), patch.object(maya, "step", lambda *a: None), \
                patch.object(maya, "check_hip_pc", return_value={"gpus": []}) as hip:
            self.assertEqual(maya.check_pc(args(backend="hip"), "minimax", gguf=True), {"gpus": []})
            hip.assert_called_once()
            with patch.object(maya, "warn", lambda *a: None):   # (the EXL3 quant: experimental on AMD, not refused)
                self.assertEqual(maya.check_pc(args(backend="hip"), "minimax"), {"gpus": []})

    def test_folder_and_tools(self):
        self.assertEqual(maya.mm3_dir(args(), Path("/m"), {}, "gguf-xs"), Path("/m/MiniMax-M3.1-Flash-Maya-XS"))
        self.assertEqual(maya.mm3_dir(args(), Path("/m"), {}), Path("/m") / maya.MM3_DIR)
        for t in maya.MM3_GGUF_PACK_TOOLS:
            self.assertTrue((Path(maya.__file__).parent / t).exists(), t)

    def test_companions(self):
        """Pictures and drafting for a GGUF, on NVIDIA and AMD alike: a whole EXL3 download beside it as it is; else the
        EXL3 repo's vision tower and draft in the companions folder (downloaded when not there - here already whole)."""
        quiet = [patch.object(maya, k, lambda *a, **kw: None) for k in ("ok", "warn", "say", "step")]
        for p in quiet:
            p.start()
        try:
            with tempfile.TemporaryDirectory() as t:
                models = Path(t)
                d = models / "MiniMax-M3.1-Flash-Maya-XS"
                m, dm = maya.mm3_models()
                small = dict(m, files={n: 10 for n in maya.MM3_VISION_FILES}, sha256={n: "" for n in maya.MM3_VISION_FILES})
                small_d = dict(dm, files={n: 10 for n in dm["files"]}, sha256={n: "" for n in dm["files"]})
                c = models / maya.MM3_COMPANIONS
                (c / "dspark").mkdir(parents=True)
                for n in maya.MM3_VISION_FILES:
                    (c / n).write_bytes(b"x" * 10)
                for n in dm["files"]:
                    (c / "dspark" / n).write_bytes(b"x" * 10)
                with patch.object(maya, "mm3_models", return_value=(small, small_d)), \
                        patch.object(maya, "offer_download", side_effect=AssertionError("nothing to download")):
                    for hip in (False, True):
                        self.assertEqual(maya.mm3_companions_step(args(), d, hip), c)
                exl3 = models / maya.MM3_DIR
                (exl3 / "dspark").mkdir(parents=True)
                for n in maya.MM3_VISION_FILES:
                    (exl3 / n).write_text("x")
                (exl3 / "dspark" / "config.json").write_text("{}")
                self.assertEqual(maya.mm3_companions_step(args(), d, False), exl3)
        finally:
            for p in quiet:
                p.stop()

    def test_unpublished_needs_the_files(self):
        """Before it is published, a GGUF quant is set up only from files already there (--model-dir), by their sizes."""
        m = maya.mm3_gguf_entry("gguf-xs")
        m["files"] = {n: 1000 + i for i, n in enumerate(m["files"])}   # (small stand-ins for the real sizes)
        m["revision"], m["sha256"] = "", {n: "" for n in m["files"]}   # (as before its upload)
        with tempfile.TemporaryDirectory() as d, patch.object(maya, "fail", fail), \
                patch.object(maya, "step", lambda *a: None), patch.object(maya, "warn", lambda *a: None), \
                patch.object(maya, "ok", lambda *a: None), patch.object(maya, "rotational", return_value=False), \
                patch.object(maya, "mm3_gguf_entry", return_value=m):
            d = Path(d)
            with self.assertRaisesRegex(Failed, "not published yet"):
                maya.mm3_gguf_model_step(args(), d, "gguf-xs")
            for n, size in m["files"].items():
                with (d / n).open("wb") as f:
                    f.truncate(size)
            self.assertEqual(len(maya.mm3_gguf_model_step(args(), d, "gguf-xs")), 3)


if __name__ == "__main__":
    unittest.main()
