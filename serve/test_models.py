"""serve/test_models.py - the web app's Models page: the run configs listed with their engine settings, the settings
written back into a config (checked first, its other keys kept), and a switch to another model - against the mock
engine, no GPU.

    python -m unittest serve.test_models -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as M  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
import serve.server as SV  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def glm_cfg(d: Path, name="maya-maya-m.json", **extra) -> Path:
    pack = d / "GLM-5.3-Flash" / "Maya-M" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack.parent / "shard.gguf").write_bytes(b"x" * 1000)
    cfg = {"exe": "/b/maya", "args": ["--glm-pack", str(pack), "--max-context", "131072", "--expert-cache", "auto",
                                      "--prefill", "auto", "--kv-streaming", "on"],
           "model_name": "glm-5.3-flash", "gpu": [0, 1], "port": 8080, "vision": {"mmproj": "/v/mmproj.gguf"},
           "env": {"MAYA_GLM_RAM_HEADROOM_GB": "4"}, "sampling": {"temperature": 1.0},
           "installer": {"models_dir": str(d), "quant": "Maya-M", "draft": "mtp", "written": "2026-10-20 09:00"},
           **extra}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


def ds4_cfg(d: Path, name="maya-gguf-q8.json", drafter=True, vision=False, quant="gguf-q8") -> Path:
    pack = d / "DeepSeek-V4.1-Flash" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "tensors.txt").write_text("t blk.0.x\n" + ("t mtp.0.w\n" if drafter else ""))
    (pack / "ds4.txt").write_text("hidden 4096\n" + ("vision.num_hidden_layers 24\n" if vision else ""))
    args = ["--pack", str(pack), "--max-context", "65536", "--tp", "2", "--cpu-threads", "16", "--vision", "off"]
    if drafter:
        args += ["--dspark", "on"]
    cfg = {"exe": "/b/maya-ds4", "args": args, "model_name": "deepseek-v4.1-flash", "gpu": [0, 1],
           "installer": {"family": "deepseek", "ds4_quant": quant, "written": "2026-10-09 10:00"}}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


def mm3_cfg(d: Path, name="maya-mm3.json", drafter=True, vision=True) -> Path:
    """MiniMax's config as maya.py's write_config_mm3 writes it."""
    pack = d / "MiniMax-M3.1-Flash-EXL3-3.5bpw" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "mm3.txt").write_text("hidden_size 3072\nmax_position_embeddings 1048576\n" +
                                  ("dspark.num_hidden_layers 2\n" if drafter else "") +
                                  ("vision.hidden_size 1280\n" if vision else ""))
    args = ["--pack", str(pack), "--max-context", "65536", "--kv-streaming", "on", "--tp", "4",
            "--dspark", "off", "--vision", "on"]
    cfg = {"exe": "/b/maya-mm3", "args": args, "model_name": "minimax-m3.1-flash", "gpu": list(range(9)),
           "reasoning_effort": "auto", "installer": {"family": "minimax", "quant": "MiniMax-M3.1-Flash",
                                                     "written": "2026-10-27 10:00"}}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


class Settings(unittest.TestCase):
    """A config's engine arguments as the page's settings, and back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_glm_read(self):
        s = M.settings_of(M.read(glm_cfg(self.d)))
        self.assertEqual(s, {"context": 131072, "gpus": [0, 1], "env": {"MAYA_GLM_RAM_HEADROOM_GB": "4"},
                             "images": True, "reserve_mib": None, "parallel": 1, "draft": "mtp", "expert_cache": None,
                             "prefill": None, "kv_streaming": True, "kv_resident": None})

    def test_glm_written_back_with_the_rest_kept(self):
        cfg = M.read(glm_cfg(self.d))
        new = M.apply(cfg, {"context": 65536, "parallel": 2, "reserve_mib": 1024, "expert_cache": 1200,
                            "prefill": 4096, "kv_resident": 49152, "gpus": [1], "env": {"A": "1"}, "images": False})
        a = new["args"]
        for f, v in (("--max-context", "65536"), ("--parallel", "2"), ("--vram-reserve-mib", "1024"),
                     ("--expert-cache", "1200"), ("--prefill", "4096"), ("--kv-resident", "49152")):
            self.assertEqual(M.flag(a, f), v, f)
        self.assertEqual((new["gpu"], new["env"], new["sampling"], new["port"]), ([1], {"A": "1"}, {"temperature": 1.0}, 8080))
        self.assertNotIn("vision", new)                       # (Images off: the encoder's entry kept aside ...)
        back = M.apply(new, {"images": True, "reserve_mib": None, "parallel": 1, "env": {}})
        self.assertEqual(back["vision"], {"mmproj": "/v/mmproj.gguf"})   # (... and back when it is switched on)
        self.assertIsNone(M.flag(back["args"], "--vram-reserve-mib"))
        self.assertNotIn("env", back)
        self.assertEqual(M.settings_of(back)["parallel"], 1)

    def test_glm_dflash2_only_when_its_drafter_is_here(self):
        cfg = M.read(glm_cfg(self.d))
        self.assertEqual(M.choices(cfg)["draft"], ["mtp"])
        with self.assertRaisesRegex(ValueError, "DFlash2"):
            M.apply(cfg, {"draft": "dflash2"})
        f = self.d / "GLM-5.3-Flash-DFlash2" / "drafter.gguf"
        f.parent.mkdir()
        f.write_bytes(b"x")
        new = M.apply(cfg, {"draft": "dflash2"})
        self.assertEqual(M.flag(new["args"], "--dflash"), str(f))
        self.assertEqual(M.settings_of(M.apply(new, {"draft": "mtp"}))["draft"], "mtp")

    def test_checked_before_anything_changes(self):
        cfg = M.read(glm_cfg(self.d))
        for bad in ({"context": "lots"}, {"parallel": 9}, {"gpus": []}, {"gpus": ["0"]}, {"reserve_mib": -1},
                    {"env": {"not a name": "1"}}, {"tp": 2}, {"prefill": 1.5}):
            with self.assertRaises(ValueError, msg=bad):
                M.apply(cfg, bad)

    def test_an_older_glm_config_keeps_its_command(self):
        """A config from before --kv-streaming (no flag: every cell in VRAM) reads as streaming off, and a save writes
        only what changed - its engine, built with it, may not know the newer flags (a Maya-S v2 setup of 2026-10-10)."""
        cfg = M.read(glm_cfg(self.d))
        cfg["args"] = ["--glm-pack", "/m/pack", "--max-context", "32768"]
        self.assertFalse(M.settings_of(cfg)["kv_streaming"])
        self.assertTrue(M.settings_of(dict(cfg, env={"STRATA_GLM_KV_RESIDENT": "65536"}))["kv_streaming"])
        new = M.apply(cfg, {"context": 65536})
        self.assertEqual(new["args"], ["--glm-pack", "/m/pack", "--max-context", "65536"])
        on = M.apply(cfg, {"kv_streaming": True})
        self.assertEqual(M.flag(on["args"], "--kv-streaming"), "on")
        self.assertNotIn("--expert-cache", on["args"])

    def test_loader_variables_not_from_the_page(self):
        """LD_PRELOAD, PATH and the like change what the engine's process loads: refused, a config's own kept."""
        cfg = M.read(glm_cfg(self.d))
        for k in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "PATH", "Path", "PYTHONPATH", "BASH_ENV"):
            with self.assertRaises(ValueError, msg=k):
                M.apply(cfg, {"env": {k: "/tmp/x"}})
        own = dict(cfg, env={"LD_LIBRARY_PATH": "/opt/rocm/lib"})
        kept = M.apply(own, {"env": {"LD_LIBRARY_PATH": "/opt/rocm/lib", "MAYA_X": "1"}})
        self.assertEqual(kept["env"], {"LD_LIBRARY_PATH": "/opt/rocm/lib", "MAYA_X": "1"})
        with self.assertRaises(ValueError):
            M.apply(own, {"env": {"LD_LIBRARY_PATH": "/tmp/evil"}})

    def test_deepseek_read_and_written(self):
        p = ds4_cfg(self.d)
        cfg = M.read(p)
        s = M.settings_of(cfg)
        self.assertEqual((s["draft"], s["tp"], s["cpu_threads"], s["images"], s["mmap"]), ("dspark", 2, 16, False, "auto"))
        new = M.apply(cfg, {"draft": "none", "tp": 1, "mmap": "off", "prefill_chunk": 2048, "cpu_threads": None})
        a = new["args"]
        self.assertEqual((M.flag(a, "--dspark"), M.flag(a, "--tp"), M.flag(a, "--mmap-experts"), M.flag(a, "--prefill-chunk")),
                         ("none", "1", "off", "2048"))
        self.assertIsNone(M.flag(a, "--cpu-threads"))
        with self.assertRaisesRegex(ValueError, "vision tower"):
            M.apply(cfg, {"images": True})
        self.assertEqual(M.choices(M.read(ds4_cfg(self.d, "maya-x.json", drafter=False)))["draft"], ["none"])

    def test_deepseek_prompt_lookup_and_draft_confidence(self):
        cfg = M.read(ds4_cfg(self.d))
        s = M.settings_of(cfg)
        self.assertEqual((s["prompt_lookup"], s["dspark_conf"]), (True, None))   # (the engine's: --suffix-draft 3, 0.5)
        off = M.apply(cfg, {"prompt_lookup": False, "dspark_conf": "0.6"})
        self.assertEqual((M.flag(off["args"], "--suffix-draft"), M.flag(off["args"], "--dspark-conf")), ("0", "0.6"))
        self.assertEqual((M.settings_of(off)["prompt_lookup"], M.settings_of(off)["dspark_conf"]), (False, 0.6))
        on = M.apply(off, {"prompt_lookup": True, "dspark_conf": None})["args"]
        self.assertFalse("--suffix-draft" in on or "--dspark-conf" in on)        # (back to the engine's own)
        own = dict(cfg, args=cfg["args"] + ["--suffix-draft", "5"])              # (a config's own match length: kept)
        self.assertEqual(M.flag(M.apply(own, {"prompt_lookup": True})["args"], "--suffix-draft"), "5")
        for bad, why in ((1.5, "from 0 to 1"), ("x", "a number"), (True, "a number")):
            with self.assertRaisesRegex(ValueError, why):
                M.apply(cfg, {"dspark_conf": bad})
        with self.assertRaisesRegex(ValueError, "Prompt lookup"):
            M.apply(cfg, {"prompt_lookup": "on"})

    def test_minimax_memory_prompt_chunk_and_draft_confidence(self):
        cfg = M.read(mm3_cfg(self.d))
        new = M.apply(cfg, {"reserve_gb": "0.5", "cache_gb": 2, "prefill_chunk": 2048, "dspark_conf": 0.65})
        a = new["args"]
        for f, v in (("--reserve-gb", "0.5"), ("--cache-gb", "2"), ("--chunk", "2048"), ("--dspark-conf", "0.65")):
            self.assertEqual(M.flag(a, f), v, f)
        self.assertNotIn("--prefill-chunk", a)                                    # (maya-mm3's own name: --chunk)
        s = M.settings_of(new)
        self.assertEqual((s["reserve_gb"], s["cache_gb"], s["prefill_chunk"], s["dspark_conf"]), (0.5, 2.0, 2048, 0.65))
        back = M.apply(new, {"reserve_gb": "", "cache_gb": None, "prefill_chunk": None, "dspark_conf": None})["args"]
        for f in ("--reserve-gb", "--cache-gb", "--chunk", "--dspark-conf"):
            self.assertNotIn(f, back)
        with self.assertRaisesRegex(ValueError, "VRAM kept free"):
            M.apply(cfg, {"reserve_gb": 100})
        with self.assertRaisesRegex(ValueError, "Prompt chunk"):
            M.apply(cfg, {"prefill_chunk": 8})

    def test_deepseek_answers_at_once(self):
        """maya-ds4 --parallel: read (1 where the config names none), written where it changes, 1-8."""
        cfg = M.read(ds4_cfg(self.d))
        self.assertEqual(M.settings_of(cfg)["parallel"], 1)
        self.assertNotIn("--parallel", M.apply(cfg, {"cpu_threads": 8})["args"])   # (an older config: as it was)
        new = M.apply(cfg, {"parallel": 4})
        self.assertEqual(M.flag(new["args"], "--parallel"), "4")
        self.assertEqual(M.settings_of(new)["parallel"], 4)
        self.assertEqual(M.flag(M.apply(new, {"parallel": 1})["args"], "--parallel"), "1")
        for bad in (0, 9):
            with self.assertRaisesRegex(ValueError, "Answers at once"):
                M.apply(cfg, {"parallel": bad})

    def test_deepseek_tensor_parallel_beyond_two(self):
        # a 9-GPU config the calibration left at --tp 8: read as it is, and its other settings still save
        cfg = M.read(ds4_cfg(self.d))
        cfg["gpu"] = list(range(9))
        cfg["args"] = M.set_flag(cfg["args"], "--tp", 8)
        self.assertEqual(M.settings_of(cfg)["tp"], 8)
        new = M.apply(cfg, {"prefill_chunk": 2048})
        self.assertEqual((M.flag(new["args"], "--tp"), M.flag(new["args"], "--prefill-chunk")), ("8", "2048"))
        self.assertEqual(M.flag(M.apply(cfg, {"tp": 9})["args"], "--tp"), "9")
        with self.assertRaisesRegex(ValueError, "Tensor parallel: from 1 to 9"):
            M.apply(cfg, {"tp": 10})
        with self.assertRaisesRegex(ValueError, "Tensor parallel: from 1 to 2"):
            M.apply(cfg, {"gpus": [0, 1]})               # (fewer GPUs than its group: tp too)
        self.assertEqual(M.flag(M.apply(cfg, {"gpus": [0], "tp": 2})["args"], "--tp"), "2")   # (one GPU: 2 as before)
        # no --tp: the engine's own 2, which a save keeps
        cfg["args"] = M.drop_flag(cfg["args"], "--tp")
        self.assertEqual(M.settings_of(cfg)["tp"], 2)
        self.assertEqual(M.flag(M.apply(cfg, {"cpu_threads": 8})["args"], "--tp"), "2")

    def test_minimax_read_and_written_with_its_own_flags(self):
        cfg = M.read(mm3_cfg(self.d))
        self.assertEqual(M.family(cfg), "minimax")
        self.assertEqual(M.family({"exe": "/b/maya-mm3", "args": ["--pack", "/p"]}), "minimax")   # (no installer)
        self.assertEqual(M.settings_of(cfg), {"context": 65536, "gpus": list(range(9)), "env": {}, "images": True,
                                              "parallel": 1, "draft": "none", "tp": 4, "kv_cache": "int8",
                                              "kv_streaming": True, "kv_resident": None, "cpu_threads": None,
                                              "reserve_gb": None, "prefill_chunk": None, "cache_gb": None,
                                              "dspark_conf": None})
        self.assertEqual(M.choices(cfg), {"draft": ["dspark", "none"], "images": True})
        new = M.apply(cfg, {"context": 262144, "draft": "dspark", "images": False, "tp": 2, "kv_cache": "fp8",
                            "kv_resident": 65536, "cpu_threads": 24})
        a = new["args"]
        for f, v in (("--max-context", "262144"), ("--dspark", "on"), ("--vision", "off"), ("--tp", "2"),
                     ("--kv-cache", "fp8"), ("--kv-streaming", "on"), ("--kv-resident", "65536"),
                     ("--cpu-threads", "24")):
            self.assertEqual(M.flag(a, f), v, f)
        # maya-mm3 stops at a flag it does not know: none of DeepSeek's or GLM's are written
        for f in ("--mmap-experts", "--prefill-chunk", "--reserve-mib", "--vram-reserve-mib", "--prefill"):
            self.assertNotIn(f, a)
        self.assertNotIn("--parallel", a)                                       # (1 and not in the config: as it was)
        with self.assertRaisesRegex(ValueError, "not a setting of this model: mmap"):
            M.apply(cfg, {"mmap": "on"})
        # tp 0: the layers split (no --tp); 3 is not a group the engine takes
        self.assertNotIn("--tp", M.apply(cfg, {"tp": 0})["args"])
        with self.assertRaisesRegex(ValueError, "1, 2 or 4"):
            M.apply(cfg, {"tp": 3})
        with self.assertRaisesRegex(ValueError, "KV cache"):
            M.apply(cfg, {"kv_cache": "q4_0"})
        # KV streaming off: the window goes with it; INT8 (the engine's default) not written where it was not
        off = M.apply(cfg, {"kv_streaming": False, "kv_resident": 65536})["args"]
        self.assertEqual((M.flag(off, "--kv-streaming"), "--kv-resident" in off, "--kv-cache" in off), ("off", False, False))
        # a pack without the drafter or the tower: neither offered
        bare = M.read(mm3_cfg(self.d, "maya-mm3-b.json", drafter=False, vision=False))
        self.assertEqual(M.choices(bare), {"draft": ["none"], "images": False})
        with self.assertRaisesRegex(ValueError, "no DSpark drafter"):
            M.apply(bare, {"draft": "dspark"})
        self.assertEqual(M.quant_label(cfg, Path("maya-mm3.json")), "EXL3 3.5 bpw")

    def test_minimax_answers_at_once(self):
        """maya-mm3 --parallel: read (1 where the config names none), written where it changes, 1-8; the KV streaming and
        DSpark choices stay as they are for the day it is 1 again (the engine leaves both out while it is above 1)."""
        cfg = M.read(mm3_cfg(self.d))
        self.assertEqual(M.settings_of(cfg)["parallel"], 1)
        self.assertNotIn("--parallel", M.apply(cfg, {"cpu_threads": 8})["args"])   # (an older config: as it was)
        new = M.apply(cfg, {"parallel": 4, "draft": "dspark"})
        self.assertEqual(M.flag(new["args"], "--parallel"), "4")
        self.assertEqual((M.flag(new["args"], "--kv-streaming"), M.flag(new["args"], "--dspark")), ("on", "on"))
        s = M.settings_of(new)
        self.assertEqual((s["parallel"], s["kv_streaming"], s["draft"]), (4, True, "dspark"))
        self.assertEqual(M.flag(M.apply(new, {"parallel": 1})["args"], "--parallel"), "1")
        for bad in (0, 9):
            with self.assertRaisesRegex(ValueError, "Answers at once"):
                M.apply(cfg, {"parallel": bad})

    def test_listing(self):
        g, ds = glm_cfg(self.d), ds4_cfg(self.d)
        (self.d / "maya-maya-m.shared-settings.json").write_text("{}")
        os.utime(g, (time.time() - 100, time.time() - 100))
        with mock.patch.object(M, "machine_gpus", return_value=[{"index": 0, "name": "RTX 3090", "vram_gb": 24.0}]):
            out = M.listing(self.d, str(g))
        self.assertEqual([m["file"] for m in out["models"]], [ds.name, g.name])   # (the most recently used first)
        mm = out["models"][1]
        self.assertEqual((mm["running"], mm["quant"], mm["family"], mm["size"]), (True, "Maya-M", "glm", 1000))
        self.assertEqual(out["models"][0]["quant"], "UD-Q8_K_XL")
        # each model's context choices are its setup's: DeepSeek's reach 1M, GLM's 128K (its own size is offered too)
        self.assertEqual((out["models"][0]["contexts"], mm["contexts"]), (M.DS4_CONTEXTS, M.CONTEXTS))
        self.assertEqual([x["index"] for x in out["gpus"]], [0, 1])               # (GPU 1: named by a config only)


    def test_context_range_per_model(self):
        # the Engine settings' Context size slider: Settings > Model's range for each model - up to what the model was
        # trained for (MiniMax's pack: 1M), else 256K (a GLM GGUF header it cannot read, a DeepSeek pack that does not
        # say), never below the config's own size; untested past 256K except on DeepSeek
        g, ds, mm = glm_cfg(self.d), ds4_cfg(self.d), mm3_cfg(self.d)
        M._trained.clear()
        with mock.patch.object(M, "machine_gpus", return_value=[]):
            got = {m["family"]: m["context_range"] for m in M.listing(self.d, None)["models"]}
        self.assertEqual(got["minimax"], {"min": 4096, "max": 1048576, "trained": 1048576, "tested": 262144})
        self.assertEqual(got["glm"], {"min": 4096, "max": 262144, "trained": None, "tested": 262144})
        self.assertEqual(got["deepseek"], {"min": 4096, "max": 262144, "trained": None, "tested": 1048576})
        cfg = json.loads(ds.read_text())
        cfg["args"][cfg["args"].index("--max-context") + 1] = "524288"
        ds.write_text(json.dumps(cfg))
        self.assertEqual(M.describe(ds, None)["context_range"]["max"], 524288)
        # GLM: the GGUF header beside its pack (the server's own reading), kept once read
        with mock.patch.object(M, "trained_context", return_value=1048576) as tc:
            for _ in range(2):
                self.assertEqual(M.describe(g, None)["context_range"]["max"], 1048576)
        tc.assert_called_once_with(str((self.d / "GLM-5.3-Flash" / "Maya-M").resolve()))
        M._trained.clear()

    def test_installs_the_setup_no_longer_lists_are_still_listed(self):
        """v1.1.34: unsloth's UD-Q4_K_XL (gguf-q4) and GLM's EXL3-sc_4.10bpw are out of the setup's pickers; a config of
        either is listed, shown and settable on the Models page as before."""
        g = glm_cfg(self.d, name="maya-exl3-sc_4.10bpw.json",
                    installer={"models_dir": str(self.d), "quant": "EXL3-sc_4.10bpw", "written": "2026-10-20 09:00"})
        ds = ds4_cfg(self.d, name="maya-gguf-q4.json", quant="gguf-q4")
        with mock.patch.object(M, "machine_gpus", return_value=[]):
            out = M.listing(self.d, None)
        by = {m["file"]: m for m in out["models"]}
        self.assertEqual(by[ds.name]["quant"], "UD-Q4_K_XL")
        self.assertEqual(by[g.name]["quant"], "EXL3-sc_4.10bpw")
        self.assertTrue(by[ds.name]["ok"] and by[g.name]["ok"])
        cfg = M.read(ds)
        self.assertEqual(M.settings_of(cfg)["context"], 65536)
        self.assertEqual(M.apply(cfg, {"context": 131072})["args"][cfg["args"].index("--max-context") + 1], "131072")
        self.assertEqual(M.maya_quants.config_label(cfg), "DeepSeek V4.1 · UD-Q4_K_XL")

    def test_the_qwen_models_label(self):
        """The CYBER-FROST config's quant shows as Q5_K_M beside the model's name the page writes from model_name; the
        label the server reports is "Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M"."""
        cfg = {"exe": "/b/maya", "args": ["--pack", "/p", "--ple-gguf", "/g.gguf"], "model_name": "cyber-frost-3.8-ps",
               "installer": {"family": "qwen", "quant": "CYBER-FROST-3.8-PS", "qwen_quant": "q5_k_m"}}
        self.assertEqual(M.quant_label(cfg, Path("maya-cyber-frost.json")), "Q5_K_M")
        self.assertEqual(M.maya_quants.config_label(cfg), "Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M")


class Served(unittest.TestCase):
    """The server on the mock engine, its config beside another model's."""

    def setUp(self):
        os.environ.pop("MAYA_RESTART_ON_UPDATE", None)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cfg, self.other = glm_cfg(self.d), ds4_cfg(self.d)
        self.svc.config_path = str(self.cfg)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.gpus = mock.patch.object(M, "machine_gpus", return_value=[])
        self.gpus.start()

    def tearDown(self):
        self.gpus.stop()
        os.environ.pop("MAYA_RESTART_ON_UPDATE", None)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


class Page(Served):
    """GET /api/models, POST /api/models/settings and /api/models/switch."""

    def test_listed(self):
        s, b = self.req("GET", "/api/models")
        self.assertEqual((s, b["running"], b["restarts"], len(b["models"])), (200, self.cfg.name, False, 2))
        self.assertTrue(next(m for m in b["models"] if m["file"] == self.cfg.name)["running"])

    def test_secrets_in_the_environment_stay_on_this_pc(self):
        """A config's HF_TOKEN / *_API_KEY reach the page masked (as the report leaves them out); a save that sends the
        mask back keeps the value, a new value replaces it."""
        cfg = json.loads(self.cfg.read_text())
        cfg["env"] = {"HF_TOKEN": "hf_secret", "MAYA_GLM_RAM_HEADROOM_GB": "4"}
        self.cfg.write_text(json.dumps(cfg))
        s, b = self.req("GET", "/api/models")
        self.assertNotIn("hf_secret", json.dumps(b))
        mine = next(m for m in b["models"] if m["file"] == self.cfg.name)
        self.assertEqual((mine["settings"]["env"]["HF_TOKEN"], mine["command"]["env"]["HF_TOKEN"]), (M.MASK, M.MASK))
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "dry_run": True,
                                                         "settings": {"env": mine["settings"]["env"]}})
        self.assertNotIn("hf_secret", json.dumps(b))
        s, _ = self.req("POST", "/api/models/settings", {"file": self.cfg.name,
                                                         "settings": {"context": 65536, "env": mine["settings"]["env"]}})
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["HF_TOKEN"], "hf_secret")
        self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"env": {"HF_TOKEN": "hf_new"}}})
        self.assertEqual(json.loads(self.cfg.read_text())["env"], {"HF_TOKEN": "hf_new"})

    def test_dry_run_saves_nothing(self):
        before = self.cfg.read_text()
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": 65536},
                                                         "dry_run": True})
        self.assertEqual((s, M.flag(b["command"]["args"], "--max-context")), (200, "65536"))
        self.assertEqual(self.cfg.read_text(), before)

    def test_another_models_settings_leave_which_one_starts_next(self):
        old = time.time() - 1000
        os.utime(self.other, (old, old))
        s, b = self.req("POST", "/api/models/settings", {"file": self.other.name, "settings": {"tp": 1}})
        self.assertEqual((s, b["changed"], b["running"], b["restarts"]), (202, True, False, False))
        self.assertEqual(M.flag(json.loads(self.other.read_text())["args"], "--tp"), "1")
        self.assertAlmostEqual(self.other.stat().st_mtime, old, delta=1)   # (still the one used longest ago)

    def test_saved_without_a_restart_here(self):
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": 65536}})
        self.assertEqual((s, b["changed"], b["running"], b["restarts"]), (202, True, True, False))
        cfg = json.loads(self.cfg.read_text())
        self.assertEqual(M.flag(cfg["args"], "--max-context"), "65536")
        self.assertEqual(cfg["installer"]["as_set_up"]["context"], 131072)      # (the page's Reset)
        self.assertEqual(M.flag(json.loads((self.d / (self.cfg.name + ".prev")).read_text())["args"], "--max-context"),
                         "131072")
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": "x"}})
        self.assertEqual(s, 400)

    def test_running_one_starts_again_with_its_new_settings(self):
        os.environ["MAYA_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(SV.os, "_exit") as ex, mock.patch.object(self.svc, "close_for_restart"):
            s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"parallel": 2}})
            self.assertEqual((s, b["restarts"]), (202, True))
            for _ in range(50):
                if ex.called:
                    break
                time.sleep(0.05)
        ex.assert_called_once_with(SV.RESTART_EXIT)
        note = json.loads((self.d / M.SWITCH_FILE).read_text())
        self.assertEqual((note["why"], Path(note["config"]).name, note["port"]),
                         ("settings", self.cfg.name, self.httpd.server_address[1]))

    def test_switch(self):
        old = time.time() - 1000
        os.utime(self.other, (old, old))
        s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
        self.assertEqual((s, b["to"], b["restarts"]), (202, self.other.name, False))
        self.assertGreater(self.other.stat().st_mtime, old + 900)            # (the next start picks it)
        self.assertFalse((self.d / M.SWITCH_FILE).exists())                   # (no restart: no note for maya.py)
        os.environ["MAYA_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(SV.os, "_exit") as ex, mock.patch.object(self.svc, "close_for_restart"):
            s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
            for _ in range(50):
                if ex.called:
                    break
                time.sleep(0.05)
        self.assertEqual((s, b["restarts"]), (202, True))
        ex.assert_called_once_with(SV.RESTART_EXIT)
        self.assertEqual(json.loads((self.d / M.SWITCH_FILE).read_text())["why"], "switch")
        s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
        self.assertEqual(s, 400)                                              # (already starting again)

    def test_switch_writes_the_models_own_defaults(self):
        """A switch to a config without its model's sampling and level (an older setup's), or with another model's,
        writes the model's own into it (the setup's MODEL_DEFAULTS), the config before kept beside it."""
        cfg = json.loads(self.other.read_text())                             # (DeepSeek's)
        cfg.pop("reasoning_effort", None)
        cfg["sampling"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}      # (not DeepSeek's)
        self.other.write_text(json.dumps(cfg))
        s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
        self.assertEqual(s, 202, b)
        got = json.loads(self.other.read_text())
        self.assertEqual((got["sampling"], got["reasoning_effort"]), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        self.assertTrue((self.d / (self.other.name + ".prev")).exists())
        self.assertFalse(M.own_defaults(self.other))                         # (its own already: nothing written)

    def test_refused(self):
        for f in (self.cfg.name, "../maya-x.json", "maya-none.json", "maya-maya-m.shared-settings.json", None):
            s, _ = self.req("POST", "/api/models/switch", {"file": f})
            self.assertEqual(s, 400, f)
        s, _ = self.req("POST", "/api/models/switch", {"file": self.other.name}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)

    def test_files_gone(self):
        # a config whose model was deleted: listed as such, and no switch to it (maya.py would stop there, no model
        # running)
        gone = glm_cfg(self.d, "maya-gone.json")
        cfg = json.loads(gone.read_text())
        cfg["args"] = M.set_flag(cfg["args"], "--glm-pack", str(self.d / "deleted" / "pack"))
        gone.write_text(json.dumps(cfg))
        s, b = self.req("GET", "/api/models")
        m = next(x for x in b["models"] if x["file"] == gone.name)
        self.assertEqual((m["ok"], m["missing"]), (False, f"its model files are not on this PC ({self.d / 'deleted'})"))
        self.assertTrue(next(x for x in b["models"] if x["file"] == self.other.name)["ok"])
        s, b = self.req("POST", "/api/models/switch", {"file": gone.name})
        self.assertEqual(s, 400)
        self.assertIn("its model files are not on this PC", json.dumps(b))
        s, _ = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": 8192}},
                        {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        self.assertEqual(M.flag(json.loads(self.cfg.read_text())["args"], "--max-context"), "131072")



def glm_quant(d: Path, quant: str, folder: Path | None = None, pack_name="pack", shards=3, dflash=False,
              models: Path | None = None) -> Path:
    """A GLM config as the setup writes it for `quant`: its GGUF shards and pack in <folder> (<models>/<quant>), the
    image encoder in vision/, the DFlash2 drafter in <models>/GLM-5.3-Flash-DFlash2 when it drafts with it."""
    models = models or d
    folder = folder or models / "GLM-5.3-Flash" / quant
    pack = folder / pack_name
    pack.mkdir(parents=True, exist_ok=True)
    names = [f"GLM-5.3-Flash-{quant}-{i:05d}-of-{shards:05d}.gguf" for i in range(1, shards + 1)]
    for n in names:
        (folder / n).write_bytes(b"g" * 1000)
    (pack / "index.txt").write_bytes(b"i" * 100)
    (pack / "sources.json").write_text(json.dumps({"files": [{"name": n, "size": 1000} for n in names]}))
    (folder / "vision").mkdir(exist_ok=True)
    (folder / "vision" / "mmproj.gguf").write_bytes(b"v" * 200)
    args = ["--glm-pack", str(pack), "--max-context", "65536"]
    if dflash:
        drafter = models / "GLM-5.3-Flash-DFlash2" / "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf"
        drafter.parent.mkdir(parents=True, exist_ok=True)
        drafter.write_bytes(b"f" * 700)
        args += ["--dflash", str(drafter)]
    cfg = {"exe": "/b/maya", "args": args, "model_name": "glm-5.3-flash", "gpu": [0],
           "vision": {"mmproj": str(folder / "vision" / "mmproj.gguf")},
           "installer": {"models_dir": str(models), "quant": quant, "draft": "dflash2" if dflash else "mtp"}}
    p = d / f"maya-{quant.lower()}.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


class Delete(unittest.TestCase):
    """A model off the list, and with its files: what its setup downloaded for it and the pack Maya built - never what
    another model runs from, nor a GGUF of your own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_its_own_folder_and_drafter(self):
        m = glm_quant(self.d, "Maya-M")
        l = glm_quant(self.d, "Maya-L", dflash=True)
        (self.d / f"run-{l.stem}.sh").write_text("#!/bin/sh\n")
        plan = M.delete_plan(self.d, l)
        folder = self.d / "GLM-5.3-Flash" / "Maya-L"
        self.assertTrue(plan["ok"])
        self.assertEqual([(Path(i["path"]).name, i["what"]) for i in plan["items"]],
                         [("Maya-L", "folder"), ("GLM-5.3-Flash-DFlash2-Q4_K_M.gguf", "drafter")])
        self.assertEqual((plan["gguf"], plan["pack"], plan["vision"], plan["drafter"], plan["drafter_users"]),
                         (3, True, True, 700, []))
        self.assertEqual(plan["bytes"], 3 * 1000 + 100 + len((folder / "pack" / "sources.json").read_bytes()) + 200 + 700)
        out = M.delete(self.d, l, files=True)
        self.assertEqual((out["deleted"], out["freed"]), (l.name, plan["bytes"]))
        self.assertFalse(folder.exists() or l.exists() or (self.d / f"run-{l.stem}.sh").exists())
        self.assertFalse((self.d / "GLM-5.3-Flash-DFlash2").exists())       # (its folder, empty now)
        self.assertTrue(m.exists() and (self.d / "GLM-5.3-Flash" / "Maya-M" / "pack").is_dir())

    def test_off_the_list_only(self):
        l = glm_quant(self.d, "Maya-L")
        prev, shared = l.with_name(l.name + ".prev"), l.with_name(l.stem + ".shared-settings.json")
        prev.write_text("{}")
        shared.write_text("{}")
        out = M.delete(self.d, l)
        self.assertEqual((out["freed"], out["files"]), (0, False))
        self.assertFalse(l.exists() or prev.exists() or shared.exists())
        self.assertTrue((self.d / "GLM-5.3-Flash" / "Maya-L" / "pack").is_dir())

    def test_a_folder_another_model_runs_from(self):
        # two quants' files in one folder (the setup's pack-<model>/ beside pack/): only its own pack and GGUF files
        folder = self.d / "GLM-5.3-Flash"
        glm_quant(self.d, "Maya-M", folder=folder)
        l = glm_quant(self.d, "Maya-L", folder=folder, pack_name="pack-maya-l")
        plan = M.delete_plan(self.d, l)
        self.assertEqual(sorted(i["what"] for i in plan["items"]), ["gguf", "gguf", "gguf", "pack"])
        self.assertFalse(plan["vision"])                    # (vision/: Maya-M's too)
        M.delete(self.d, l, files=True)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         [f"GLM-5.3-Flash-Maya-M-{i:05d}-of-00003.gguf" for i in (1, 2, 3)] + ["pack", "vision"])

    def test_your_own_gguf_stays(self):
        own = self.d / "elsewhere"
        l = glm_quant(self.d, "Maya-L", folder=own, models=self.d / "models")
        plan = M.delete_plan(self.d, l)
        self.assertEqual([i["what"] for i in plan["items"]], ["pack"])
        self.assertIn("Your own model files", plan["kept"][0])
        M.delete(self.d, l, files=True)
        self.assertEqual(len(list(own.glob("*.gguf"))), 3)
        self.assertFalse((own / "pack").exists())

    def test_a_drafter_another_model_drafts_with(self):
        m = glm_quant(self.d, "Maya-M", dflash=True)
        l = glm_quant(self.d, "Maya-L", dflash=True)
        os.utime(m, (1_700_000_000, 1_700_000_000))
        plan = M.delete_plan(self.d, l)
        self.assertEqual(plan["drafter_users"], [m.name])
        out = M.delete(self.d, l, files=True)
        self.assertEqual(out["drafter_users"], [m.name])
        cfg = json.loads(m.read_text())
        self.assertNotIn("--dflash", cfg["args"])           # (it drafts with MTP now: its drafter is gone)
        self.assertEqual(M.settings_of(cfg)["draft"], "mtp")
        self.assertEqual(m.stat().st_mtime, 1_700_000_000)  # (which model starts next: as it was)
        self.assertIn("--dflash", json.loads(m.with_name(m.name + ".prev").read_text())["args"])

    def test_without_the_setups_record(self):
        p = ds4_cfg(self.d)                                 # (no models_dir: which files are its own is not known)
        plan = M.delete_plan(self.d, p)
        self.assertEqual((plan["ok"], plan["items"]), (False, []))
        with self.assertRaises(ValueError):
            M.delete(self.d, p, files=True)
        self.assertTrue(p.exists())


class Leftovers(unittest.TestCase):
    """The files in the models folder no model here runs from: a whole folder, or one quant's beside another's."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.models = self.d / "models"
        self.big = mock.patch.object(M, "LEFTOVER_MIN", 500)       # (the fixtures' files are small)
        self.big.start()

    def tearDown(self):
        self.big.stop()
        self.tmp.cleanup()

    def test_what_no_model_uses(self):
        glm_quant(self.d, "Maya-M", models=self.models)
        l = glm_quant(self.d, "Maya-L", models=self.models)
        self.assertEqual(M.leftovers(self.d)["items"], [])     # (both GLM quants are on the list)
        old = self.models / "DSV4"                            # (a model whose config was deleted by hand)
        (old / "pack").mkdir(parents=True)
        (old / "pack" / "tensors.txt").write_text("t blk.0.x\n")
        (old / "model-00001-of-00002.safetensors").write_bytes(b"s" * 2000)
        (old / ".cache" / "huggingface").mkdir(parents=True)
        (self.models / "download.log").write_text("small\n")
        l.unlink()
        out = M.leftovers(self.d)
        self.assertEqual(out["folders"], [str(self.models.resolve())])
        got = {Path(i["path"]).name: (i["folder"], i["pack"], i["hub"]) for i in out["items"]}
        self.assertEqual(got, {"DSV4": (True, True, True), "Maya-L": (True, True, False)})
        self.assertEqual(out["bytes"], sum(i["bytes"] for i in out["items"]))
        gone = M.delete_leftover(self.d, out["items"][0]["path"])
        self.assertFalse(Path(gone["deleted"]).exists())
        self.assertTrue((self.models / "GLM-5.3-Flash" / "Maya-M" / "pack").is_dir())

    def test_never_a_folder_a_model_reads_through_its_pack(self):
        # DeepSeek's drafter and vision tower beside its GGUF folder: no config names them, the pack reads them
        p = ds4_cfg(self.d)
        cfg = json.loads(p.read_text())
        root = self.models / "DeepSeek-V4.1-Flash"
        self.models.mkdir(parents=True, exist_ok=True)   # (root itself: the rename makes it; Windows will not rename onto a folder)
        (self.d / "DeepSeek-V4.1-Flash").rename(root)
        cfg["args"][1] = str(root / "pack")
        cfg["installer"]["models_dir"] = str(self.models)
        p.write_text(json.dumps(cfg))
        for side in ("dspark", "vision"):
            (root / side).mkdir()
            (root / side / f"{side}.bin").write_bytes(b"d" * 2000)
        self.assertEqual(M.leftovers(self.d)["items"], [])

    def test_only_what_is_listed_is_deleted(self):
        m = glm_quant(self.d, "Maya-M", models=self.models)
        with self.assertRaisesRegex(ValueError, "not a file no model uses"):
            M.delete_leftover(self.d, str(self.models / "GLM-5.3-Flash" / "Maya-M"))
        with self.assertRaisesRegex(ValueError, "not a file no model uses"):
            M.delete_leftover(self.d, str(self.d))
        self.assertTrue(m.exists() and (self.models / "GLM-5.3-Flash" / "Maya-M").is_dir())


class DeletePage(Served):
    """POST /api/models/delete: not the model running, the app's own page only, dry_run deletes nothing."""

    def test_delete(self):
        l = glm_quant(self.d, "Maya-L", dflash=True)
        s, b = self.req("POST", "/api/models/delete", {"file": l.name, "files": True, "dry_run": True})
        self.assertEqual((s, b["file"], b["ok"], b["gguf"]), (200, l.name, True, 3))
        self.assertTrue(l.exists())
        s, _ = self.req("POST", "/api/models/delete", {"file": l.name, "files": True},
                        {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        s, b = self.req("POST", "/api/models/delete", {"file": l.name, "files": True})
        self.assertEqual((s, b["deleted"], b["freed"] > 3000), (200, l.name, True))
        s, b = self.req("GET", "/api/models")
        self.assertNotIn(l.name, [m["file"] for m in b["models"]])
        self.assertFalse((self.d / "GLM-5.3-Flash" / "Maya-L").exists())

    def test_not_the_one_running(self):
        for f in (self.cfg.name, "../maya-x.json", "maya-none.json", None):
            s, _ = self.req("POST", "/api/models/delete", {"file": f})
            self.assertEqual(s, 400, f)
        # the running one with a model to switch to: only where Maya can start again by itself
        s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "then": self.other.name})
        self.assertEqual(s, 400)
        self.assertIn("cannot switch", b["error"]["message"])
        self.assertTrue(self.cfg.exists())
        self.assertFalse((self.d / M.DELETE_FILE).exists())

    def test_the_running_one_after_a_switch(self):
        os.environ["MAYA_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(Service, "_restart_into") as restart:
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "dry_run": True})
            self.assertEqual((s, b["running"]), (200, True))
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name})       # (no model to switch to)
            self.assertEqual(s, 400)
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "then": self.cfg.name})
            self.assertEqual(s, 400)                                                     # (not itself)
            self.assertFalse((self.d / M.DELETE_FILE).exists())
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "then": self.other.name})
        self.assertEqual((s, b["deleting"], b["to"]), (202, self.cfg.name, self.other.name))
        self.assertEqual(restart.call_args[0][0].name, self.other.name)
        self.assertTrue(self.cfg.exists())                    # (it runs until the switch)
        # the next server, on the other model: it deletes it, once
        self.svc.config_path = str(self.other)
        self.svc.finish_delete()
        self.assertFalse(self.cfg.exists() or (self.d / M.DELETE_FILE).exists())
        self.assertEqual((self.svc.deleted["deleted"], self.svc.deleted["model"]), (self.cfg.name, "glm-5.3-flash"))
        s, b = self.req("GET", "/api/models")
        self.assertEqual((b["deleted"]["deleted"], [m["file"] for m in b["models"]]), (self.cfg.name, [self.other.name]))

    def test_a_switch_that_did_not_happen_deletes_nothing(self):
        note = {"config": str(self.cfg.resolve()), "files": True, "time": time.time()}
        (self.d / M.DELETE_FILE).write_text(json.dumps(note))
        self.svc.finish_delete()                              # (the same model runs again)
        self.assertTrue(self.cfg.exists())
        self.assertIn("running now", self.svc.deleted["error"])
        (self.d / M.DELETE_FILE).write_text(json.dumps(dict(note, time=time.time() - 3600)))
        self.svc.config_path = str(self.other)
        self.svc.finish_delete()                              # (a stale note: an hour old)
        self.assertTrue(self.cfg.exists())
        self.assertFalse((self.d / M.DELETE_FILE).exists())

    def test_leftovers_from_the_page_only(self):
        models = self.d / "models"
        glm_quant(self.d, "Maya-S", models=models)            # (the folder stays known: a config downloaded into it)
        glm_quant(self.d, "Maya-L", models=models).unlink()
        with mock.patch.object(M, "LEFTOVER_MIN", 500):
            s, b = self.req("GET", "/api/models")
            [item] = b["leftovers"]["items"]
            s, _ = self.req("POST", "/api/models/leftovers", {"path": item["path"]}, {"Origin": "http://elsewhere.example"})
            self.assertEqual(s, 403)
            s, out = self.req("POST", "/api/models/leftovers", {"path": item["path"]})
        self.assertEqual((s, out["freed"]), (200, item["bytes"]))
        self.assertFalse(Path(item["path"]).exists())
        s, _ = self.req("POST", "/api/models/leftovers", {"path": item["path"]})
        self.assertEqual(s, 400)


if __name__ == "__main__":
    unittest.main()
