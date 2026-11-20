"""serve/test_models.py - the web app's Models page: the run configs listed with their engine settings, the settings
written back into a config (checked first, its other keys kept), and a switch to another model - against the mock
engine, no GPU.

    python -m unittest serve.test_models -v
"""
from __future__ import annotations

import contextlib
import io
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


def hy4_cfg(d: Path, name="maya-hy4-flash-maya-xs.json") -> Path:
    """Hy4-Flash's config as maya.py's setup writes it."""
    model = d / "Hy4-Flash-Maya-XS"
    pack = model / "maya-hy4"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "hy4.txt").write_text(f"n_layer 32\ncontext_length 1048576\nmodel_dir {model}\n")
    (pack / "tensors.txt").write_text("file 0 Hy4-Flash-Maya-XS-IQ2_XXS-00001-of-00002.gguf\n"
                                      "file 1 Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf\nt token_embd.weight 1 0 0 8 2 4\n")
    args = ["--pack", str(pack), "--max-context", "32768", "--gpus", "0,1", "--cpu-threads", "12", "--reserve-gb", "0.5",
            "--chunk", "256", "--vram-experts", "0", "--profile", "/m/hy4.profile"]
    cfg = {"exe": "/b/maya-hy4", "args": args, "tokenizer": str(pack / "tokenizer"), "model_name": "hy4-flash",
           "gpu": [0, 1], "sampling": {"temperature": 0.9, "top_p": 1.0}, "reasoning_effort": "high",
           "installer": {"family": "hy4", "hy4_quant": "maya-xs", "quant": "Hy4-Flash-Maya-XS", "layout": "standard",
                         "models_dir": str(d), "model_dir": str(model), "written": "2026-11-19 10:00"}}
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
        # (Hy4-Flash's cards: 1 GB or more in use on each - the most the setup counts - unless a test reads its own)
        self.used = mock.patch.object(M, "hy4_used", side_effect=lambda gpus: {i: 1024 for i in gpus})
        self.used.start()
        # (... and no RAM figure - the VRAM alone, no KV streaming - unless a test gives one)
        self.ram = mock.patch.object(M, "hy4_ram_gb", return_value=None)
        self.ram.start()

    def tearDown(self):
        self.ram.stop()
        self.used.stop()
        self.tmp.cleanup()

    def test_glm_read(self):
        s = M.settings_of(M.read(glm_cfg(self.d)))
        self.assertEqual(s, {"context": 131072, "gpus": [0, 1], "env": {"MAYA_GLM_RAM_HEADROOM_GB": "4"}, "ablate": False,
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
        self.assertTrue(M.settings_of(dict(cfg, env={"MAYA_GLM_KV_RESIDENT": "65536"}))["kv_streaming"])
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

    def test_deepseek_fast_prompt_reading_is_an_opt_in_switch(self):
        from serve.server import child_env
        cfg = M.read(ds4_cfg(self.d))
        self.assertIs(M.settings_of(cfg)["fast_prefill"], False)                # (off by default)
        self.assertNotIn("DS4_YOCO_CONE", {k: v for k, v in child_env(cfg).items() if k not in os.environ})
        on = M.apply(cfg, {"fast_prefill": True})
        self.assertEqual((on["fast_prefill"], on["args"]), (True, cfg["args"]))  # (a config key, no engine flag)
        self.assertIs(M.settings_of(on)["fast_prefill"], True)
        self.assertEqual(child_env(on)["DS4_YOCO_CONE"], "1")
        off = M.apply(on, {"fast_prefill": False})
        self.assertNotIn("fast_prefill", off)                                   # (off: nothing in the config)
        self.assertEqual(child_env({**off, "fast_prefill": False})["DS4_YOCO_CONE"], "0")
        with self.assertRaisesRegex(ValueError, "Fast prompt reading"):
            M.apply(cfg, {"fast_prefill": "yes"})
        with self.assertRaisesRegex(ValueError, "not a setting"):              # (DeepSeek's only)
            M.apply(M.read(mm3_cfg(self.d)), {"fast_prefill": True})

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
        self.assertEqual(M.settings_of(cfg), {"context": 65536, "gpus": list(range(9)), "env": {}, "ablate": False, "images": True,
                                              "parallel": 1, "draft": "none", "tp": 4, "kv_cache": "int8",
                                              "kv_streaming": True, "kv_resident": None, "cpu_threads": None,
                                              "reserve_gb": None, "prefill_chunk": None, "cache_gb": None,
                                              "dspark_conf": None})
        self.assertEqual({k: v for k, v in M.choices(cfg).items() if k != "ablate"}, {"draft": ["dspark", "none"], "images": True})
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
        self.assertEqual(M.choices(bare)["ablate"]["available"], False)
        self.assertEqual({k: v for k, v in M.choices(bare).items() if k != "ablate"}, {"draft": ["none"], "images": False})
        with self.assertRaisesRegex(ValueError, "no DSpark drafter"):
            M.apply(bare, {"draft": "dspark"})
        self.assertEqual(M.quant_label(cfg, Path("maya-mm3.json")), "EXL3 3.5 bpw")

    def test_hy4_read_and_written_with_its_own_flags(self):
        cards = mock.patch.object(M, "machine_gpus", return_value=[{"index": i, "name": "V100", "vram_gb": 16.0}
                                                                   for i in range(4)])
        cards.start()
        self.addCleanup(cards.stop)
        cfg = M.read(hy4_cfg(self.d))
        self.assertEqual(M.family(cfg), "hy4")
        self.assertEqual(M.family({"exe": "C:/b/maya-hy4.exe", "args": ["--pack", "/p"]}), "hy4")   # (no installer)
        self.assertEqual(M.settings_of(cfg), {"context": 32768, "gpus": [0, 1], "env": {}, "ablate": False,
                                              "images": False, "draft": "none", "tp": "auto", "cpu_threads": 12,
                                              "reserve_gb": 0.5, "prefill_chunk": 256, "kv": "auto"})
        self.assertEqual(M.choices(cfg), {"draft": ["none"], "images": False,
                                          "ablate": {"available": False, "why": "not available for this model"}})
        self.assertEqual(M.quant_label(cfg, Path("maya-hy4-flash-maya-xs.json")), "Maya-XS")
        before = list(cfg["args"])
        self.assertEqual(M.apply(cfg, {})["args"], before)                      # (a save that changes nothing)
        self.assertEqual(M.apply(cfg, M.settings_of(cfg)), cfg)                 # (the page's: what it was shown, back)
        self.assertEqual(M.apply(cfg, {"parallel": 1, "kv_streaming": False, "kv_resident": None})["args"], before)
        new = M.apply(cfg, {"context": 16384, "cpu_threads": 24, "reserve_gb": 1.5, "env": {"MAYA_X": "1"}})
        a = new["args"]
        for f, v in (("--max-context", "16384"), ("--cpu-threads", "24"), ("--reserve-gb", "1.5"), ("--gpus", "0,1"),
                     ("--chunk", "256"), ("--vram-experts", "0"), ("--profile", "/m/hy4.profile")):
            self.assertEqual(M.flag(a, f), v, f)
        self.assertEqual(new["env"], {"MAYA_X": "1"})
        # maya-hy4 stops at a flag it does not know: none of the other engines' are written
        self.assertEqual({x for x in a if x.startswith("--")}, {x for x in before if x.startswith("--")})
        # empty: the engine's own (the flag goes)
        bare = M.apply(cfg, {"cpu_threads": None, "reserve_gb": ""})["args"]
        self.assertEqual(("--cpu-threads" in bare, "--reserve-gb" in bare), (False, False))
        # the GPUs: the config's list, and --gpus counting the cards picked (one card: no flag)
        one = M.apply(cfg, {"gpus": [1]})
        self.assertEqual((one["gpu"], "--gpus" in one["args"]), ([1], False))
        three = M.apply(cfg, {"gpus": [0, 2, 3]})
        self.assertEqual((three["gpu"], M.flag(three["args"], "--gpus")), ([0, 2, 3], "0,1,2"))
        # KV cache: auto (no flag, the default) or 8-bit rows (--kv q8: the setup's opt-in faster drafting) - another
        # setting left as it is; a config's own --kv f16 stays offered, other formats are refused
        q8 = M.apply(cfg, {"kv": "q8"})
        self.assertEqual((M.flag(q8["args"], "--kv"), M.settings_of(q8)["kv"]), ("q8", "q8"))
        self.assertEqual([x for x in q8["args"] if x not in ("--kv", "q8")], before)
        self.assertEqual(M.apply(M.apply(q8, {"draft": "none"}), {})["args"], q8["args"])   # (drafting: rows kept)
        self.assertEqual(M.apply(q8, {"kv": "auto"})["args"], before)
        f16 = {**cfg, "args": before + ["--kv", "f16"]}
        self.assertEqual(M.flag(M.apply(f16, M.settings_of(f16))["args"], "--kv"), "f16")
        for bad in ("f16", "int8"):
            with self.assertRaisesRegex(ValueError, "KV cache: auto or 8-bit"):
                M.apply(cfg, {"kv": bad})
        # what the engine does not have yet: a plain no, nothing written
        for key, value, why in (("parallel", 2, "one request at a time"), ("draft", "mtp", "this pack has no MTP block"),
                                ("images", True, "this pack has no vision tower"), ("kv_cache", "fp8", "auto or 8-bit"),
                                ("kv_streaming", True, "KV streaming: .* by itself"),
                                ("kv_resident", 8192, "KV streaming: .* by itself"),
                                ("ablate", True, "Refusal ablation: not available for this model")):
            with self.assertRaisesRegex(ValueError, why):
                M.apply(cfg, {key: value})
        for key in ("mmap", "reserve_mib", "expert_cache", "dspark_conf", "cache_gb"):
            with self.assertRaisesRegex(ValueError, "not a setting of this model: " + key):
                M.apply(cfg, {key: 1})
        with self.assertRaisesRegex(ValueError, "CPU threads"):
            M.apply(cfg, {"cpu_threads": 0})
        self.assertEqual(M.turbo_patch(cfg), {})                                # (Turbo: nothing of the engine's)
        # its pack: the context the model was trained for, the files it reads (delete, sizes)
        M._trained.clear()
        self.assertEqual(M.trained_context_of(cfg), 1048576)
        pack = M.pack_of(cfg)
        self.assertEqual(M.pack_reads(pack), [pack.parent / "Hy4-Flash-Maya-XS-IQ2_XXS-00001-of-00002.gguf",
                                              pack.parent / "Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf"])
        with mock.patch.object(M, "machine_gpus", return_value=[]):
            row = M.listing(self.d, None)["models"][0]
        # (no VRAM figure: the contexts up to the one the setup wrote - test_hy4_contexts_that_fit)
        self.assertEqual((row["family"], row["quant"], row["contexts"]), ("hy4", "Maya-XS", [4096, 8192, 16384, 32768]))
        self.assertEqual(row["context_range"], {"min": 4096, "max": 32768, "trained": 1048576, "tested": 32768})

    def test_hy4_a_new_context_gets_the_setups_chunk_again(self):
        """A small-card layout's prompt chunk is the setup's choice for one context (maya.py's hy4_fit: a picture's
        room first, the larger chunk only beside the largest picture's).  A context changed on the Models page gets
        that choice again, so the larger chunk never takes a picture's room there - unless the chunk is its user's."""
        def cfg_with(ctx, chunk, layout="8gb", extra=(), name="maya-hy4-flash-maya-xs.json"):
            p = hy4_cfg(self.d, name)
            c = json.loads(p.read_text())
            pack = Path(M.flag(c["args"], "--pack"))
            (pack / "hy4.txt").write_text((pack / "hy4.txt").read_text().split("vision.")[0] + "vision.dim 1152\n")
            a = ["--pack", str(pack), "--max-context", str(ctx)] + (["--chunk", str(chunk)] if chunk else [])
            a += ["--reserve-gb", "0.5", "--vram-experts", "0"] if layout == "8gb" else []
            c.update(args=a + list(extra), gpu=[0])
            c["installer"]["layout"] = layout
            p.write_text(json.dumps(c, indent=1))
            return M.read(p)

        def cards(gb):
            return mock.patch.object(M, "machine_gpus", return_value=[{"index": 0, "name": "x", "vram_gb": gb}])

        def chunk(c):
            return M.flag(c["args"], "--chunk")

        def by_hand(c):
            return c["installer"].get("chunk_by_hand")
        wide = cfg_with(4096, 512)                     # (the setup's on a card with room for both at 4K)
        self.assertTrue(M.settings_of(wide)["images"])
        with cards(9.5):                               # 8.5 GB for the model: 6.5 + 0.95 at 4K (pictures take none)
            self.assertEqual(M.apply(wide, M.settings_of(wide)), wide)      # (a save that changes nothing)
            self.assertEqual(chunk(M.apply(wide, {"cpu_threads": 8})), "512")
            # (8K: 6.8 + 0.95 fits; 16K and 32K: 7.6 / 7.9 in F16 rows + 0.95 is more than there is)
            for ctx, want in ((8192, "512"), (16384, "256"), (32768, "256")):
                new = M.apply(wide, {**M.settings_of(wide), "context": ctx})   # (the page sends all it shows)
                self.assertEqual((M.flag(new["args"], "--max-context"), chunk(new), by_hand(new)), (str(ctx), want, None))
                fit = M.hy4_refit(new)                 # ... and the largest picture is read either way
                self.assertEqual((fit["vision"], fit["vision_small"], fit["chunk"]), (True, False, int(want)), ctx)
                self.assertNotIn("--kv", new["args"])
            at32 = M.apply(wide, {"context": 32768})
            self.assertEqual(M.hy4_refit(at32)["kv"], "f16")   # (no 8-bit rows for a picture's sake)
            # back at 4K: the larger chunk again, as the setup writes it there
            self.assertEqual(chunk(M.apply(at32, {"context": 4096})), "512")
            # pictures on or off: the same chunk (they take no VRAM of their own)
            self.assertEqual(chunk(M.apply(at32, {"context": 8192, "images": False})), "512")
            self.assertEqual(chunk(M.apply(at32, {"context": 8192})), "512")
            self.assertEqual(chunk(M.apply(at32, {"context": 16384, "images": False})), "256")
            # another VRAM kept free than the layout's: the sums are not for it - the layout's own chunk
            self.assertEqual(chunk(M.apply(at32, {"context": 4096, "reserve_gb": 0.25})), "256")
            # a chunk set by hand stays: one that is not the layout's ...
            hand = cfg_with(4096, 384)
            self.assertTrue(M.hy4_chunk_by_hand(hand))
            self.assertEqual(chunk(M.apply(hand, {"context": 32768})), "384")
            none = cfg_with(4096, None)
            self.assertNotIn("--chunk", M.apply(none, {"context": 32768})["args"])
            # ... and one the page's user changes: kept at every context from then on
            at32 = M.apply(cfg_with(4096, 512), {"context": 32768})
            mine = M.apply(at32, {**M.settings_of(at32), "prefill_chunk": 512})
            self.assertEqual((chunk(mine), by_hand(mine)), ("512", True))
            for ctx in (4096, 8192, 32768):
                new = M.apply(mine, {**M.settings_of(mine), "context": ctx})
                self.assertEqual((chunk(new), by_hand(new)), ("512", True), ctx)
            with self.assertRaisesRegex(ValueError, r"at 65536 tokens; this card has 8\.5 GB for them"):
                M.apply(mine, {"context": 65536})      # (what fits is counted with the chunk that stays)
            both = M.apply(at32, {"context": 16384, "prefill_chunk": 512})  # (changed with the context: the user's)
            self.assertEqual((chunk(both), by_hand(both)), ("512", True))
            # put back to the layout's choice by its user: the layout's again
            back = M.apply(mine, {"prefill_chunk": 256})
            self.assertEqual((chunk(back), by_hand(back)), ("256", None))
            self.assertEqual(chunk(M.apply(back, {"context": 4096})), "512")
        tight = cfg_with(4096, 512)
        with cards(8.4):                               # 7.4 GB: 32K in 8-bit rows - the largest picture all the same
            new = M.apply(tight, {"context": 32768})
            fit = M.hy4_refit(new)
            self.assertEqual((chunk(new), fit["kv"], fit["vision"], fit["vision_small"]), ("256", "q8", True, False))
            # the page's Reset (what the setup wrote, its chunk with its context): not a chunk set by hand
            new["installer"]["as_set_up"] = M.settings_of(tight)
            reset = M.apply(new, new["installer"]["as_set_up"])
            self.assertEqual((M.flag(reset["args"], "--max-context"), chunk(reset), by_hand(reset)), ("4096", "512", None))
        with cards(8.0):                               # an 8 GB card taken to have 1 GB in use: 7.0 GB
            # (the contexts offered count the chunk the page would write, not the 512 of the context it has)
            self.assertEqual(M.listing(self.d, None)["models"][0]["contexts"], [4096, 8192, 16384])
            for ctx in (8192, 16384):
                self.assertEqual(chunk(M.apply(tight, {"context": ctx})), "256")
            with self.assertRaisesRegex(ValueError, "largest that fits is 16384"):
                M.apply(tight, {"context": 32768})
            # a chunk of its user's: only the contexts that fit with it are offered, and taken
            mine = dict(tight, installer={**tight["installer"], "chunk_by_hand": True})
            self.assertEqual(M.hy4_contexts(mine), [4096])
            with self.assertRaisesRegex(ValueError, r"need about 7\.8 GB of VRAM at 8192 tokens"):
                M.apply(mine, {"context": 8192})
            row = M.listing(self.d, None)["models"][0]
            self.assertFalse([k for k in row if "8bit" in k or "q8" in k])
        # a large card, the standard layout: the engine's own chunk - none written, one found kept
        for extra in ((), ("--chunk", "2048")):
            big = cfg_with(32768, None, "standard", extra)
            with cards(24.0):
                for ctx in (4096, 65536):
                    new = M.apply(big, {**M.settings_of(big), "context": ctx})
                    self.assertEqual(new["args"], M.set_flag(big["args"], "--max-context", ctx))
                    self.assertEqual(new["installer"], big["installer"])

    def test_hy4_page_on_an_amd_apu_counts_the_pcs_memory(self):
        """An AMD config ("backend": "hip") asks for the AMD cards, which keep what says they are AMD: the page then
        counts a unified-memory APU's memory as the setup does (the PC's, not the BIOS's 0.5 GB of VRAM; nvidia-smi's
        reading of another card is not its)."""
        import maya
        import setup as SU
        p = hy4_cfg(self.d)
        c = json.loads(p.read_text())
        c.update(args=["--pack", M.flag(c["args"], "--pack"), "--max-context", "32768"], gpu=[0], backend="hip")
        p.write_text(json.dumps(c, indent=1))
        cfg = M.read(p)
        halo = [{"index": 0, "name": "AMD Radeon (gfx1151)", "vram_gb": 0.5, "arch": "gfx1151", "vendor": "amd"}]
        M._gpu_cache.clear()
        M._hy4_used_cache.clear()
        asked = []
        try:
            with mock.patch.object(SU, "amd_gpus", side_effect=lambda *a: asked.append("amd") or halo), \
                    mock.patch.object(SU, "gpus", side_effect=AssertionError("the NVIDIA list is not this config's")), \
                    mock.patch.object(maya, "mem_gb", return_value=(61.0, 57.0)), mock.patch.object(maya, "WIN", False), \
                    mock.patch.object(maya, "amd_gpu_limit_gib", return_value=None), \
                    mock.patch.object(maya, "gpu_used_mib", return_value={0: 31000}), \
                    mock.patch.object(maya, "maya_engine_mib", return_value={}):
                self.assertEqual(M.machine_gpus("hip"), halo)
                room = M.hy4_room(cfg)
            self.assertEqual(asked, ["amd"])
            self.assertAlmostEqual(room["usable"], 53.0)               # (61 GB less the 8 GB kept for the system)
            self.assertEqual(room["contexts"], M.HY4_CONTEXTS)
        finally:
            M._gpu_cache.clear()
            M._hy4_used_cache.clear()

    def test_hy4_page_offers_the_streamed_contexts_the_setup_offers(self):
        """KV streaming (maya-hy4 --kv-streaming auto): with this PC's RAM read, the Models page offers the longer
        contexts the setup's hy4_fit offers - those the VRAM holds only with the global layers' older rows in RAM -
        and a config moved to one gets --kv-streaming auto; one that fits without it, or a config with streaming off
        by hand, is as before."""
        import maya
        p = hy4_cfg(self.d)
        c = json.loads(p.read_text())
        c.update(args=["--pack", M.flag(c["args"], "--pack"), "--max-context", "16384", "--chunk", "256",
                       "--reserve-gb", "0.5", "--vram-experts", "0"], gpu=[0])
        c["installer"]["layout"] = "8gb"
        p.write_text(json.dumps(c, indent=1))
        small = M.read(p)
        xs = maya.hy4_quant_table("maya-xs")
        card = [{"index": 0, "vram_gb": 8.0}]
        with mock.patch.object(M, "machine_gpus", return_value=[{"index": 0, "name": "x", "vram_gb": 8.0}]):
            before = M.hy4_room(small)["contexts"]
            with mock.patch.object(M, "hy4_ram_gb", return_value=64.0):
                fit = maya.hy4_fit(xs, card, used={0: 1024}, layout="8gb", ram_gb=64.0)
                room = M.hy4_room(small)
                self.assertEqual(room["contexts"], fit["contexts"])
                self.assertGreater(len(fit["streamed"]), 0)
                self.assertEqual([x for x in room["contexts"] if x not in fit["streamed"]], before)
                longest = max(fit["streamed"])
                a = M.apply(small, {"context": longest})["args"]
                self.assertEqual((M.flag(a, "--max-context"), M.flag(a, "--kv-streaming")), (str(longest), "auto"))
                self.assertIsNone(M.flag(M.apply(small, {"context": max(before)})["args"], "--kv-streaming"))
                # (the page's Context size line: RAM a token and the positions in VRAM at the streamed contexts)
                row = M.listing(self.d, None)["models"][0]
                self.assertEqual(sorted(int(x) for x in row["kv_streamed"]), fit["streamed"])
                self.assertIn(row["kv_streamed"][str(longest)]["ram"], (13 * 1152, 13 * 672))
                self.assertGreaterEqual(row["kv_streamed"][str(longest)]["win"], 256)
                with self.assertRaisesRegex(ValueError, f"the largest that fits is {longest}"):
                    M.apply(small, {"context": 1048576})          # (1M: the pooled keys alone do not fit 8 GB)
                off = dict(small, args=small["args"] + ["--kv-streaming", "off"])
                self.assertEqual(M.hy4_room(off)["contexts"], before)
                with self.assertRaisesRegex(ValueError, f"the largest that fits is {max(before)}"):
                    M.apply(off, {"context": longest})
            with mock.patch.object(M, "hy4_ram_gb", return_value=16.0):   # (a small RAM: no room for the rows)
                self.assertEqual(M.hy4_room(small)["contexts"], before)

    def test_hy4_page_reads_the_vram_in_use_as_the_setup_does(self):
        """The Models page offers the contexts the setup offers: the cards' VRAM in use is the setup's own reading
        (nvidia-smi's figure now, at most 1 GB a card counted) - and a loaded Maya engine's own VRAM is not "in use",
        else the page would refuse the config that is running."""
        import maya
        p = hy4_cfg(self.d)
        c = json.loads(p.read_text())
        c.update(args=["--pack", M.flag(c["args"], "--pack"), "--max-context", "16384", "--chunk", "256",
                       "--reserve-gb", "0.5", "--vram-experts", "0"], gpu=[0])
        c["installer"]["layout"] = "8gb"
        p.write_text(json.dumps(c, indent=1))
        small = M.read(p)
        xs = maya.hy4_quant_table("maya-xs")
        asked = []

        def pc(used, own=None, before=None):
            """nvidia-smi's answers: MiB in use on the card, of it a Maya engine's (None: not said), the reading the
            server took before it started the engine (None: none taken)."""
            M._hy4_used_cache.clear()
            M._HY4_BEFORE.clear()
            st = contextlib.ExitStack()
            st.enter_context(mock.patch.object(maya, "gpu_used_mib", side_effect=lambda: asked.append(1) or dict(used)))
            st.enter_context(mock.patch.object(maya, "maya_engine_mib", return_value=own or {}))
            st.enter_context(mock.patch.object(M, "machine_gpus", return_value=[{"index": 0, "name": "x", "vram_gb": 8.0}]))
            if before is not None:
                with mock.patch.object(maya, "gpu_used_mib", return_value=before):
                    M.hy4_note_before_start()
            return st
        refused = r"need about 7\.1 GB of VRAM at 32768 tokens; this card has 7\.0 GB for them"
        self.used.stop()
        try:
            # the owner's 8 GB card: 571 MiB in use - the setup offers 32K, and so does the page
            with pc({0: 571}):
                fit = maya.hy4_fit(xs, [{"index": 0, "vram_gb": 8.0}], used={0: 571}, layout="8gb")
                self.assertEqual(fit["contexts"], [4096, 8192, 16384, 32768])
                self.assertEqual(M.hy4_room(small)["contexts"], fit["contexts"])
                self.assertAlmostEqual(M.hy4_room(small)["usable"], fit["usable"])
                self.assertEqual(M.flag(M.apply(small, {"context": 32768})["args"], "--max-context"), "32768")
                row = M.listing(self.d, None)["models"][0]
                self.assertEqual((row["contexts"], row["context_range"]["max"]), (fit["contexts"], 32768))
                self.assertEqual(len(asked), 1)                       # (one reading for all of it: kept for 10 s)
            for used in ({0: 1024}, {0: 3000}):                       # (1 GB or more in use: at most 1 GB counts)
                with pc(used):
                    self.assertEqual(M.hy4_room(small)["contexts"],
                                     maya.hy4_fit(xs, [{"index": 0, "vram_gb": 8.0}], used=used, layout="8gb")["contexts"])
                    with self.assertRaisesRegex(ValueError, refused):
                        M.apply(small, {"context": 32768})
            # the model is loaded on the card (7.4 GB in use, most of it the engine's own): not counted
            with pc({0: 7400}, own={0: 6829.0}):                      # (nvidia-smi says how much is the engine's)
                self.assertEqual(M.flag(M.apply(small, {"context": 32768})["args"], "--max-context"), "32768")
                self.assertAlmostEqual(M.hy4_room(small)["usable"], 8 - 571 / 1024)
            with pc({0: 7400}, own={0: None}, before={0: 571}):       # (Windows: the reading from before its start)
                self.assertEqual(M.flag(M.apply(small, {"context": 32768})["args"], "--max-context"), "32768")
                self.assertAlmostEqual(M.hy4_room(small)["usable"], 8 - 571 / 1024)
            with pc({0: 7400}, own={0: None}):                        # (neither: the card as it reads, 1 GB counted)
                with self.assertRaisesRegex(ValueError, refused):
                    M.apply(small, {"context": 32768})
            with pc({}):                                              # (no reading: the setup's sum without one)
                self.assertAlmostEqual(M.hy4_room(small)["usable"], maya.hy4_usable_gib([{"index": 0, "vram_gb": 8.0}], {}))
            # the server notes the reading just before it starts Hy4-Flash's engine (no other engine's start asks)
            for exe, noted in (("/b/maya-hy4", {0: 571}), ("C:/b/maya-hy4.exe", {0: 571}), ("/b/maya", {}), ("/b/maya-mm3", {})):
                M._HY4_BEFORE.clear()
                with mock.patch.object(maya, "gpu_used_mib", return_value={0: 571}), \
                        mock.patch.object(SV.subprocess, "Popen", side_effect=OSError("not started")):
                    with self.assertRaises(OSError):
                        SV.StrataEngine(exe, ["--pack", "x"])
                self.assertEqual(M._HY4_BEFORE, noted, exe)
        finally:
            M._hy4_used_cache.clear()
            M._HY4_BEFORE.clear()
            self.used.start()

    def test_hy4_page_counts_the_vram_windows_gives_the_engine(self):
        """Windows keeps part of every card from the programs on it: the setup measured what the engine's CUDA gets
        (maya.py's hy4_engine_room, kept in the config with the card's VRAM) and the page counts no more than that -
        on the card it was measured on; a config without it, or another card now: as before."""
        import maya
        p = hy4_cfg(self.d)
        c = json.loads(p.read_text())
        c.update(args=["--pack", M.flag(c["args"], "--pack"), "--max-context", "8192", "--chunk", "256",
                       "--reserve-gb", "0.5", "--vram-experts", "0", "--dfly2"], gpu=[0])
        c["installer"].update(layout="8gb", hy4_quant="maya-xs", engine_room_gb={"0": [7.48, 8.0]})
        p.write_text(json.dumps(c, indent=1))
        cfg = M.read(p)
        for vram, usable in ((8.0, 7.48), (12.0, 12.0 - 76 / 1024)):    # (76 MiB in use, nvidia-smi's reading)
            with mock.patch.object(M, "machine_gpus", return_value=[{"index": 0, "name": "x", "vram_gb": vram}]), \
                    mock.patch.object(M, "hy4_used", side_effect=lambda gpus: {i: 76 for i in gpus}):
                self.assertAlmostEqual(M.hy4_room(cfg)["usable"], usable)
                self.assertEqual(M.hy4_cards(cfg, M.machine_gpus())[0].get("engine_room_gb"),
                                 7.48 if vram == 8.0 else None)
                fit = M.hy4_refit(cfg)
                self.assertAlmostEqual(fit["usable"], usable)
        self.assertEqual(M.hy4_cards({"installer": {"engine_room_gb": {"0": "x", "a": [1, 2]}}},
                                     [{"index": 0, "vram_gb": 8.0}]), {0: {"index": 0, "vram_gb": 8.0}})
        self.assertEqual(maya.hy4_usable_gib([{"index": 0, "vram_gb": 8.0}], {0: 1024}), 7.0)

    def test_hy4_page_counts_the_share_the_engine_measured(self):
        """The setup kept CUDA's reading beside the room (its first guess for CUDA's own share): once the engine has
        measured the share on that card (maya.py's HY4_ROOM_STORE, this PC, card and driver), the page counts the
        reading with it, as the next setup does; no record, or a config without the reading: the setup's figure."""
        import maya
        cfg = {"installer": {"engine_room_gb": {"0": [7.537, 8.0, 7.037]}}}
        gl = [{"index": 0, "name": "x", "vram_gb": 8.0}]
        M._gpu_cache.pop("room_own", None)
        self.addCleanup(M._gpu_cache.pop, "room_own", None)
        for own, room in (({0: 0.42}, 7.457), ({}, 7.537)):
            with mock.patch.object(M, "hy4_room_own", return_value=own):
                self.assertAlmostEqual(M.hy4_cards(cfg, gl)[0]["engine_room_gb"], room)
        with mock.patch.object(M, "hy4_room_own", side_effect=AssertionError("not asked")):
            self.assertAlmostEqual(M.hy4_cards({"installer": {"engine_room_gb": {"0": [7.48, 8.0]}}}, gl)[0]
                                   ["engine_room_gb"], 7.48)
        # the record for this PC's card and driver, by the setup's own key (setup.py's list: its PCI id and driver)
        raw = [{"index": 0, "name": "x", "vram_gb": 8.0, "pci": "00000000:01:00.0", "driver": "610.88"}]
        with mock.patch.object(maya, "hy4_room_load", return_value={maya.hy4_room_key(raw[0]): {"own_gib": 0.42}}), \
                mock.patch.object(maya.S, "gpus", return_value=raw):
            self.assertEqual(M.hy4_room_own(), {0: 0.42})
            M._gpu_cache.pop("room_own", None)
            with mock.patch.object(maya.S, "gpus", return_value=[dict(raw[0], driver="615.10")]):
                self.assertEqual(M.hy4_room_own(), {})
        # the server's start: the card to measure, and the figure said once kept
        with mock.patch.object(maya, "hy4_room_due", return_value={"index": 0}):
            self.assertEqual(M.hy4_room_due({}), {"index": 0})
        with mock.patch.object(maya, "hy4_room_due", side_effect=OSError("no driver")):
            self.assertIsNone(M.hy4_room_due({}))
        with mock.patch.object(maya, "hy4_room_measure", return_value={"own_gib": 0.42}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(M.hy4_room_measured({}, {"name": "RTX"}, {}), {"own_gib": 0.42})
        self.assertIn("0.42 GB", out.getvalue())
        with mock.patch.object(maya, "hy4_room_measure", side_effect=ValueError):
            self.assertIsNone(M.hy4_room_measured({}, {}, {}))

    def test_hy4_contexts_that_fit(self):
        """Hy4-Flash's context on the Models page: only what fits the config's cards by its layout's plan, as the
        setup works it out (maya.py's hy4_fit) - the engine has no RAM fallback for these parts and stops at its
        start, at every start from then on, with a context that does not fit."""
        def cfg_with(layout, ctx, gpus, name):
            p = hy4_cfg(self.d, name)
            c = json.loads(p.read_text())
            a = ["--pack", M.flag(c["args"], "--pack"), "--max-context", str(ctx)]
            if layout == "8gb":                        # (the layout's own flags, as hy4_fit writes them)
                a += ["--chunk", "256", "--reserve-gb", "0.5", "--vram-experts", "0"]
            a += ["--gpus", ",".join(str(i) for i in range(len(gpus)))] if len(gpus) > 1 else []
            c.update(args=a, gpu=gpus)
            c["installer"]["layout"] = layout
            p.write_text(json.dumps(c, indent=1))
            return M.read(p)

        def cards(*gb):
            return mock.patch.object(M, "machine_gpus", return_value=[{"index": i, "name": "x", "vram_gb": v}
                                                                      for i, v in enumerate(gb)])
        small = cfg_with("8gb", 8192, [0], "maya-hy4-flash-maya-xs.json")
        with cards(8.0):                               # an 8 GB card, the 8 GB layout at 8K (the Uranus tester's)
            # (16K fits with the latent cache in 8-bit rows - the engine's --kv auto takes them by itself; 32K: not
            #  on a card taken to have 1 GB in use)
            self.assertEqual(M.flag(M.apply(small, {"context": 16384})["args"], "--max-context"), "16384")
            self.assertNotIn("--kv", M.apply(small, {"context": 16384})["args"])
            with self.assertRaisesRegex(ValueError, r"Context: Hy4-Flash's parts that must sit on the GPU need about "
                                                    r"7\.1 GB of VRAM at 32768 tokens; this card has 7\.0 GB for them - "
                                                    r"the largest that fits is 16384"):
                M.apply(small, {"context": 32768})
            with self.assertRaisesRegex(ValueError, "largest that fits is 16384"):
                M.apply(small, {"context": 131072})
            f16 = dict(small, args=small["args"] + ["--kv", "f16"])   # (F16 rows by name: only what fits with them)
            with self.assertRaisesRegex(ValueError, r"need about 7\.6 GB of VRAM at 16384 tokens; this card has 7\.0 GB "
                                                    r"for them - the largest that fits is 8192"):
                M.apply(f16, {"context": 16384})
            self.assertEqual((M.hy4_room(small)["contexts_q8"], M.hy4_room(f16)["contexts_q8"]), ([16384], []))
            self.assertEqual(M.flag(M.apply(small, {"context": 8192})["args"], "--max-context"), "8192")
            self.assertEqual(M.flag(M.apply(small, {"context": 4096})["args"], "--max-context"), "4096")
            self.assertEqual(M.apply(small, M.settings_of(small)), small)
            with self.assertRaisesRegex(ValueError, r"need about 8\.3 GB of VRAM at 8192 tokens"):
                M.apply(small, {"reserve_gb": 2})       # (more VRAM kept free: the context has to fit with it)
            with self.assertRaisesRegex(ValueError, "could not be read"):
                M.apply(small, {"gpus": [0, 1], "context": 32768})   # (a card this PC does not list adds no VRAM)
            row = M.listing(self.d, None)["models"][0]
            self.assertEqual((row["contexts"], row["context_range"]["max"]), ([4096, 8192, 16384], 16384))
            # (the page never names the cache's rows: it gets where the smaller ones start, for its GB figure, and
            #  whether a larger context needs more VRAM - one line under the slider)
            self.assertEqual((row["kv_small_from"], row["context_more"]), (16384, True))
            self.assertNotIn("contexts_8bit", row)
        with cards(8.0, 8.0):                          # a second card: room for more
            self.assertEqual(M.flag(M.apply(small, {"gpus": [0, 1], "context": 32768})["args"], "--max-context"), "32768")
        with cards(4.0, 4.0):
            with self.assertRaisesRegex(ValueError, "these 2 cards have 6.0 GB for them - no context fits"):
                M.apply(small, {"gpus": [0, 1]})
        # the standard layout on a 32 GB card: every context offered is taken
        big = cfg_with("standard", 32768, [0], "maya-hy4-flash-maya-xs.json")
        with cards(32.0):
            for c in M.HY4_CONTEXTS:
                self.assertEqual(M.flag(M.apply(big, {"context": c})["args"], "--max-context"), str(c))
            row = M.listing(self.d, None)["models"][0]
            self.assertEqual((row["contexts"], row["context_range"]["max"]), (M.HY4_CONTEXTS, 1048576))   # (the model's 1M)
            self.assertEqual((row["kv_small_from"], row["context_more"]), (None, False))
        with cards(10.0):                              # ... and on a 10 GB one: up to what its plan fits
            with self.assertRaisesRegex(ValueError, r"at 65536 tokens; this card has 9\.0 GB for them"):
                M.apply(big, {"context": 65536})
            self.assertEqual(M.listing(self.d, None)["models"][0]["contexts"], [4096, 8192, 16384, 32768])
        # no VRAM figure (nvidia-smi does not answer, or a card the PC does not list): only up to the context it has
        for none in (cards(), cards(None), mock.patch.object(M, "hy4_room", return_value=None)):
            with none:
                with self.assertRaisesRegex(ValueError, "Context: the VRAM of this config's cards could not be read, "
                                                        "so Hy4-Flash keeps at most the 32768 tokens"):
                    M.apply(big, {"context": 65536})
                for c in (32768, 16384, 4096):
                    self.assertEqual(M.flag(M.apply(big, {"context": c})["args"], "--max-context"), str(c))
                self.assertEqual(M.listing(self.d, None)["models"][0]["contexts"], [4096, 8192, 16384, 32768])
        # the other models' contexts: as they were, whatever the cards
        with cards(8.0):
            glm = M.read(glm_cfg(self.d, "maya-glm.json"))
            self.assertEqual(M.flag(M.apply(glm, {"context": 131072})["args"], "--max-context"), "131072")

    def test_exl3_label_from_the_folder(self):
        """A DeepSeek EXL3 quant set up by hand under the "exl3" key (turboderp's 2.25 bpw): its folder says its bits,
        not LibertAIDAI's 2.9 the key names; the setup's own folders keep the setup's label."""
        own = {"exe": "maya-ds4", "args": ["--pack", "/m/DeepSeek-V4.1-Flash-1017-EXL3-2.25bpw/maya-ds4"],
               "installer": {"family": "deepseek", "ds4_quant": "exl3",
                             "model_dir": "/m/DeepSeek-V4.1-Flash-1017-EXL3-2.25bpw"}}
        self.assertEqual(M.quant_label(own, Path("maya-deepseek-exl3-225-hip.json")), "EXL3 2.25 bpw")
        del own["installer"]["model_dir"]                       # (the pack's folder says it too)
        self.assertEqual(M.quant_label(own, Path("x.json")), "EXL3 2.25 bpw")
        setup = {"exe": "maya-ds4", "args": ["--pack", "/m/DeepSeek-V4.1-Flash-1017/maya-ds4"],
                 "installer": {"family": "deepseek", "ds4_quant": "exl3", "model_dir": "/m/DeepSeek-V4.1-Flash-1017"}}
        self.assertEqual(M.quant_label(setup, Path("maya-ds4.json")), "EXL3 2.9 bpw")
        setup35 = {"exe": "maya-ds4", "args": ["--pack", "/m/DeepSeek-V4.1-Flash-1017-3.5bpw/maya-ds4"],
                   "installer": {"family": "deepseek", "ds4_quant": "exl3-3.5"}}
        self.assertEqual(M.quant_label(setup35, Path("maya-ds4.json")), "EXL3 3.5 bpw")
        gguf = {"exe": "maya-ds4", "args": ["--pack", "/m/Some-3bpw-GGUF/maya-ds4"],
                "installer": {"family": "deepseek", "ds4_quant": "gguf-q8"}}
        self.assertEqual(M.quant_label(gguf, Path("maya-ds4.json")), "UD-Q8_K_XL")   # (a GGUF: never read)

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
        # a GGUF folder of your own, outside the models folders: nothing in it is deleted, the pack Maya built in it too
        own = self.d / "elsewhere"
        (self.d / "models").mkdir()
        l = glm_quant(self.d, "Maya-L", folder=own, models=self.d / "models")
        plan = M.delete_plan(self.d, l)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["items"], [])
        self.assertIn("Your own model files", plan["kept"][0])
        self.assertIn("outside Maya's models folders", plan["kept"][1])
        out = M.delete(self.d, l, files=True)
        self.assertEqual(out["freed"], 0)
        self.assertEqual(len(list(own.glob("*.gguf"))), 3)
        self.assertTrue((own / "pack" / "index.txt").exists())
        self.assertFalse(l.exists())                        # (off the list all the same)

    def test_a_drafter_another_model_drafts_with(self):
        # the DFlash2 drafter both quants draft with: it stays, the other model keeps drafting with it
        m = glm_quant(self.d, "Maya-M", dflash=True)
        l = glm_quant(self.d, "Maya-L", dflash=True)
        os.utime(m, (1_700_000_000, 1_700_000_000))
        plan = M.delete_plan(self.d, l)
        drafter = self.d / "GLM-5.3-Flash-DFlash2" / "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf"
        self.assertEqual(plan["drafter_users"], [m.name])
        self.assertEqual(plan["drafter"], None)
        self.assertEqual([(Path(s["path"]).name, s["what"], s["users"]) for s in plan["shared"]],
                         [(drafter.name, "drafter", [m.name])])
        self.assertNotIn("drafter", [i["what"] for i in plan["items"]])
        out = M.delete(self.d, l, files=True)
        self.assertEqual(out["drafter_users"], [])
        self.assertTrue(drafter.is_file())
        self.assertIn("--dflash", json.loads(m.read_text())["args"])
        self.assertEqual(m.stat().st_mtime, 1_700_000_000)  # (its config untouched)
        self.assertFalse((self.d / "GLM-5.3-Flash" / "Maya-L").exists())

    def test_what_another_model_shares_stays_and_is_named(self):
        # Maya-L's pack built in Maya-M's folder: the folder, Maya-M's GGUFs and the vision encoder both use stay
        folder = self.d / "GLM-5.3-Flash"
        m = glm_quant(self.d, "Maya-M", folder=folder)
        l = glm_quant(self.d, "Maya-L", folder=folder, pack_name="pack-maya-l")
        plan = M.delete_plan(self.d, l)
        got = {(s["what"], Path(s["path"]).name, tuple(s["users"])) for s in plan["shared"]}
        self.assertIn(("folder", "GLM-5.3-Flash", (m.name,)), got)
        self.assertIn(("vision", "mmproj.gguf", (m.name,)), got)
        self.assertEqual(plan["bytes"], sum(i["bytes"] for i in plan["items"]))

    def test_never_outside_the_models_folders(self):
        # a pack that is a link to a folder elsewhere, a vision file linked out: the real paths are outside - kept
        outside = self.d / "outside"
        (outside / "pack").mkdir(parents=True)
        (outside / "pack" / "index.txt").write_text("i")
        (outside / "mmproj.gguf").write_bytes(b"v" * 10)
        l = glm_quant(self.d, "Maya-L", models=self.d / "models")
        folder = self.d / "models" / "GLM-5.3-Flash" / "Maya-L"
        cfg = json.loads(l.read_text())
        try:
            os.symlink(outside / "pack", folder / "pack-link", target_is_directory=True)
            os.symlink(outside / "mmproj.gguf", folder / "vision" / "linked.gguf")
        except (OSError, NotImplementedError):
            self.skipTest("this file system cannot link")
        cfg["args"][1] = str(folder / "pack-link")
        cfg["vision"]["mmproj"] = str(folder / "vision" / "linked.gguf")
        l.write_text(json.dumps(cfg))
        plan = M.delete_plan(self.d, l)
        self.assertTrue(all(not Path(i["path"]).resolve().is_relative_to(outside.resolve()) for i in plan["items"]))
        M.delete(self.d, l, files=True)
        self.assertTrue((outside / "pack" / "index.txt").is_file() and (outside / "mmproj.gguf").is_file())

    def test_a_models_folder_maya_does_not_know(self):
        # a config naming a models folder that is not one (a drive's root, gone): nothing of its files is deleted
        l = glm_quant(self.d, "Maya-L")
        cfg = json.loads(l.read_text())
        cfg["installer"]["models_dir"] = Path(self.d).anchor
        l.write_text(json.dumps(cfg))
        plan = M.delete_plan(self.d, l)
        self.assertEqual((plan["ok"], plan["items"]), (False, []))
        self.assertIn("not one Maya downloads into", plan["why"])

    def test_hard_links_are_counted(self):
        m = glm_quant(self.d, "Maya-M")
        l = glm_quant(self.d, "Maya-L")
        shard = self.d / "GLM-5.3-Flash" / "Maya-L" / "GLM-5.3-Flash-Maya-L-00001-of-00003.gguf"
        try:
            os.link(shard, self.d / "GLM-5.3-Flash" / "Maya-M" / "tables.bin")
        except (OSError, NotImplementedError):
            self.skipTest("this file system cannot hard link")
        plan = M.delete_plan(self.d, l)
        self.assertEqual(plan["linked"], 1)
        self.assertEqual(plan["items"][0]["linked"], 1)
        self.assertTrue(m.exists())

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

    def test_never_a_folder_another_pack_reads_files_from(self):
        # an EXL3 quant's pack reading the Engram tables from the GGUF set's folder and the vision tower from another's
        # (its tensors.txt's file lines, relative to ds4.txt's model_dir): no config names those folders
        p = ds4_cfg(self.d)
        cfg = json.loads(p.read_text())
        exl3 = self.models / "DeepSeek-V4.1-Flash-EXL3-2.25bpw"
        pack = exl3 / "maya-ds4"
        pack.mkdir(parents=True)
        (exl3 / "model-00001-of-00001.safetensors").write_bytes(b"s" * 2000)
        (pack / "ds4.txt").write_text(f"hidden_size 5120\nmodel_dir {exl3}\n")
        (pack / "tensors.txt").write_text("file 0 model-00001-of-00001.safetensors\n"
                                          "file 1 ../DeepSeek-V4.1-Flash-Maya-GGUF/Engram/E-00008-of-00010.gguf\n"
                                          "file 2 ../DeepSeek-V4.1-Flash-GGUF/vision/vision.bin\n"
                                          "t embed.weight BF16 0 0 10 5 2\nt layers.0.engram.w BF16 1 0 10 5 2\n")
        cfg["args"][1] = str(pack)
        cfg["installer"]["models_dir"] = str(self.models)
        p.write_text(json.dumps(cfg))
        gguf, other = self.models / "DeepSeek-V4.1-Flash-Maya-GGUF", self.models / "DeepSeek-V4.1-Flash-GGUF"
        (gguf / "Engram").mkdir(parents=True)
        (gguf / "Engram" / "E-00008-of-00010.gguf").write_bytes(b"e" * 2000)
        (gguf / "DeepSeek-V4.1-Flash-Maya-S-00001-of-00002.gguf").write_bytes(b"g" * 2000)
        (other / "vision").mkdir(parents=True)
        (other / "vision" / "vision.bin").write_bytes(b"v" * 2000)
        old = self.models / "Unused"                       # (and one no pack reads: still listed)
        old.mkdir()
        (old / "x.gguf").write_bytes(b"x" * 2000)
        self.assertEqual([i["name"] for i in M.leftovers(self.d)["items"]], ["Unused"])
        with self.assertRaisesRegex(ValueError, "not a file no model uses"):
            M.delete_leftover(self.d, str(gguf))
        self.assertTrue((gguf / "Engram" / "E-00008-of-00010.gguf").is_file())

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

    def test_not_while_a_model_downloads(self):
        l = glm_quant(self.d, "Maya-L")
        with mock.patch.object(type(self.svc.pull_slot()), "busy", return_value=True):
            s, b = self.req("POST", "/api/models/delete", {"file": l.name, "files": True})
            self.assertEqual(s, 400)
            self.assertIn("downloading", b["error"]["message"])
            s, _ = self.req("POST", "/api/models/delete", {"file": l.name, "files": True, "dry_run": True})
            self.assertEqual(s, 200)                          # (what it would free: shown all the same)
        self.assertTrue(l.exists() and (self.d / "GLM-5.3-Flash" / "Maya-L").is_dir())

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


class Ablate(unittest.TestCase):
    """Refusal ablation on the Models page: the switch only with a direction file on an engine that projects it, the
    default kept beside the engine's arguments (no restart), the direction loaded off by default."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def with_direction(self, p: Path) -> dict:
        f = self.d / ("DeepSeek-V4.1-Flash" + M.ABLATE_SUFFIX)
        f.write_bytes(b"GGUF")
        cfg = M.read(p)
        cfg["ablate"] = {"file": str(f), "model": "DeepSeek-V4.1-Flash", "fitted_on": "Maya-S"}
        cfg["args"] = M.ablate_args(cfg, cfg["args"])
        return cfg

    def test_no_direction_no_switch(self):
        cfg = M.read(ds4_cfg(self.d))
        self.assertFalse(M.choices(cfg)["ablate"]["available"])
        self.assertIn("no refusal direction", M.choices(cfg)["ablate"]["why"])
        self.assertFalse(M.settings_of(cfg)["ablate"])
        with self.assertRaises(ValueError):
            M.apply(cfg, {"ablate": True})
        self.assertEqual(M.apply(cfg, {"ablate": False})["args"], cfg["args"])   # (the command as before)
        self.assertNotIn("--ablate", M.ablate_args(cfg, list(cfg["args"])))

    def test_direction_loaded_off_and_default_switch(self):
        cfg = self.with_direction(ds4_cfg(self.d))
        self.assertEqual(M.flag(cfg["args"], "--ablate-default"), "off")
        self.assertTrue(M.choices(cfg)["ablate"]["available"])
        on = M.apply(cfg, {"ablate": True})
        self.assertTrue(on["ablate"]["default"])
        self.assertTrue(M.settings_of(on)["ablate"])
        self.assertEqual(on["args"], cfg["args"])          # (an engine setting it is not: no restart)
        self.assertNotIn("sampling", on)                   # (the model's own sampling defaults stay its own)
        off = M.apply(on, {"ablate": False})
        self.assertFalse(M.settings_of(off)["ablate"])

    def test_glm_loads_it(self):
        cfg = M.read(glm_cfg(self.d))
        f = self.d / "x.gguf"
        f.write_bytes(b"GGUF")
        cfg["ablate"] = {"file": str(f)}
        self.assertTrue(M.choices(cfg)["ablate"]["available"])
        args = M.ablate_args(cfg, list(cfg["args"]))
        self.assertEqual(M.flag(args, "--ablate"), str(f))
        self.assertEqual(M.flag(args, "--ablate-default"), "off")
        cfg["backend"] = "metal"   # (the Mac engine has no projection)
        self.assertFalse(M.choices(cfg)["ablate"]["available"])
        self.assertEqual(M.choices(cfg)["ablate"]["why"], "not available for this model")
        self.assertNotIn("--ablate", M.ablate_args(cfg, list(cfg["args"])))

    def test_qwen_greyed_out(self):
        cfg = M.read(glm_cfg(self.d))
        cfg["installer"]["family"] = "qwen"   # (CYBER-FROST: the `maya` engine without the projection)
        f = self.d / "x.gguf"
        f.write_bytes(b"GGUF")
        cfg["ablate"] = {"file": str(f)}
        self.assertFalse(M.choices(cfg)["ablate"]["available"])
        self.assertEqual(M.choices(cfg)["ablate"]["why"], "not available for this model")
        self.assertNotIn("--ablate", M.ablate_args(cfg, list(cfg["args"])))


if __name__ == "__main__":
    unittest.main()
