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


def ds4_cfg(d: Path, name="maya-gguf-q8.json", drafter=True, vision=False) -> Path:
    pack = d / "DeepSeek-V4.1-Flash" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "tensors.txt").write_text("t blk.0.x\n" + ("t mtp.0.w\n" if drafter else ""))
    (pack / "ds4.txt").write_text("hidden 4096\n" + ("vision.num_hidden_layers 24\n" if vision else ""))
    args = ["--pack", str(pack), "--max-context", "65536", "--tp", "2", "--cpu-threads", "16", "--vision", "off"]
    if drafter:
        args += ["--dspark", "on"]
    cfg = {"exe": "/b/maya-ds4", "args": args, "model_name": "deepseek-v4.1-flash", "gpu": [0, 1],
           "installer": {"family": "deepseek", "ds4_quant": "gguf-q8", "written": "2026-10-09 10:00"}}
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
                                              "draft": "none", "tp": 4, "kv_cache": "int8", "kv_streaming": True,
                                              "kv_resident": None, "cpu_threads": None})
        self.assertEqual(M.choices(cfg), {"draft": ["dspark", "none"], "images": True})
        new = M.apply(cfg, {"context": 262144, "draft": "dspark", "images": False, "tp": 2, "kv_cache": "fp8",
                            "kv_resident": 65536, "cpu_threads": 24})
        a = new["args"]
        for f, v in (("--max-context", "262144"), ("--dspark", "on"), ("--vision", "off"), ("--tp", "2"),
                     ("--kv-cache", "fp8"), ("--kv-streaming", "on"), ("--kv-resident", "65536"),
                     ("--cpu-threads", "24")):
            self.assertEqual(M.flag(a, f), v, f)
        # maya-mm3 stops at a flag it does not know: none of DeepSeek's or GLM's are written
        for f in ("--mmap-experts", "--prefill-chunk", "--reserve-mib", "--vram-reserve-mib", "--parallel", "--prefill"):
            self.assertNotIn(f, a)
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


class Page(unittest.TestCase):
    """GET /api/models, POST /api/models/settings and /api/models/switch."""

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


if __name__ == "__main__":
    unittest.main()
