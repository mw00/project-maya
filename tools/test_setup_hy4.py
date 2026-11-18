"""tools/test_setup_hy4.py - Hy4-flash in the setup, the launcher and the dashboard: its name and setup ids, its configs
recognised (maya-hy4's --pack is not DeepSeek's), the pinned download manifest, the context recommended by the
smallest GPU's VRAM, the config the setup writes (maya-hy4's own flags only, the model's own sampling and level), the
engine target the launcher builds and starts, the Models page's settings, and its two thinking levels (High, Off) in
the template and the web app.  No GPU, no network.

    python -m unittest tools.test_setup_hy4 -v
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402

SHARDS = ("Hy4-flash-Q4_K_M-00001-of-00002.gguf", "Hy4-flash-Q4_K_M-00002-of-00002.gguf")


def hy4_cfg(d: Path, mtp: bool = True, n_mtp: int = 1) -> dict:
    pack = d / "Hy4-flash" / "maya-hy4"
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    (pack / "hy4.txt").write_text(f"arch hy_v4_flash\nn_layer 42\nn_mtp {n_mtp}\ncontext_length 1048576\n"
                                  f"model_dir {pack.parent}\n")
    (pack / "tensors.txt").write_text("".join(f"file {i} {n}\n" for i, n in enumerate(SHARDS)) + "t x F32 0 0 4 1\n")
    return {"exe": "/b/build/maya-hy4", "args": ["--pack", str(pack), "--max-context", "65536"] + (["--mtp"] if mtp else []),
            "model_name": "hy4-flash", "gpu": [0, 1], "tokenizer": str(pack / "tokenizer"),
            "installer": {"family": "hy4", "quant": "Hy4-flash", "hy4_quant": "q4_k_m"}}


class Names(unittest.TestCase):
    def test_shown_and_ids(self):
        self.assertEqual(Q.shown("hy4", "q4_k_m", 168.35), "Hy4-flash · Q4_K_M (168 GB)")
        self.assertEqual(Q.config_stem("hy4", "q4_k_m"), "maya-hy4-flash")
        for name in ("hy4-flash", "Hy4-flash", "hy4", "hy4-flash-q4_k_m", "HY4-Q4_K_M"):
            self.assertEqual(Q.resolve(name), ("hy4", "q4_k_m"), name)
        self.assertEqual((maya.HY4, maya.HY4_MODEL_NAME, maya.HY4_SHOWN), ("Hy4-flash", "hy4-flash", "Hy4-flash"))

    def test_family(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = hy4_cfg(Path(t))
            self.assertEqual(Q.family_of(cfg), "hy4")
            cfg.pop("installer")                       # (by its engine too: maya-hy4's --pack is not DeepSeek's)
            self.assertEqual(Q.family_of(cfg), "hy4")
            c = hy4_cfg(Path(t) / "b")
            self.assertEqual((Q.config_key(c), Q.config_label(c)), ("q4_k_m", "Hy4-flash · Q4_K_M"))
            self.assertEqual(Q.model_defaults(c), ({"temperature": 0.9, "top_p": 1.0}, "high"))
            self.assertEqual(maya.setup_title(c), "Hy4-flash")

    def test_model_option_and_family_choice(self):
        a = type("A", (), {"model": "hy4-flash", "mm3_quant": None})()
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.model, maya.HY4)
        b = type("A", (), {"model": maya.HY4})()
        self.assertEqual(maya.choose_family(b, {}), "hy4")

    def test_the_setups_picker(self):
        """The model picker: Hy4-flash's row last, one row a family (the picker's keys and its rows line up)."""
        rows = []

        def pick(title, intro, opts, default, yes, *x):
            rows.extend(opts)
            return len(opts) - 1
        a = type("A", (), {"model": None, "gguf_dir": None, "draft": None, "model_dir": None, "yes": True})()
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]):
            self.assertEqual(maya.choose_family(a, {}), "hy4")
        self.assertEqual(len(rows), len(maya.FAMILIES) + 1)   # (GLM-5.3 (744B): a row of the "glm" family)
        label, note, about = rows[-1]
        self.assertEqual(label, "Hy4-flash")
        self.assertIn("Q4_K_M", about)
        self.assertIn("text only", about)
        self.assertEqual(maya.FAMILY_TITLE["hy4"], "Hy4-flash")


class Setup(unittest.TestCase):
    def test_manifest_pinned(self):
        m = maya.hy4_entry()
        self.assertEqual(m["repo"], "AngelSlim/Hy4-flash-GGUF")
        self.assertEqual(len(m["revision"]), 40)
        self.assertEqual(sorted(m["files"]), list(SHARDS))
        for name, sha in m["sha256"].items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertGreater(m["files"][name], 50e9)
        self.assertAlmostEqual(m["download_gb"], 168.35, delta=0.1)

    def test_context_by_vram(self):
        """The cache sits on every GPU of a tensor-parallel group: the smallest GPU decides."""
        gpus = lambda *v: {"gpus": [{"vram_gb": x} for x in v]}   # noqa: E731
        self.assertEqual(maya.hy4_recommended_context(gpus(24, 16, 16)), 131072)   # (2.0 GB at 128K)
        self.assertEqual(maya.hy4_recommended_context(gpus(12)), 65536)
        self.assertEqual(maya.hy4_recommended_context(gpus(24, 6)), 32768)
        self.assertEqual(maya.hy4_recommended_context({"gpus": []}), 32768)
        self.assertIn(maya.HY4_DEFAULT_CONTEXT, maya.HY4_CONTEXTS)

    def test_config(self):
        """maya-hy4's own flags only (it stops at one it does not know), the model's sampling and level, its engine;
        a setup again keeps the engine settings set by hand."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            pack = t / "Hy4-flash" / "maya-hy4"
            (pack / "tokenizer").mkdir(parents=True)
            a = type("A", (), dict(port=None, env=[], host=None, api_key=None))()
            with patch.object(maya, "ROOT", t), patch.object(maya, "write_run_script", lambda *x: t / "run"), \
                    patch.object(maya, "ok"), patch.object(maya, "say"):
                p = maya.write_config_hy4(a, {"gpus": [{"index": 0}, {"index": 3}]}, {"lib_dirs": ["/cuda"]}, pack,
                                          131072, t, pack.parent, True)
                self.assertEqual(p.name, "maya-hy4-flash.json")
                c = json.loads(p.read_text())
                self.assertEqual(c["args"], ["--pack", str(pack), "--max-context", "131072", "--mtp"])
                self.assertEqual(Path(c["exe"]).name, "maya-hy4")
                self.assertEqual((c["sampling"], c["reasoning_effort"], c["model_name"], c["gpu"]),
                                 ({"temperature": 0.9, "top_p": 1.0}, "high", "hy4-flash", [0, 3]))
                self.assertEqual((c["installer"]["family"], c["installer"]["drafting"]), ("hy4", "on"))
                self.assertEqual(Q.family_of(c), "hy4")
                c["args"] += ["--chunk", "2048", "--reserve-gb", "0.6"]   # (set by hand: kept)
                c["port"], c["host"] = 8080, "0.0.0.0"
                p.write_text(json.dumps(c))
                p = maya.write_config_hy4(a, {"gpus": [{"index": 0}]}, {}, pack, 65536, t, pack.parent, False)
            c = json.loads(p.read_text())
            self.assertEqual(c["args"], ["--pack", str(pack), "--max-context", "65536", "--chunk", "2048",
                                         "--reserve-gb", "0.6"])
            self.assertEqual((c["host"], c["lib_dirs"], c["installer"]["drafting"]), ("0.0.0.0", ["/cuda"], "off"))

    def test_the_engine_target(self):
        """The launcher builds and starts maya-hy4 for it, and the GPU list knows the engine's process."""
        self.assertEqual(maya.exe_for_target("maya-hy4").name, "maya-hy4")
        self.assertIn("maya-hy4", maya.ENGINE_PROCESSES)
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(maya.bench_exe(hy4_cfg(Path(t))).name, "maya-hy4")
            cfg = hy4_cfg(Path(t))
            self.assertFalse(maya.needs_speed_calibration(cfg))
            self.assertFalse(maya.needs_prefill_calibration(cfg))

    def test_the_gpu_question(self):
        """Several GPUs: the question explains Hy4-flash's tensor parallel (not another model's split)."""
        seen = []

        def pick(title, intro, opts, default, yes, *x):
            seen.extend(intro)
            return default
        found = [{"index": 0, "arch": "86", "vram_gb": 24, "name": "RTX 3090"}] + \
            [{"index": i, "arch": "120", "vram_gb": 16, "name": "RTX 5060 Ti"} for i in (1, 2)]
        a = type("A", (), {"gpus": None, "gpu": None, "yes": True, "check": False})()
        with patch.object(maya, "choose", pick), patch.object(maya, "say"):
            got = maya.choose_gpus(a, found, maya.HY4_MIN_CC, "hy4")
        self.assertEqual([g["index"] for g in got], [0, 1, 2])
        self.assertTrue(seen[0].startswith("Hy4-flash runs best on all of them"), seen)

    def test_linux_nvidia_only(self):
        a = type("A", (), {"backend": "hip"})()
        with patch.object(maya, "step"), self.assertRaises(SystemExit):
            maya.check_pc(a, "hy4")


class Dashboard(unittest.TestCase):
    def test_models_page(self):
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = hy4_cfg(Path(t))
            self.assertEqual(M.choices(cfg), {"draft": ["mtp", "none"], "images": False, "text_only": True,
                                              "ablate": {"available": False, "why": "not available for this model"}})
            s = M.settings_of(cfg)
            self.assertEqual((s["draft"], s["tp"], s["reserve_gb"], s["prefill_chunk"]), ("mtp", "auto", None, None))
            self.assertEqual(M.trained_context_ds4(M.pack_of(cfg)), 1048576)
            self.assertEqual(M.pack_reads(M.pack_of(cfg)), [Path(t) / "Hy4-flash" / n for n in SHARDS])
            out = M.apply(cfg, {"draft": "none", "tp": 0, "reserve_gb": 0.6, "prefill_chunk": 2048, "context": 131072})
            self.assertEqual(out["args"], ["--pack", str(M.pack_of(cfg)), "--max-context", "131072", "--tp", "0",
                                           "--reserve-gb", "0.6", "--chunk", "2048"])
            back = M.apply(out, {"draft": "mtp", "tp": "auto", "reserve_gb": None, "prefill_chunk": None})
            self.assertEqual(back["args"], ["--pack", str(M.pack_of(cfg)), "--max-context", "131072", "--mtp"])
            for bad in ({"kv": "int8"}, {"parallel": 2}, {"images": True}, {"expert_cache": 100}):
                with self.assertRaises(ValueError, msg=bad):
                    M.apply(cfg, bad)
            with self.assertRaises(ValueError):
                M.apply(hy4_cfg(Path(t) / "b", n_mtp=0), {"draft": "mtp"})

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_web_apps_levels(self):
        """Hy4-flash thinks at High or not at all: the web app offers those two, High the default; a level saved on
        another model (Max, Low) becomes High."""
        # (state.js reads the browser's globals as it loads: stand-ins, as serve/test_glm_dsa_think_budget.py's)
        js = ("const noop = () => {}; const mem = {};"
              "globalThis.document = {addEventListener: noop, hidden: false, getElementById: () => null};"
              "globalThis.window = globalThis;"
              "globalThis.matchMedia = () => ({matches: false, addEventListener: noop, addListener: noop});"
              "globalThis.localStorage = {getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); },"
              " removeItem: (k) => { delete mem[k]; }};"
              "globalThis.location = {origin: 'http://x', href: 'http://x/', search: '', hash: ''};"
              "const st = await import('./serve/web/state.js');"   # (THINKING: a live binding - read after setFamily)
              "const {modelName} = await import('./serve/web/lib.js');"
              "const f = st.setFamily('hy4-flash');"
              "console.log(JSON.stringify([f, st.THINKING, st.DEFAULTS.thinking, st.levelFor('max'),"
              " st.levelFor('low') || st.DEFAULTS.thinking, modelName('hy4-flash')]));"
              "process.exit(0);")
        out = subprocess.run(["node", "--input-type=module", "-e", js], cwd=str(ROOT), capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), ["hy4", ["none", "high"], "high", "high", "high", "Hy4 Flash"])


if __name__ == "__main__":
    unittest.main()
