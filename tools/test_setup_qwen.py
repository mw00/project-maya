"""tools/test_setup_qwen.py - CYBER-FROST-3.8-PS in the setup and the dashboard: its name everywhere
("Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M"; its ids stay), the setup ids, its configs recognised (not taken for DeepSeek's --pack), the pinned download manifest, the
context recommended by the VRAM, the drafting choice written into the config, the Models page's settings, and its
template's thinking (always on; thinking off closes the block as its Qwen base writes it).  No GPU, no network.

    python -m unittest tools.test_setup_qwen -v
"""
from pathlib import Path
import json
import os
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

TEMPLATE = ("{%- set enable_thinking = true %}{%- for m in messages %}"
            "{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
            "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}")


def qwen_cfg(d: Path, mtp: bool = True) -> dict:
    pack = d / "CYBER-FROST-3.8-PS" / "pack"
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    if mtp:
        (pack / "mtp").mkdir()
        (pack / "mtp" / "dense.txt").write_text("x")
    g = str(pack.parent / "CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf")
    return {"exe": "/b/maya", "args": ["--pack", str(pack), "--native", g, "--ple-gguf", g, "--spec", "4", "--prefill",
                                       "auto", "--expert-cache", "auto", "--max-context", "65536"],
            "model_name": "cyber-frost-3.8-ps", "gpu": [0], "tokenizer": str(pack / "tokenizer"),
            "installer": {"family": "qwen", "quant": "CYBER-FROST-3.8-PS", "qwen_quant": "q5_k_m"}}


class Names(unittest.TestCase):
    def test_shown(self):
        self.assertEqual(Q.shown("qwen", "q5_k_m", 112.53), "Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M (113 GB)")
        self.assertEqual(Q.config_stem("qwen", "q5_k_m"), "maya-cyber-frost")

    def test_ids(self):
        for name in ("cyber-frost", "CYBER-FROST-3.8-PS", "cyber-frost-3.8-ps-q5_k_m", "cyber-frost-3.8",
                     "cyber-frost-q5_k_m"):   # (every id before the rename is still taken)
            self.assertEqual(Q.resolve(name), ("qwen", "q5_k_m"), name)

    def test_the_ids_stay_the_name_on_screen_changes(self):
        """The rename is the display name only: the folder, the installer record, the model_name the API reports, the
        HF repo, the config's file name and --model keep their ids."""
        self.assertEqual(Q.SHOWN["qwen"], "Qwen-3.8-Flash-Next (CYBER-FROST)")
        self.assertEqual((maya.QWEN, maya.QWEN_MODEL_NAME, Q.MODEL["qwen"]),
                         ("CYBER-FROST-3.8-PS", "cyber-frost-3.8-ps", "CYBER-FROST-3.8-PS"))
        self.assertEqual(maya.QWEN_SHOWN, Q.SHOWN["qwen"])
        self.assertEqual(maya.qwen_entry()["repo"], "peasantsmith/CYBER-FROST-3.8-PS-GUFF")
        self.assertEqual(Q.config_stem("qwen", "q5_k_m"), "maya-cyber-frost")

    def test_the_setups_pickers(self):
        """The model picker's row, and the step lines of its setup, name it as the screen does; DeepSeek's row no
        longer offers UD-Q4_K_XL."""
        rows = []

        def pick(title, intro, opts, default, yes, *x):
            rows.extend(opts)
            return default
        a = type("A", (), {"model": None, "gguf_dir": None, "draft": None, "model_dir": None, "yes": True})()
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]):
            maya.choose_family(a, {})
        (label, note, about) = rows[3]
        self.assertEqual(label, "Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M (113 GB)")
        self.assertTrue(about.startswith("Qwen-3.8-Flash-Next (CYBER-FROST): 180B"), about)
        self.assertNotIn("CYBER-FROST-3.8-PS", about)
        self.assertIn("UD-Q8_K_XL", rows[1][2])
        self.assertNotIn("UD-Q4_K_XL", rows[1][2])

    def test_family(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = qwen_cfg(Path(t))
            self.assertEqual(Q.family_of(cfg), "qwen")
            cfg.pop("installer")                       # (by its arguments too: --pack with --ple-gguf is not DeepSeek's)
            self.assertEqual(Q.family_of(cfg), "qwen")
            self.assertEqual(Q.config_label(qwen_cfg(Path(t) / "b")), "Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M")

    def test_model_option(self):
        a = type("A", (), {"model": "cyber-frost", "mm3_quant": None})()
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.model, maya.QWEN)


class Setup(unittest.TestCase):
    def test_manifest_pinned(self):
        m = maya.qwen_entry()
        self.assertEqual(m["repo"], "peasantsmith/CYBER-FROST-3.8-PS-GUFF")
        self.assertEqual(len(m["revision"]), 40)
        for name, sha in m["sha256"].items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertGreater(m["files"][name], 100e9)
        self.assertTrue((ROOT / "LICENSE.qwen-community-1.0").is_file())

    def test_context_by_vram(self):
        self.assertEqual(maya.qwen_recommended_context(32), 262144)   # (INT8 K / V: 3.5 GB at 256K)
        self.assertEqual(maya.qwen_recommended_context(16), 131072)
        self.assertEqual(maya.qwen_recommended_context(8), 65536)
        self.assertEqual(maya.qwen_recommended_context(4), 32768)
        # (the experts do not all fit in VRAM and RAM: the VRAM goes to expert slots, the context shrinks)
        self.assertEqual(maya.qwen_recommended_context(32, True), 131072)
        self.assertEqual(maya.qwen_recommended_context(96, True), 262144)
        self.assertEqual(maya.qwen_recommended_context(8, True), 32768)
        self.assertIn(maya.QWEN_DEFAULT_CONTEXT, maya.QWEN_CONTEXTS)

    def test_drafting_chosen(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "maya-cyber-frost.json"
            p.write_text(json.dumps(qwen_cfg(Path(t))))
            with patch.object(maya, "ok"), patch.object(maya, "say"):
                maya.keep_drafting(p, True)
                args = json.loads(p.read_text())["args"]
                self.assertEqual(Path(args[args.index("--mtp") + 1]).name, "mtp")
                self.assertNotIn("--mtp-max-t", args)
                maya.keep_drafting(p, False)
            cfg = json.loads(p.read_text())
            self.assertIn("--mtp", cfg["args"])            # (the server always loads the drafter)
            self.assertEqual(cfg["args"][cfg["args"].index("--mtp-max-t") + 1], "1")
            self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "none")
            self.assertFalse(maya.needs_speed_calibration(cfg))


class Dashboard(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_web_apps_names(self):
        """serve/web/lib.js's modelName: the full name "Qwen-3.8-Flash-Next (CYBER-FROST)" wherever there is room, the
        short "Qwen-3.8-Flash-Next" in the chat header's button; the other models' names as before."""
        js = ("import {modelName} from './serve/web/lib.js';"
              "const ids = ['cyber-frost-3.8-ps', 'glm-5.3-flash', 'deepseek-v4.1-flash', 'minimax-m3.1-flash'];"
              "console.log(JSON.stringify(ids.map((i) => [modelName(i), modelName(i, true)])));")
        out = subprocess.run(["node", "--input-type=module", "-e", js], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), [
            ["Qwen-3.8-Flash-Next (CYBER-FROST)", "Qwen-3.8-Flash-Next"], ["GLM 5.3 Flash", "GLM 5.3 Flash"],
            ["DeepSeek V4.1 Flash", "DeepSeek V4.1 Flash"], ["MiniMax M3.1 Flash", "MiniMax M3.1 Flash"]])
        head = (ROOT / "serve/web/chat-view.js").read_text(encoding="utf-8")
        self.assertIn("const name = modelName(S.health.model, true);", head)   # (the header's button: the short form)
        # (the model menu's card: the short form above the line that has the full label "... (CYBER-FROST) · Q5_K_M")
        self.assertIn("<b>${modelName(h.model, !!h.model_label)}</b>", head)

    def test_models_page(self):
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = qwen_cfg(Path(t))
            self.assertEqual(M.choices(cfg), {"draft": ["mtp", "none"], "images": False,   # (Qwen: no ablation)
                                              "ablate": {"available": False, "why": "not available for this model"}})
            s = M.settings_of(cfg)
            self.assertEqual((s["draft"], s["kv"], s["expert_cache"]), ("none", "fp16", None))
            out = M.apply(cfg, {"draft": "mtp", "kv": "int8", "ram_gb": 40, "context": 131072})
            self.assertEqual(out["args"][out["args"].index("--kv") + 1], "int8")
            self.assertTrue(out["args"][out["args"].index("--mtp") + 1].endswith("mtp"))
            self.assertNotIn("--mtp-max-t", out["args"])
            off = M.apply(out, {"draft": "none"})
            self.assertIn("--mtp", off["args"])
            self.assertEqual(off["args"][off["args"].index("--mtp-max-t") + 1], "1")
            self.assertEqual(M.settings_of(off)["draft"], "none")
            self.assertEqual(out["env"], {"MAYA_EXPERT_RAM_GB": "40"})
            with self.assertRaises(ValueError):             # (a config set up without the encoder)
                M.apply(cfg, {"images": True})
            with self.assertRaises(ValueError):
                M.apply(qwen_cfg(Path(t) / "b", mtp=False), {"draft": "mtp"})

    def test_models_page_images(self):
        """Images on a config with the setup's encoder: on, off (kept as vision_off, the engine's --vision dropped) and
        on again."""
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = qwen_cfg(Path(t))
            cfg["args"].append("--vision")
            cfg["vision"] = {"exe": "/b/maya-vision", "mmproj": "/m/vision/mmproj-F16.gguf", "model": "/m/x.gguf",
                             "gpu": True}
            self.assertEqual(M.choices(cfg)["images"], True)
            self.assertTrue(M.settings_of(cfg)["images"])
            off = M.apply(cfg, {"images": False})
            self.assertNotIn("--vision", off["args"])
            self.assertEqual((off.get("vision"), off["vision_off"]), (None, cfg["vision"]))
            self.assertFalse(M.settings_of(off)["images"])
            self.assertTrue(M.choices(off)["images"])          # (it can be switched on again)
            on = M.apply(off, {"images": True})
            self.assertEqual((on["vision"], on.get("vision_off")), (cfg["vision"], None))
            self.assertEqual(on["args"].count("--vision"), 1)

    def test_thinking_always_on_and_off_form(self):
        from serve.frontend import ChatTemplate
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "chat_template.jinja"
            p.write_text(TEMPLATE, encoding="utf-8")
            tpl = ChatTemplate(p)
            self.assertTrue(tpl.forced_thinking)
            msgs = [{"role": "user", "content": "hi"}]
            self.assertTrue(tpl.render(msgs, reasoning_effort="max").endswith("<|im_start|>assistant\n<think>\n"))
            self.assertTrue(tpl.render(msgs, enable_thinking=False).endswith("<think>\n\n</think>\n\n"))


class Vision(unittest.TestCase):
    """Pictures: the base model's vision tower (the fine-tune kept it byte for byte) as a pinned mmproj, on by default."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.patches = [patch.object(maya, n, lambda *a, **k: None) for n in ("say", "ok", "warn", "step")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def a(self, **kw):
        return type("A", (), dict(dict(no_vision=False, rebuild=False, download_model=False, yes=False), **kw))()

    def test_manifest_pinned(self):
        m = maya.qwen_vision_model()
        self.assertEqual((m["repo"], len(m["revision"]), list(m["files"])),
                         ("unsloth/Qwen3.8-Flash-Next-GGUF", 40, ["mmproj-F16.gguf"]))
        self.assertRegex(m["sha256"]["mmproj-F16.gguf"], r"^[0-9a-f]{64}$")
        self.assertEqual(maya.file_url(m, "mmproj-F16.gguf"), "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/"
                         "resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/mmproj-F16.gguf")

    def test_no_vision(self):
        self.assertIsNone(maya.qwen_vision_step(self.a(no_vision=True), {}, {}, self.d, self.d, self.d / "m.gguf"))

    def test_download_declined(self):
        with patch.object(maya, "offer_download", lambda *a, **k: False):
            self.assertIsNone(maya.qwen_vision_step(self.a(), {}, {}, self.d, self.d, self.d / "m.gguf"))

    def test_on_by_default(self):
        mm = self.d / "vision" / "mmproj-F16.gguf"
        with patch.object(maya, "file_incomplete", lambda *a: None), patch.object(maya, "sha256_checked", lambda *a: True), \
                patch.object(maya, "vision_encoder_ready", lambda *a: True):
            v = maya.qwen_vision_step(self.a(), {}, {}, self.d, self.d, self.d / "m.gguf")
        self.assertEqual(v, {"exe": str(maya.VEXE), "mmproj": str(mm), "model": str(self.d / "m.gguf"), "gpu": True})
        with patch.object(maya, "file_incomplete", lambda *a: None), patch.object(maya, "sha256_checked", lambda *a: True), \
                patch.object(maya, "vision_encoder_ready", lambda *a: False):   # (the encoder did not compile: said)
            self.assertIsNone(maya.qwen_vision_step(self.a(), {}, {}, self.d, self.d, self.d / "m.gguf"))

    def test_config(self):
        """The config: "vision" and the engine's --vision with the encoder, neither without; a setup again writes them
        from this setup's step 7 (an earlier text-only config turns pictures on)."""
        pack = self.d / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        v = {"exe": "/b/maya-vision", "mmproj": "/m/vision/mmproj-F16.gguf", "model": "/m/x.gguf", "gpu": True}
        a = type("A", (), dict(port=None, env=[], host=None, api_key=None))()
        with patch.object(maya, "ROOT", self.d), patch.object(maya, "write_run_script", lambda *x: self.d / "run"):
            old = self.d / (maya.Q.config_stem("qwen", "q5_k_m") + ".json")
            old.write_text(json.dumps({"args": ["--pack", str(pack)], "vision_off": v}))
            for vision in (v, None):
                p = maya.write_config_qwen(a, {"gpus": [{"index": 0}]}, {}, pack, self.d / "x.gguf", 65536,
                                           self.d, self.d, vision)
                c = json.loads(p.read_text())
                self.assertEqual((c.get("vision"), "--vision" in c["args"]), (vision, vision is not None))


UD = "ud-iq2_xs"
UD_SHARDS = [f"Qwen3.8-Flash-Next-UD-IQ2_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
UD_MTP = "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
UD_REV = "d96201194f103c906994f283f504209dc8d2a10c"


class UnslothUD(unittest.TestCase):
    """unsloth's Qwen3.8-Flash-Next UD-IQ2_XS: its names and ids, the pinned manifest (shards 1-2 and the MTP head
    from unsloth, the n-gram shard from Maya's repo), its config's defaults, and one copy of a shared file."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.patches = [patch.object(maya, n, lambda *a, **k: None) for n in ("say", "ok", "warn", "step")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_names_and_ids(self):
        self.assertEqual(Q.shown("qwen", UD, 79.97), "Qwen-3.8-Flash-Next · unsloth UD-IQ2_XS (80 GB)")
        self.assertEqual(Q.shown("qwen", "q5_k_m"), "Qwen-3.8-Flash-Next (CYBER-FROST) · Q5_K_M")   # (as before)
        self.assertEqual(Q.config_stem("qwen", UD), "maya-qwen-ud-iq2_xs")
        self.assertEqual(Q.config_stem("qwen", "q5_k_m"), "maya-cyber-frost")
        for name in ("qwen-ud-iq2_xs", "Qwen3.8-Flash-Next-UD-IQ2_XS", "qwen3.8-ud-iq2_xs"):
            self.assertEqual(Q.resolve(name), ("qwen", UD), name)
        self.assertEqual(Q.resolve("cyber-frost"), ("qwen", "q5_k_m"))
        self.assertEqual(maya.QWEN_QUANTS, ("q5_k_m", UD))

    def test_model_option(self):
        a = type("A", (), {"model": "qwen-ud-iq2_xs", "mm3_quant": None, "qwen_quant": None})()
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.qwen_quant), (maya.QWEN, UD))
        a = type("A", (), {"model": "cyber-frost", "mm3_quant": None, "qwen_quant": None})()
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.qwen_quant, "q5_k_m")
        a = type("A", (), {"model": "qwen-ud-iq2_xs", "mm3_quant": None, "qwen_quant": "q5_k_m"})()
        self.assertIn("--qwen-quant", maya.model_option(a))

    def test_manifest_pinned(self):
        m = maya.qwen_entry(UD)
        self.assertEqual((m["repo"], m["revision"], m["folder"]), ("unsloth/Qwen3.8-Flash-Next-GGUF", UD_REV, "UD-IQ2_XS"))
        self.assertEqual(list(m["files"]), UD_SHARDS + [UD_MTP])
        for name, sha in m["sha256"].items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
        self.assertEqual((m["ngram"], m["mtp"]), (UD_SHARDS[2], UD_MTP))
        self.assertEqual(m["sha256"][UD_SHARDS[2]], "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6")
        self.assertEqual(m["files"][UD_SHARDS[2]], 28800138432)
        self.assertAlmostEqual(m["download_gb"], 79.97, delta=0.01)
        u = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/" + UD_REV
        self.assertEqual(maya.file_url(m, UD_SHARDS[0]), f"{u}/UD-IQ2_XS/{UD_SHARDS[0]}")
        self.assertEqual(maya.file_url(m, UD_SHARDS[1]), f"{u}/UD-IQ2_XS/{UD_SHARDS[1]}")
        self.assertEqual(maya.file_url(m, UD_MTP), f"{u}/{UD_MTP}")   # (the MTP head: the repo's top)
        # the n-gram table: Maya's own repo (its downloads counted there), byte for byte unsloth's shard
        self.assertEqual(maya.file_url(m, UD_SHARDS[2]),
                         "https://huggingface.co/peasantsmith/CYBER-FROST-3.8-PS-GUFF/resolve/"
                         f"b8069f8e31925001c5484cbb256daa3fdc0a3ca4/Qwen3.8-Flash-Next-ngram/{UD_SHARDS[2]}")
        self.assertEqual(maya.qwen_entry()["repo"], "peasantsmith/CYBER-FROST-3.8-PS-GUFF")   # (CYBER-FROST's: as before)
        self.assertNotIn("sources", maya.qwen_entry())

    def test_model_step_downloads_into_its_folder(self):
        seen = {}

        def offer(a, quant, d, shards, m):
            seen.update(quant=quant, d=d, shards=[s.name for s in shards], m=m)
            return False
        a = type("A", (), {"model_dir": None})()
        d = maya.qwen_dir(a, self.d, {"args": ["--native", str(self.d / "CYBER-FROST-3.8-PS" / "x.gguf")],
                                      "installer": {"qwen_quant": "q5_k_m"}}, UD)
        self.assertEqual(d, self.d / "Qwen3.8-Flash-Next-UD-IQ2_XS")   # (CYBER-FROST's folder is not this one's)
        with patch.object(maya, "offer_download", offer), patch.object(maya, "configs", lambda: []):
            self.assertIsNone(maya.qwen_model_step(a, d, UD, self.d))
        self.assertEqual((seen["d"], seen["shards"]), (d, UD_SHARDS + [UD_MTP]))
        self.assertEqual(seen["quant"], "Qwen-3.8-Flash-Next · unsloth UD-IQ2_XS (80 GB)")

    def test_config_defaults(self):
        """The model's own defaults in its config: the card's thinking-mode sampling (min_p 0 too), xhigh, its chat
        template from the pack's tokenizer, MTP loaded, INT8 KV, the n-gram shard memory-mapped; the experts' place by
        --expert-cache auto (this PC's VRAM / RAM)."""
        d = self.d / "Qwen3.8-Flash-Next-UD-IQ2_XS"
        pack = d / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "mtp").mkdir()
        (pack / "mtp" / "dense.txt").write_text("x")
        a = type("A", (), dict(port=None, env=[], host=None, api_key=None))()
        with patch.object(maya, "ROOT", self.d), patch.object(maya, "write_run_script", lambda *x: self.d / "run"):
            p = maya.write_config_qwen(a, {"gpus": [{"index": 0}, {"index": 1}]}, {}, pack, d / UD_SHARDS[0], 65536,
                                       self.d, d, None, UD, d / UD_SHARDS[2])
        self.assertEqual(p.name, "maya-qwen-ud-iq2_xs.json")
        c = json.loads(p.read_text())
        args = c["args"]
        val = lambda k: args[args.index(k) + 1]   # noqa: E731
        self.assertEqual((val("--native"), val("--ple-gguf")), (str(d / UD_SHARDS[0]), str(d / UD_SHARDS[2])))
        self.assertEqual((val("--ple-io"), val("--kv"), val("--expert-cache"), val("--mtp")), ("mmap", "int8", "auto",
                                                                                              str(pack / "mtp")))
        self.assertEqual(c["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
        self.assertEqual((c["reasoning_effort"], c["model_name"], c["tokenizer"]),
                         ("xhigh", "qwen3.8-flash-next", str(pack / "tokenizer")))
        self.assertEqual((c["installer"]["qwen_quant"], c["installer"]["quant"]), (UD, "Qwen3.8-Flash-Next-UD-IQ2_XS"))
        self.assertEqual(Q.family_of(c), "qwen")
        self.assertEqual(Q.config_label(c), "Qwen-3.8-Flash-Next · unsloth UD-IQ2_XS")
        self.assertEqual(Q.model_defaults(c), ({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}, "xhigh"))
        # CYBER-FROST's config: its defaults as before (no min_p, the GGUF itself for the table, no --ple-io)
        with patch.object(maya, "ROOT", self.d), patch.object(maya, "write_run_script", lambda *x: self.d / "run"):
            p = maya.write_config_qwen(a, {"gpus": [{"index": 0}]}, {}, pack, self.d / "cf.gguf", 65536, self.d,
                                       self.d)
        c = json.loads(p.read_text())
        self.assertEqual(p.name, "maya-cyber-frost.json")
        self.assertEqual((c["sampling"], c["reasoning_effort"], c["model_name"]),
                         ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "xhigh", "cyber-frost-3.8-ps"))
        self.assertNotIn("--ple-io", c["args"])
        self.assertEqual(c["args"][c["args"].index("--ple-gguf") + 1], str(self.d / "cf.gguf"))

    def test_choose_quant(self):
        a = type("A", (), {"qwen_quant": None, "yes": True})()
        with patch.object(maya, "choose", lambda t, i, rows, default, yes: default):
            self.assertEqual(maya.choose_qwen_quant(a, {}), "q5_k_m")
            self.assertEqual(maya.choose_qwen_quant(a, {"installer": {"qwen_quant": UD}}), UD)
        a.qwen_quant = UD
        self.assertEqual(maya.choose_qwen_quant(a, {}), UD)

    def test_shared_file_one_copy(self):
        """A file another Qwen setup has (the mmproj: CYBER-FROST's and the base model's are the same bytes) is
        hard-linked by its size and sha256 - never by its name alone - with its sha256 record."""
        import hashlib
        body = b"mmproj" * 1000
        sha = hashlib.sha256(body).hexdigest()
        src = self.d / "CYBER-FROST-3.8-PS" / "vision" / "mmproj-F16.gguf"
        src.parent.mkdir(parents=True)
        src.write_bytes(body)
        dst = self.d / "Qwen3.8-Flash-Next-UD-IQ2_XS" / "vision" / "mmproj-F16.gguf"
        with patch.object(maya, "configs", lambda: []):
            places = maya.qwen_places(self.d, "vision/mmproj-F16.gguf")
            self.assertIsNone(maya.link_published(dst, len(body), "0" * 64, places))   # (another file: not linked)
            self.assertFalse(dst.exists())
            self.assertEqual(maya.link_published(dst, len(body), sha, places), src)
        self.assertTrue(dst.exists())
        self.assertTrue(os.path.samefile(src, dst))
        self.assertIn("mmproj-F16.gguf", json.loads((dst.parent / maya.VERIFIED).read_text()))


if __name__ == "__main__":
    unittest.main()
