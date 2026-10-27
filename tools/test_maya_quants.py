"""tools/test_maya_quants.py - the quants' names (maya_quants.py): the model first, the tier by its experts' bits -
the setup ids and every name used before as aliases (--model, --ds4-quant, --mm3-quant), the names on screen, configs
written by earlier releases still loaded and shown, MiniMax's GGUFs under the scheme (files, repo folders, config) and
DeepSeek's context recommended by the PC's memory.  No GPU, no network.

    python -m unittest tools.test_maya_quants -v
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402

# old name -> (family, the setup's key): every --model / --ds4-quant / --mm3-quant value of v1.1.29 and before
ALIASES = {
    "Maya-S-v2-IQ2_XXS": ("glm", "Maya-S-v2-IQ2_XXS"), "Maya-S24": ("glm", "Maya-S24"), "Maya-M": ("glm", "Maya-M"),
    "Maya-M-Derisked": ("glm", "Maya-M-Derisked"), "Maya-L": ("glm", "Maya-L"),
    "EXL3-3.05bpw": ("glm", "EXL3-3.05bpw"), "EXL3-4.05bpw": ("glm", "EXL3-4.05bpw"),
    "EXL3-sc_4.10bpw": ("glm", "EXL3-sc_4.10bpw"),
    "DeepSeek-V4.1-Flash-1017": ("deepseek", None), "MiniMax-M3.1-Flash": ("minimax", None),
}
# the setup ids (new) -> (family, key, what the picker and the dashboard show)
IDS = {
    "glm-maya-s": ("glm", "Maya-S-v2-IQ2_XXS", "GLM-5.3 · Maya-S"),
    "glm-maya-s24": ("glm", "Maya-S24", "GLM-5.3 · Maya-S24"),
    "glm-maya-m": ("glm", "Maya-M", "GLM-5.3 · Maya-M"),
    "glm-maya-m-derisked": ("glm", "Maya-M-Derisked", "GLM-5.3 · Maya-M-Derisked"),
    "glm-maya-l": ("glm", "Maya-L", "GLM-5.3 · Maya-L"),
    "deepseek-maya-xs": ("deepseek", "maya-xs", "DeepSeek V4.1 · Maya-XS"),
    "deepseek-maya-s": ("deepseek", "maya-s", "DeepSeek V4.1 · Maya-S"),
    "deepseek-maya-m": ("deepseek", "maya-m", "DeepSeek V4.1 · Maya-M"),
    "minimax-maya-xs": ("minimax", "gguf-xs", "MiniMax M3.1 · Maya-XS"),
    "minimax-maya-s": ("minimax", "gguf-s", "MiniMax M3.1 · Maya-S"),
}


def args(**kw):
    base = dict(model=None, ds4_quant=None, mm3_quant=None, model_dir=None, gguf_dir=None, draft=None, backend="cuda",
                yes=True, check=False, port=None, host=None, api_key=None, env=[], no_vision=False)
    base.update(kw)
    return SimpleNamespace(**base)


class Aliases(unittest.TestCase):
    def test_setup_ids(self):
        for i, (family, key, shown) in IDS.items():
            self.assertEqual(Q.resolve(i), (family, key), i)
            self.assertEqual(Q.resolve(i.upper()), (family, key), i)          # (any case)
            self.assertEqual(Q.setup_id(family, key), i)
            self.assertEqual(Q.shown(family, key), shown)
        self.assertEqual(sorted(IDS), sorted(i for _, _, _, i in Q.QUANTS))

    def test_full_names_model_first(self):
        self.assertEqual(Q.full_name("minimax", "gguf-xs"), "MiniMax-M3.1-Flash-Maya-XS")
        self.assertEqual(Q.full_name("deepseek", "maya-xs"), "DeepSeek-V4.1-Flash-1017-Maya-XS")
        self.assertEqual(Q.full_name("glm", "Maya-S-v2-IQ2_XXS"), "GLM-5.3-Flash-Maya-S")
        for family, key, _ in IDS.values():
            self.assertEqual(Q.resolve(Q.full_name(family, key)), (family, key))
        self.assertEqual(maya.MM3_QUANTS["gguf-xs"], Q.full_name("minimax", "gguf-xs"))   # (its folder on this PC)

    def test_old_names_still_work(self):
        for old, hit in ALIASES.items():
            self.assertEqual(Q.resolve(old, list(maya.MODELS)), hit, old)
        self.assertEqual(set(maya.MODELS) - {k for k in ALIASES}, set())     # (every GLM download kept)
        self.assertIsNone(Q.resolve("Maya-XS", list(maya.MODELS)))           # (a bare tier names no model)
        self.assertIsNone(Q.resolve("glm-maya-zz", list(maya.MODELS)))

    def test_shown_never_a_bare_tier(self):
        for family, key, _ in IDS.values():
            self.assertTrue(Q.shown(family, key).startswith(Q.SHOWN[family] + " · "))
        self.assertEqual(Q.shown("minimax", "gguf-xs", 92.98), "MiniMax M3.1 · Maya-XS (93 GB)")
        self.assertEqual(Q.shown("deepseek", "gguf-q8"), "DeepSeek V4.1 · UD-Q8_K_XL")
        self.assertEqual(Q.shown("minimax", "exl3"), "MiniMax M3.1 · EXL3 3.5 bpw")
        self.assertEqual(Q.shown("glm", "EXL3-3.05bpw"), "GLM-5.3 · EXL3-3.05bpw")
        self.assertEqual(Q.shown("glm", "UD-Q2_K_XL"), "GLM-5.3 · UD-Q2_K_XL")   # (a GGUF of the user's: its name)
        self.assertEqual(Q.shown("deepseek"), "DeepSeek V4.1")


class ModelOption(unittest.TestCase):
    """--model as typed -> what the setup reads: the old values as before, the new ids to the same."""

    def run_opt(self, **kw):
        a = args(**kw)
        return maya.model_option(a), a

    def test_new_ids(self):
        why, a = self.run_opt(model="minimax-maya-xs")
        self.assertEqual((why, a.model, a.mm3_quant), (None, maya.MM3, "gguf-xs"))
        why, a = self.run_opt(model="deepseek-maya-xs")
        self.assertEqual((why, a.model, a.ds4_quant), (None, maya.DS4, "maya-xs"))
        why, a = self.run_opt(model="GLM-Maya-S")
        self.assertEqual((why, a.model), (None, "Maya-S-v2-IQ2_XXS"))
        self.assertEqual(maya.choose_family(a, {}), "glm")
        self.assertEqual(maya.choose_model(a, Path("/m"), {}), ("download", "Maya-S-v2-IQ2_XXS"))

    def test_old_values_unchanged(self):
        for old in ("Maya-S24", "Maya-L", "EXL3-sc_4.10bpw"):
            why, a = self.run_opt(model=old)
            self.assertEqual((why, a.model), (None, old))
        why, a = self.run_opt(model=maya.DS4, ds4_quant="maya-xs")
        self.assertEqual((why, a.model, a.ds4_quant), (None, maya.DS4, "maya-xs"))
        why, a = self.run_opt(model=maya.MM3, mm3_quant="gguf-xs")
        self.assertEqual((why, a.model, a.mm3_quant), (None, maya.MM3, "gguf-xs"))
        why, a = self.run_opt(model=maya.MM3, mm3_quant="maya-s")              # (the tier, as --ds4-quant takes it)
        self.assertEqual((why, a.mm3_quant), (None, "gguf-s"))
        why, a = self.run_opt()
        self.assertEqual((why, a.model), (None, None))

    def test_errors(self):
        why, _ = self.run_opt(model="Maya-XS")
        self.assertIn("not a model Maya knows", why)
        self.assertIn("minimax-maya-xs", why)
        why, _ = self.run_opt(model="minimax-maya-xs", mm3_quant="gguf-s")
        self.assertIn("give one of them", why)
        why, a = self.run_opt(model="minimax-maya-xs", mm3_quant="gguf-xs")    # (the same quant twice: fine)
        self.assertIsNone(why)

    def test_command_line(self):
        """The parser takes any name (no fixed choices) and refuses an unknown one with the list."""
        r = subprocess.run([sys.executable, str(Path(maya.__file__)), "--model", "glm-maya-zz", "--check"],
                           capture_output=True, text=True, timeout=120, env=dict(maya.os.environ, MAYA_SH="1"))
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("not a model Maya knows", r.stderr)


def v1129_configs(root: Path) -> dict:
    """Configs as v1.1.29 wrote them (DeepSeek's without its family in the installer record, as before v1.1.24)."""
    cfgs = {
        "maya-maya-s24.json": {"exe": str(root / "build/maya"), "args": ["--glm-pack", "/m/Maya-S24/pack",
                                                                           "--max-context", "65536"],
                               "model_name": "glm-5.3-flash", "env": {"MAYA_GLM_CPU_LANE": "8"},
                               "installer": {"models_dir": "/m", "quant": "Maya-S24", "draft": "mtp"}},
        "maya-ds4.json": {"exe": str(root / "build/maya-ds4"), "args": ["--pack", "/m/D/maya-ds4", "--max-context",
                                                                        "65536", "--tp", "2"],
                          "model_name": "deepseek-v4.1-flash",
                          "installer": {"models_dir": "/m", "quant": maya.DS4, "ds4_quant": "maya-xs"}},
        "maya-mm3.json": {"exe": str(root / "build/maya-mm3"), "args": ["--pack", "/m/E/maya-mm3", "--max-context",
                                                                        "131072", "--kv-cache", "int8"],
                          "model_name": "minimax-m3.1-flash", "host": "0.0.0.0",
                          "installer": {"family": "minimax", "models_dir": "/m", "quant": maya.MM3,
                                        "mm3_quant": "exl3"}},
    }
    for name, c in cfgs.items():
        (root / name).write_text(json.dumps(c))
    return cfgs


class OldConfigs(unittest.TestCase):
    """Configs, installs and command lines from before the naming scheme keep working: each config loads with its
    model, its files and its settings, is shown model first, and is the next setup's default."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cfgs = v1129_configs(self.root)
        self.patch = patch.object(maya, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_loaded_and_shown(self):
        names = sorted(p.name for p in maya.configs())
        self.assertEqual(names, sorted(self.cfgs))
        want = {"maya-maya-s24.json": ("glm", "GLM-5.3 · Maya-S24"),
                "maya-ds4.json": ("deepseek", "DeepSeek V4.1 · Maya-XS"),
                "maya-mm3.json": ("minimax", "MiniMax M3.1 · EXL3 3.5 bpw")}
        for name, (family, label) in want.items():
            cfg = maya.read_json(self.root / name)
            self.assertEqual(maya.config_family(cfg), family, name)
            self.assertEqual(Q.config_label(cfg), label, name)
        self.assertEqual(list(maya.read_json(self.root / "maya-maya-s24.json")["env"].values()), ["8"])   # (kept)
        # (a DeepSeek config from before its quants were recorded: the EXL3 quant's; a hand-written one: no label)
        self.assertEqual(Q.config_label({"exe": "/b/maya-ds4", "args": ["--pack", "/p"], "installer": {}}),
                         "DeepSeek V4.1 · EXL3 2.9 bpw")
        self.assertIsNone(Q.config_label({"exe": "/b/maya", "args": [], "model_name": "qwen3.8-flash-next"}))

    def test_the_next_setups_default(self):
        """A setup again offers what the config before has, under its new name."""
        glm = maya.read_json(self.root / "maya-maya-s24.json")
        rows = []

        def pick(title, intro, opts, default, yes, *x):
            rows.extend(o[0] for o in opts)
            return default
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]):
            self.assertEqual(maya.choose_model(args(), self.root, glm["installer"]), ("download", "Maya-S24"))
        self.assertTrue(any(r.startswith("GLM-5.3 · Maya-S24: download") for r in rows), rows)
        self.assertFalse(any(r.startswith("Maya-") for r in rows), rows)       # (never a bare tier)
        ds4 = maya.read_json(self.root / "maya-ds4.json")
        rows.clear()
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]):
            self.assertEqual(maya.choose_ds4_quant(args(), ds4), "maya-xs")
        self.assertTrue(any(r.startswith("DeepSeek V4.1 · Maya-XS (") for r in rows), rows)

    def test_the_screens_card(self):
        """The setup's screen shows what runs model first (server_command's "label"); without one as before."""
        try:
            from setup_look import serving_card
        except ImportError:   # (Textual / rich: from Maya's own wheels, in .venv)
            self.skipTest("the setup screen's libraries are not installed here")
        s = {"model": "glm-5.3-flash", "quant": "Maya-S24", "context": "65536", "dashboard": "http://127.0.0.1:9/",
             "api": "http://127.0.0.1:9/v1", "state": "ready", "since": 0, "speed": None}
        label = Q.config_label(maya.read_json(self.root / "maya-maya-s24.json"))
        self.assertIn("GLM-5.3 · Maya-S24 · 64K context", serving_card(dict(s, label=label), "").plain)
        self.assertIn("glm-5.3-flash · Maya-S24 · 64K context", serving_card(s, "").plain)
        self.assertIn('"label": Q.config_label(cfg)', Path(maya.__file__).read_text(encoding="utf-8"))

    def test_minimax_gguf_config_named_by_the_scheme(self):
        """A GGUF quant's config is maya-minimax-maya-xs.json; it takes the EXL3 config's settings the first time and
        leaves that config as it is; a GGUF config of this quant under the name before is moved to the new name."""
        pack = self.root / "M" / "maya-mm3"
        pack.mkdir(parents=True)
        (pack / "mm3.txt").write_text("format gguf\n")
        pc = {"gpus": [{"index": 0}]}
        quiet = [patch.object(maya, k, lambda *a, **kw: None) for k in ("ok", "say", "step")]
        for p in quiet:
            p.start()
        try:
            before = (self.root / "maya-mm3.json").read_text()
            p = maya.write_config_mm3(args(), pc, {"lib_dirs": []}, pack, 65536, Path("/m"), pack.parent, "gguf-xs")
            self.assertEqual(p.name, "maya-minimax-maya-xs.json")
            c = json.loads(p.read_text())
            self.assertEqual((c["host"], c["installer"]["mm3_quant"]), ("0.0.0.0", "gguf-xs"))   # (settings kept)
            self.assertEqual((self.root / "maya-mm3.json").read_text(), before)                    # (EXL3's as it was)
            self.assertEqual(Q.config_label(c), "MiniMax M3.1 · Maya-XS")
            # a pre-release GGUF config under the old name: moved, its dashboard settings with it
            p.unlink()
            old = dict(json.loads(before), installer={"family": "minimax", "mm3_quant": "gguf-s"})
            (self.root / "maya-mm3-hip.json").write_text(json.dumps(old))
            (self.root / "maya-mm3-hip.shared-settings.json").write_text("{}")
            hip = {"gpus": [{"index": 0}], "backend": "hip"}
            with patch.object(maya, "hip_reserve_mib", return_value=1024):
                p = maya.write_config_mm3(args(), hip, {"lib_dirs": []}, pack, 65536, Path("/m"), pack.parent, "gguf-s")
            self.assertEqual(p.name, "maya-minimax-maya-s-hip.json")
            self.assertFalse((self.root / "maya-mm3-hip.json").exists())
            self.assertTrue((self.root / "maya-minimax-maya-s-hip.shared-settings.json").exists())
            self.assertEqual(json.loads(p.read_text())["host"], "0.0.0.0")
            # the EXL3 quant keeps its name
            p = maya.write_config_mm3(args(), pc, {"lib_dirs": []}, pack, 65536, Path("/m"), pack.parent, "exl3")
            self.assertEqual(p.name, "maya-mm3.json")
        finally:
            for p in quiet:
                p.stop()


class MiniMaxFiles(unittest.TestCase):
    """MiniMax's GGUFs under the scheme: <Model>-Maya-<Tier>-<maintype>-0000N-of-0000M.gguf in the model's one repo, a
    folder per tier; the files of the names before are used as they are."""

    def test_manifest(self):
        for q, tier, main, n in (("gguf-xs", "Maya-XS", "IQ1_S", 2), ("gguf-s", "Maya-S", "IQ2_XXS", 3)):
            m = maya.mm3_gguf_entry(q)
            self.assertEqual((m["repo"], m["folder"]), ("peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF", tier))
            base = f"MiniMax-M3.1-Flash-{tier}-{main}"
            self.assertEqual(list(m["files"]), [f"{base}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)] +
                             [f"MiniMax-M3.1-Flash-{tier}-maya-indexers.bin"])   # (not a {main} file)
            self.assertEqual(len(m["earlier_names"]), len(m["files"]))
            self.assertTrue(all(o.startswith(f"MiniMax-M3.1-Flash-{tier}-") for o in m["earlier_names"]))
            self.assertEqual(maya.hf_url(m["repo"], "main", m["folder"], f"{base}-00001-of-{n:05d}.gguf"),
                             f"https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF/resolve/main/{tier}/"
                             f"{base}-00001-of-{n:05d}.gguf")
            self.assertTrue(maya.mm3_gguf_published(q))                      # (uploaded: the commit and every sha256)
            self.assertRegex(m["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(all(len(h) == 64 for h in m["sha256"].values()))

    def test_files_under_their_earlier_names(self):
        m = maya.mm3_gguf_entry("gguf-xs")
        m["files"] = {n: 1000 + i for i, n in enumerate(m["files"])}
        quiet = [patch.object(maya, k, lambda *a, **kw: None) for k in ("ok", "warn", "step")]
        with tempfile.TemporaryDirectory() as d, patch.object(maya, "mm3_gguf_entry", return_value=m), \
                patch.object(maya, "rotational", return_value=False):
            for p in quiet:
                p.start()
            try:
                d = Path(d)
                for old, size in zip(m["earlier_names"], m["files"].values()):
                    with (d / old).open("wb") as f:
                        f.truncate(size)
                got = maya.mm3_gguf_model_step(args(), d, "gguf-xs")
            finally:
                for p in quiet:
                    p.stop()
        self.assertEqual([f.name for f in got], m["earlier_names"])

    def test_picker(self):
        rows = []

        def pick(title, intro, opts, default, yes, *x):
            rows.extend(o[0] for o in opts)
            return default
        with patch.object(maya, "choose", pick), patch.object(maya, "mm3_gguf_published", return_value=True), \
                patch.object(maya, "mem_gb", return_value=(128, 120)), patch.object(maya.S, "gpus", return_value=[]):
            self.assertEqual(maya.choose_mm3_quant(args(), {}), "gguf-xs")   # (the one a 128 GB PC holds)
        self.assertIn("MiniMax M3.1 · Maya-XS (93 GB)", rows)
        self.assertIn("MiniMax M3.1 · Maya-S (122 GB)", rows)


class DeepSeekContext(unittest.TestCase):
    """DeepSeek's recommended context: 256K wherever VRAM + RAM hold the files and the caches, else 64K (as before)."""

    def test_by_memory(self):
        xs = maya.ds4_memory_gb("maya-xs")
        self.assertEqual(xs, 121)
        self.assertEqual(maya.ds4_memory_gb("exl3"), maya.DS4_MEMORY_GB)
        self.assertGreater(maya.ds4_memory_gb("gguf-q8"), 500)             # (unsloth's: its files)
        self.assertEqual(maya.ds4_recommended_context(96, 128, xs), 262144)     # (a 128 GB Halo + its GPU share)
        self.assertEqual(maya.ds4_recommended_context(24, 256, xs, 1), 262144)
        self.assertEqual(maya.ds4_recommended_context(24, 64, xs), 65536)      # (experts from the SSD: 64K)
        self.assertEqual(maya.ds4_recommended_context(48, 256, maya.ds4_memory_gb("gguf-q8")), 65536)
        self.assertEqual(maya.ds4_recommended_context(48, 0, xs), 65536)       # (RAM not readable)
        spare = maya.ds4_kv_gb(131072, 2)                                      # (room for 128K's caches, not 256K's)
        self.assertEqual(maya.ds4_recommended_context(0, xs + spare, xs, 2), 131072)
        self.assertEqual(maya.DS4_RECOMMENDED_CONTEXT, 262144)

    def test_the_setup_recommends_it(self):
        """set_up_ds4 offers the recommendation as the context menu's default; a setup again keeps the context the
        config before has (choose_context: the earlier answer is the default)."""
        seen = []

        class Asked(Exception):
            pass

        def ctx(a, prev, contexts, recommended):
            seen.append(recommended)
            raise Asked
        with patch.object(maya, "check_pc", return_value={"gpus": [{"index": 0, "vram_gb": 24}]}), \
                patch.object(maya, "mem_gb", return_value=(256, 250)), patch.object(maya, "choose_context", ctx), \
                patch.object(maya, "step", lambda *a: None), patch.object(maya, "ok", lambda *a: None):
            with self.assertRaises(Asked):
                maya.set_up_ds4(args(ds4_quant="maya-xs"), {})
            with patch.object(maya, "mem_gb", return_value=(64, 60)), self.assertRaises(Asked):
                maya.set_up_ds4(args(ds4_quant="maya-xs"), {})
        self.assertEqual(seen, [262144, 65536])
        with patch.object(maya, "choose", lambda t, i, opts, default, yes, *x: default):
            self.assertEqual(maya.choose_context(args(context=None), 65536, maya.DS4_CONTEXTS, 262144), 65536)
            self.assertEqual(maya.choose_context(args(context=None), None, maya.DS4_CONTEXTS, 262144), 262144)


if __name__ == "__main__":
    unittest.main()
