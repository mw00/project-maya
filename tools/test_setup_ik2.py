"""tools/test_setup_ik2.py - Inkling-2-Mini (maya-ik2, an experimental preview) in the setup: its names and setup ids,
its own defaults (its author's), the pinned manifest, the one-line refusals on the hosts the preview
does not run on (AMD, a Mac, Windows, two GPUs or more asked for, a card older than V100 / RTX 20 or under 24 GB), the
one GPU it picks, what fits on the card (the context, the drafter's room), the picker's row, the tokenizer step, the
config it writes (its pictures too) and a start of it.  No GPU, no network.

    python -m unittest tools.test_setup_ik2 -v
"""
from pathlib import Path
import io
import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402
import ik2_fit as F  # noqa: E402


def args(**kw):
    base = dict(model=maya.IK2, hy4_quant=None, mm3_quant=None, ds4_quant=None, qwen_quant=None, model_dir=None,
                models_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, check=False, port=None, host=None,
                api_key=None, env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None, repack=False,
                parallel=None, drafting=None)
    base.update(kw)
    return SimpleNamespace(**base)


def card(gib, index=0, arch="86", name="NVIDIA GeForce RTX 3090"):
    return {"index": index, "name": name, "arch": arch, "vram_gb": float(gib), "driver": "580.0"}


def amd(arch, gib, index=0, **kw):
    return {"index": index, "name": "AMD Radeon", "arch": arch, "vram_gb": float(gib), **kw}


def refusal(fn):
    """The line a fail() printed, and that it stopped (SystemExit)."""
    out = io.StringIO()
    with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
        try:
            fn()
        except SystemExit:
            return out.getvalue()
    raise AssertionError("no refusal: " + out.getvalue()[-400:])


class Names(unittest.TestCase):
    def test_shown_and_ids(self):
        self.assertEqual(Q.shown("ik2", Q.IK2_QUANT, 18.27), "Inkling 2 Mini · unsloth UD-Q4_K_XL (18 GB)")
        self.assertEqual(Q.config_stem("ik2", Q.IK2_QUANT), "maya-inkling-2-mini")
        for name in ("inkling-2-mini", "Inkling-2-Mini", "inkling-2-mini-ud-q4_k_xl", "ik2-mini", "INKLING2-MINI"):
            self.assertEqual(Q.resolve(name), ("ik2", Q.IK2_QUANT), name)
        self.assertIsNone(Q.resolve("inkling-2-small"))   # (Inkling-2-Small: by hand, no setup)
        self.assertEqual((maya.IK2, maya.IK2_SHOWN, maya.IK2_MODEL_NAME),
                         ("Inkling-2-Mini", "Inkling 2 Mini", "inkling-2-mini"))
        self.assertIn("ik2", maya.FAMILIES)
        self.assertEqual(maya.FAMILY_TITLE["ik2"], "Inkling-2-Mini")

    def test_model_option(self):
        a = args(model="inkling-2-mini")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.model, maya.IK2)
        self.assertEqual(maya.choose_family(a, {}), "ik2")

    def test_its_own_defaults(self):
        """Inkling-2-Mini has its own entry, its author's values (not Inkling-2-Small's): temperature 1.0, top-p 1.0,
        no top-k, no penalties, its template's own level by name (high: 0.9); a config written by hand for
        Inkling-2-Small keeps the PR's."""
        mini = {"exe": "/b/maya-ik2", "installer": {"family": "ik2", "ik2_quant": Q.IK2_QUANT}}
        self.assertEqual(Q.model_key(mini), "ik2-mini")
        self.assertEqual(Q.model_defaults(mini), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        by_hand = {"exe": "build/maya-ik2", "model_name": "inkling-2-mini",
                   "args": ["--model", "Inkling-2-Mini-UD-Q4_K_XL.gguf"]}
        self.assertEqual(Q.model_key(by_hand), "ik2-mini")
        small = {"exe": "build/maya-ik2", "model_name": "inkling-2-small",
                 "args": ["--model", "Inkling-2-Small-UD-Q4_K_XL-00001-of-00009.gguf"]}
        self.assertEqual((Q.model_key(small), Q.model_defaults(small)[1]), ("ik2", "auto"))

    def test_label(self):
        cfg = {"exe": "/b/maya-ik2", "installer": {"family": "ik2", "ik2_quant": Q.IK2_QUANT}}
        self.assertEqual(Q.config_label(cfg), "Inkling 2 Mini · unsloth UD-Q4_K_XL")
        self.assertEqual(Q.config_key(cfg), Q.IK2_QUANT)


class Manifest(unittest.TestCase):
    def test_pinned(self):
        man = json.loads(maya.IK2_MANIFEST.read_text(encoding="utf-8"))
        self.assertRegex(man["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(man["repo"], "unsloth/Inkling-2-Mini-GGUF")
        self.assertEqual(man["gguf"], "Inkling-2-Mini-UD-Q4_K_XL.gguf")      # (the repo's top: the looped layout)
        for name, (size, sha) in man["files"].items():
            self.assertGreater(size, 100 << 20, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
        self.assertEqual(list(man["files"]), [man["gguf"]])                  # (the drafter: its own entry)
        self.assertEqual(man["files"][man["gguf"]][1],
                         "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15")   # (the author's)

    def test_drafter_pinned(self):
        """Drafting: thinkingmachines' own mtp.safetensors from the model's repo at a pinned revision, by size and
        sha256 (the ones docs/INKLING2.md lists)."""
        m = maya.ik2_drafter_model()
        self.assertEqual((m["repo"], m["revision"], m["folder"]),
                         ("thinkingmachines/Inkling-2-Mini", "ed40e7da625f594fc9c4990f5f66be24873c22d7", ""))
        self.assertEqual(m["files"], {"mtp.safetensors": 635801416})
        self.assertEqual(m["sha256"]["mtp.safetensors"],
                         "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2")
        self.assertIn("7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2",
                      (maya.ROOT / "docs" / "INKLING2.md").read_text(encoding="utf-8"))
        self.assertEqual(maya.file_url(m, "mtp.safetensors"), "https://huggingface.co/thinkingmachines/Inkling-2-Mini/"
                         "resolve/ed40e7da625f594fc9c4990f5f66be24873c22d7/mtp.safetensors")

    def test_vision_pinned(self):
        """The pictures: unsloth's mmproj-BF16.gguf from the GGUF's own repo and revision, by size and sha256."""
        man = json.loads(maya.IK2_MANIFEST.read_text(encoding="utf-8"))
        m = maya.ik2_vision_model()
        self.assertEqual((m["repo"], m["revision"], m["folder"]), (man["repo"], man["revision"], ""))
        self.assertEqual(m["files"], {"mmproj-BF16.gguf": 641805760})
        self.assertEqual(m["sha256"]["mmproj-BF16.gguf"],
                         "2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7")
        self.assertEqual(maya.file_url(m, "mmproj-BF16.gguf"), "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/"
                         f"resolve/{man['revision']}/mmproj-BF16.gguf")

    def test_entry_and_urls(self):
        m = maya.ik2_entry()
        self.assertAlmostEqual(m["download_gb"], 17.63, places=1)
        gguf = maya.file_url(m, m["gguf"])
        self.assertTrue(gguf.startswith("https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/"), gguf)
        self.assertTrue(gguf.endswith("/Inkling-2-Mini-UD-Q4_K_XL.gguf"), gguf)
        self.assertNotIn("Mainline_Support", gguf)


class Refusals(unittest.TestCase):
    """One plain line, before anything is built or downloaded, wherever the preview does not run."""

    def test_the_lines(self):
        ok = [card(24)]
        cases = [
            ("metal", False, 0, [], "NVIDIA and AMD GPUs only"),
            ("hip", True, 0, [], "Linux only"),
            ("hip", False, 2, [], "one GPU only"),
            ("hip", False, 0, [amd("gfx1030", 16)], "RDNA 3, 3.5 or 4"),
            ("hip", False, 0, [amd("gfx1201", 16)], "24 GB of VRAM or more"),
            ("hip", False, 0, [amd("gfx1151", 0.5, unified_memory=True, memory_gb=20)], "24 GB of VRAM or more"),
            ("cuda", True, 0, ok, "Linux only"),
            ("cuda", False, 2, ok, "one GPU only"),
            ("cuda", False, 0, [card(24, arch="61", name="Tesla P40")], "V100 / RTX 20 or newer"),
            ("cuda", False, 0, [card(6, arch="86")], "8 GB of VRAM or more"),
            ("cuda", False, 0, [card(6, arch="70"), card(4, 1, arch="89")], "8 GB of VRAM or more"),
        ]
        for backend, win, asked, cards, part in cases:
            why = F.unsupported(backend, win, asked, cards)
            self.assertIn(part, why or "", (backend, win, asked))
            self.assertNotIn("\n", why)
        for cards in (ok, [card(23.99, arch="89")], [card(80, arch="90")], [card(12, arch="86"), card(24, 1)], [],
                      [card(32, arch="70", name="Tesla V100-PCIE-32GB")], [card(24, arch="75", name="NVIDIA TITAN RTX")],
                      [card(8, arch="86")], [card(16, arch="70"), card(12, 1, arch="89")]):
            self.assertIsNone(F.unsupported("cuda", False, 1, cards))
        # (AMD on Linux: RDNA 3 / 3.5 / 4 with 24 GB, or an APU whose shared memory leaves that much)
        for cards in ([amd("gfx1100", 24)], [amd("gfx1201", 32)], [amd("gfx1030", 16), amd("gfx1100", 24, 1)],
                      [amd("gfx1151", 0.5, unified_memory=True, memory_gb=168)], []):
            self.assertIsNone(F.unsupported("hip", False, 1, cards))

    def test_ram_and_vram_together(self):
        """No SSD tier (the owner's call): RAM + VRAM under the model is refused in one line naming the amount."""
        why = F.unsupported("cuda", False, 1, [card(8, arch="86")], ram_gb=12.0)
        self.assertIn("24 GB of RAM and VRAM together", why or "")
        self.assertNotIn("\n", why)
        self.assertIsNone(F.unsupported("cuda", False, 1, [card(8, arch="86")], ram_gb=32.0))
        self.assertIsNone(F.unsupported("cuda", False, 1, [card(24, arch="86")], ram_gb=8.0))   # (all on the card)
        # (only the experts the card does not keep are pinned: a 16 GB PC, ~15.4 GiB as Linux reports it, with an
        #  8 GB card)
        self.assertIsNone(F.unsupported("cuda", False, 1, [card(8, arch="86")], ram_gb=15.4))

    def test_check_pc_refuses_first(self):
        for a, part in ((args(gpus="0,1"), "one GPU only"),):
            with patch.object(maya.S, "gpus", side_effect=AssertionError("no GPU read before the refusal")), \
                    patch.object(maya, "WIN", False):
                out = refusal(lambda: maya.check_pc(a, "ik2"))
            self.assertIn(part, out)
            self.assertEqual(len([ln for ln in out.splitlines() if "[X]" in ln]), 1)
        with patch.object(maya, "WIN", True):
            self.assertIn("Linux only", refusal(lambda: maya.check_pc(args(), "ik2")))

    def test_check_pc_refuses_an_old_or_small_card(self):
        for found, part in (([card(24, arch="61", name="Tesla P40")], "V100 / RTX 20 or newer"),
                            ([card(6, arch="86", name="NVIDIA GeForce RTX 3050 6GB")], "8 GB of VRAM or more")):
            with patch.object(maya, "WIN", False), patch.object(maya.sys, "platform", "linux"), \
                    patch.object(maya.S, "is_wsl", return_value=False), patch.object(maya.S, "gpus", return_value=found):
                self.assertIn(part, refusal(lambda: maya.check_pc(args(), "ik2")))
        with patch.object(maya, "WIN", False), patch.object(maya.sys, "platform", "linux"), \
                patch.object(maya.S, "is_wsl", return_value=False), \
                patch.object(maya.S, "gpus", return_value=[card(6, 0), card(24, 1, arch="86")]):
            # (--gpu 0 names the 6 GB card: refused; the 24 GB one would do)
            self.assertIn("8 GB of VRAM or more", refusal(lambda: maya.check_pc(args(gpu=0), "ik2")))
        with patch.object(maya, "WIN", False), patch.object(maya.sys, "platform", "linux"), \
                patch.object(maya.S, "is_wsl", return_value=False), patch.object(maya, "mem_gb", return_value=(12.0, 10.0)), \
                patch.object(maya.S, "gpus", return_value=[card(8, arch="86", name="NVIDIA GeForce RTX 3050")]):
            # (no SSD tier: 12 GB of RAM beside an 8 GB card cannot hold the model)
            self.assertIn("24 GB of RAM and VRAM together", refusal(lambda: maya.check_pc(args(), "ik2")))

    def test_one_gpu_picked(self):
        found = [card(12, 0, arch="86"), card(24, 1, arch="86"), card(24, 2, arch="89"), card(32, 3, arch="61")]
        with patch.object(maya, "ok"):
            got = maya.choose_gpus(args(), found, maya.IK2_MIN_CC, "ik2")
            self.assertEqual([g["index"] for g in got], [1])          # (the largest new enough; the Pascal is too old)
            got = maya.choose_gpus(args(), found, maya.IK2_MIN_CC, "ik2", before=[2])
            self.assertEqual([g["index"] for g in got], [2])          # (the one its config ran on)
            got = maya.choose_gpus(args(), [card(12, 0, arch="86"), card(32, 1, arch="70", name="Tesla V100")],
                                   maya.IK2_MIN_CC, "ik2")
            self.assertEqual([g["index"] for g in got], [1])          # (a V100: Volta on)
        self.assertEqual(maya.ik2_gpus_asked(args(gpus="0, 1")), 2)
        self.assertEqual(maya.ik2_gpus_asked(args(gpu=3)), 1)
        self.assertEqual(maya.ik2_gpus_asked(args()), 0)


class Fit(unittest.TestCase):
    def test_a_24_gb_card(self):
        """The contributor's figures on an RTX 3090: 262,144 with an FP8 cache and every expert in VRAM, as
        docs/INKLING2.md says it fits."""
        self.assertEqual(F.recommended_context(24.0), 262144)
        p = F.plan(24.0, 262144)
        self.assertTrue(p["resident"])
        self.assertLessEqual(p["need_gib"], 24.0 - F.MARGIN_GIB)
        self.assertFalse(F.plan(24.0, 262144, "f16")["resident"])   # (F16: twice the cache - the slower path)
        self.assertEqual(F.recommended_context(24.0, "f16"), 131072)

    def test_scales_with_the_card(self):
        self.assertTrue(F.plan(32.0, 262144, "f16")["resident"])     # (an RTX 5090: F16 at 256K too)
        self.assertEqual(F.plan(0, 262144)["resident"], True)          # (VRAM not read: the engine measures)
        self.assertEqual(F.recommended_context(8.0), F.CONTEXTS[0])    # (nothing resident: the smallest)

    def test_smaller_cards_offload(self):
        """8 / 12 / 16 GB cards: the offload path, a share of the experts on the card growing with it; the verdict
        says so, and "tight" where the RAM is under what the path pins."""
        tiers = [F.plan(v, 32768) for v in (8.0, 12.0, 16.0)]
        self.assertTrue(all(p["layout"] == "offload" and not p["resident"] for p in tiers))
        self.assertTrue(0 < tiers[0]["tier_gib"] < tiers[1]["tier_gib"] < tiers[2]["tier_gib"] < F.EXPERTS_GIB)
        self.assertEqual(F.plan(24.0, 262144)["layout"], "resident")
        self.assertEqual(F.verdict(24.0, 64.0, 262144)[0], "fast")
        self.assertEqual(F.verdict(8.0, 64.0, 32768)[0], "runs")
        # (the RAM: the experts the card does not keep, pinned, and the engine's own beside them)
        for p in tiers:
            self.assertAlmostEqual(p["ram_gib"], F.EXPERTS_GIB - p["tier_gib"] + F.OFF_HOST_GIB, places=1)
        self.assertTrue(tiers[0]["ram_gib"] > tiers[1]["ram_gib"] > tiers[2]["ram_gib"])
        self.assertEqual(F.verdict(8.0, 15.4, 32768)[0], "runs")       # (16 GB of RAM)
        self.assertEqual(F.verdict(8.0, tiers[0]["ram_gib"] + 0.5, 32768)[0], "tight")
        self.assertEqual(F.verdict(8.0, 12.0, 32768)[0], "does not fit")


class Picker(unittest.TestCase):
    def test_its_row_last_and_picked(self):
        got = {}

        def pick(title, intro, opts, default, yes, *x):
            got.update(rows=[o[0] for o in opts], about={o[0]: o[2] for o in opts})
            return len(opts) - 1
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]):
            fam = maya.choose_family(args(model=None, model_dir=None, gguf_dir=None, draft=None), {})
        self.assertEqual((got["rows"][-1], fam), ("Inkling 2 Mini (experimental)", "ik2"))
        about = got["about"]["Inkling 2 Mini (experimental)"]
        for part in ("EXPERIMENTAL PREVIEW", "thinkingmachines' Inkling-2-Mini", "UD-Q4_K_XL", "18 GB",
                     "V100 / RTX 20", "Linux", "8 GB or more", "Halo / APU with 24 GB", "pictures"):
            self.assertIn(part, about)


class Steps(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ("ok", "say", "step", "warn"):
            p = patch.object(maya, name, lambda *a, **kw: None)
            p.start()
            self.addCleanup(p.stop)

    def files(self):
        d = self.root / "models" / maya.IK2
        d.mkdir(parents=True, exist_ok=True)
        gguf = d / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 60)
        return d, gguf

    def test_tokenizer_with_its_own_template(self):
        d, gguf = self.files()
        seen = []

        def run(cmd, check=True, **kw):
            seen.append(cmd)
            out = Path(cmd[cmd.index("--out") + 1]) / "tokenizer"
            out.mkdir(parents=True, exist_ok=True)
            (out / "vocab.json").write_text("{}", encoding="utf-8")
            (out / "chat_template.jinja").write_text("x", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with patch.object(maya, "run", run):
            tok = maya.ik2_tokenizer_step(args(), d, gguf)
            self.assertEqual(tok, d / "maya-ik2" / "tokenizer")
            cmd = seen[0]
            self.assertEqual(cmd[cmd.index("--chat-template") + 1], str(maya.IK2_TEMPLATE))
            self.assertIn("--check", cmd)
            rec = json.loads((d / "maya-ik2" / "sources.json").read_text(encoding="utf-8"))
            self.assertEqual([f["name"] for f in rec["files"]], [gguf.name])
            maya.ik2_tokenizer_step(args(), d, gguf)              # (the same tools and GGUF: not exported again)
            self.assertEqual(len(seen), 1)

    def test_the_config(self):
        d, gguf = self.files()
        tok = d / "maya-ik2" / "tokenizer"
        tok.mkdir(parents=True)
        (tok / "vocab.json").write_text("{}", encoding="utf-8")
        pc = {"gpus": [card(24, 1)], "backend": "cuda"}
        with patch.object(maya, "ROOT", self.root), \
                patch.object(maya, "write_run_script", lambda p, port: p.with_name("run-" + p.stem + ".sh")):
            path = maya.write_config_ik2(args(), pc, {}, gguf, tok, 262144, self.root / "models", d)
        cfg = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(path.name, "maya-inkling-2-mini.json")
        self.assertEqual(cfg["exe"], str(maya.IK2_EXE))
        self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "262144", "--drafting", "auto"])
        self.assertEqual(cfg["env"], {"IK2_KV": "fp8"})
        self.assertEqual((cfg["sampling"], cfg["reasoning_effort"]), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        self.assertEqual((cfg["model_name"], cfg["gpu"], cfg["tokenizer"]), ("inkling-2-mini", [1], str(tok)))
        inst = cfg["installer"]
        self.assertEqual((inst["family"], inst["ik2_quant"], inst["experimental"]), ("ik2", Q.IK2_QUANT, True))
        self.assertEqual((Q.family_of(cfg), Q.model_key(cfg)), ("ik2", "ik2-mini"))
        # (a smaller context; the user's own variable kept)
        cfg["env"]["MY_VAR"] = "1"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with patch.object(maya, "ROOT", self.root), \
                patch.object(maya, "write_run_script", lambda p, port: p.with_name("run-" + p.stem + ".sh")):
            maya.write_config_ik2(args(), pc, {}, gguf, tok, 65536, self.root / "models", d)
        cfg = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "65536", "--drafting", "auto"])
        self.assertEqual(cfg["env"], {"IK2_KV": "fp8", "MY_VAR": "1"})
        # (the drafter: --draft beside --drafting auto, the default; on / off by hand; off names no drafter)
        mtp = d / "mtp.safetensors"
        for how, want in ((None, ["--draft", str(mtp), "--drafting", "auto"]),
                          ("measure", ["--draft", str(mtp), "--drafting", "auto"]),
                          ("on", ["--draft", str(mtp), "--drafting", "on"]), ("off", ["--drafting", "off"])):
            with patch.object(maya, "ROOT", self.root), \
                    patch.object(maya, "write_run_script", lambda p, port: p.with_name("run-" + p.stem + ".sh")):
                maya.write_config_ik2(args(drafting=how), pc, {}, gguf, tok, 65536, self.root / "models", d,
                                      None, mtp)
            cfg = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["args"][4:], want, how)
            self.assertEqual(cfg["installer"]["drafting"], {"measure": "auto", None: "auto"}.get(how, how))

    def test_drafter_step(self):
        """Drafting on by default: the drafter beside the GGUF, sha256-checked; none with --drafting off or a declined
        download (the prompt lookup then drafts alone); truthful where its room does not fit at the context."""
        d, gguf = self.files()
        va = lambda **kw: args(**dict(dict(rebuild=False, download_model=False, yes=False), **kw))   # noqa: E731
        self.assertIsNone(maya.ik2_drafter_step(va(drafting="off"), d, 24, 65536, "fp8"))
        with patch.object(maya, "offer_download", lambda *a, **k: False):
            self.assertIsNone(maya.ik2_drafter_step(va(), d, 24, 65536, "fp8"))
        with patch.object(maya, "file_incomplete", lambda *a: None), \
                patch.object(maya, "sha256_checked", lambda *a: True):
            self.assertEqual(maya.ik2_drafter_step(va(), d, 24, 65536, "fp8"), d / "mtp.safetensors")
            said = []
            with patch.object(maya, "warn", said.append):
                self.assertEqual(maya.ik2_drafter_step(va(), d, 24, 262144, "fp8"), d / "mtp.safetensors")
            self.assertIn("does not fit beside the model", " ".join(said))

    def test_drafter_room(self):
        """tools/ik2_fit.py's drafter room (maya-ik2's draft_room): 24 GB cards hold it up to about 200K tokens with the
        FP8 cache, 32 GB at 262K; it grows 4.75 KiB a token."""
        self.assertTrue(F.draft_fits(24, 131072, "fp8"))
        self.assertFalse(F.draft_fits(24, 262144, "fp8"))
        self.assertTrue(F.draft_fits(32, 262144, "fp8"))
        self.assertTrue(F.draft_fits(0, 262144, "fp8"))          # (not known: the engine measures)
        self.assertAlmostEqual(F.draft_gib(262144) - F.draft_gib(0), 262144 * 4864 / 2**30, places=6)

    def test_drafting_off_reason(self):
        """Volta / Turing and the offload path run with drafting off; sm_80+ with every expert in VRAM and AMD keep it."""
        self.assertIn("Volta", F.drafting_off_reason("70", 32, 65536, "fp8"))
        self.assertIn("Volta", F.drafting_off_reason("75", 24, 65536, "fp8"))
        self.assertIn("offload", F.drafting_off_reason("86", 8, 65536, "fp8"))
        self.assertIsNone(F.drafting_off_reason("86", 24, 65536, "fp8"))
        self.assertIsNone(F.drafting_off_reason("120", 32, 262144, "fp8"))
        self.assertIsNone(F.drafting_off_reason("gfx1151", 150, 262144, "fp8", hip=True))
        self.assertIsNone(F.drafting_off_reason(None, 0, 65536, "fp8"))      # (not known)

    def run_setup(self, gpu, ram=64.0, backend="cuda", **kw):
        """set_up_ik2 with its steps stubbed: what it hands the drafter step and the config (drafting, drafter)."""
        d, gguf = self.files()
        pc = {"gpus": [gpu], "backend": backend}
        seen = {}
        stub = lambda name, ret: patch.object(maya, name, lambda *a, **k: ret)   # noqa: E731
        drafter = patch.object(maya, "ik2_drafter_step", lambda *a, **k: seen.setdefault("drafter", d / "mtp.safetensors"))
        cfg = patch.object(maya, "write_config_ik2", lambda a, *r: seen.update(drafting=a.drafting, drafter=r[-1]))
        said = []
        with patch.object(maya, "check_pc", return_value=pc), patch.object(maya, "mem_gb", return_value=(ram, 10.0)),                 patch.object(maya, "choose_context", lambda a, prev, opts, rec: 65536),                 stub("ik2_dir", d), stub("tools_step", d), stub("build_step", {}), stub("ik2_model_step", gguf),                 stub("ik2_tokenizer_step", d), stub("ik2_vision_step", None), drafter, cfg,                 patch.object(maya.IK2_EXE.__class__, "exists", lambda self: True),                 patch.object(maya, "ok", said.append), patch.object(maya, "warn", lambda *a: None),                 patch.object(maya, "step", lambda *a: None), patch.object(maya, "hip_unified_memory", lambda g: False), patch.object(maya, "WIN", False):
            maya.set_up_ik2(self.va(**kw), {})
        return seen, " ".join(said)

    def test_setup_drafting_off_where_it_does_not_run(self):
        """Volta / Turing and offload layouts: no drafter downloaded, drafting off, one line saying why."""
        for gpu in (card(32, arch="70", name="Tesla V100"), card(24, arch="75", name="RTX 2080 Ti"), card(8)):
            seen, said = self.run_setup(gpu, drafting="on")
            self.assertEqual(seen["drafting"], "off", gpu)
            self.assertIsNone(seen["drafter"], gpu)
            self.assertIn("drafting off", said)

    def test_setup_drafting_kept_where_it_runs(self):
        """NVIDIA sm_80+ holding every expert and AMD: the drafter is fetched and drafting stays auto."""
        for gpu, backend in ((card(24), "cuda"), (card(32, arch="120"), "cuda"), (amd("gfx1100", 24), "hip")):
            seen, said = self.run_setup(gpu, backend=backend)
            self.assertIsNone(seen["drafting"], gpu)
            self.assertEqual(seen["drafter"].name, "mtp.safetensors", gpu)
            self.assertNotIn("drafting off", said)

    def va(self, **kw):
        return args(**dict(dict(rebuild=False, download_model=False, yes=False), **kw))

    def test_vision_step(self):
        """Step 7: on by default - the mmproj beside the GGUF and maya-vision, no text model in the entry (the pinned
        llama.cpp does not open Inkling's GGUF; the encoder needs none); none with --no-vision, a declined download
        or an encoder that did not compile."""
        d, gguf = self.files()
        self.assertIsNone(maya.ik2_vision_step(self.va(no_vision=True), {}, {}, d, d))
        with patch.object(maya, "offer_download", lambda *a, **k: False):
            self.assertIsNone(maya.ik2_vision_step(self.va(), {}, {}, d, d))
        sure = [patch.object(maya, "file_incomplete", lambda *a: None), patch.object(maya, "sha256_checked", lambda *a: True)]
        for p in sure:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(maya, "vision_encoder_ready", lambda *a: True):
            v = maya.ik2_vision_step(self.va(), {}, {}, d, d)
        self.assertEqual(v, {"exe": str(maya.VEXE), "mmproj": str(d / "mmproj-BF16.gguf"), "gpu": True})
        with patch.object(maya, "vision_encoder_ready", lambda *a: False):
            self.assertIsNone(maya.ik2_vision_step(self.va(), {}, {}, d, d))

    def test_the_config_with_pictures(self):
        d, gguf = self.files()
        tok = d / "maya-ik2" / "tokenizer"
        tok.mkdir(parents=True)
        v = {"exe": "/b/maya-vision", "mmproj": str(d / "mmproj-BF16.gguf"), "gpu": True}
        pc = {"gpus": [card(24, 0)], "backend": "cuda"}
        with patch.object(maya, "ROOT", self.root), \
                patch.object(maya, "write_run_script", lambda p, port: p.with_name("run-" + p.stem + ".sh")):
            path = maya.write_config_ik2(args(), pc, {}, gguf, tok, 262144, self.root / "models", d, v)
            cfg = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["vision"], v)
            self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "262144", "--drafting", "auto"])
            maya.write_config_ik2(args(), pc, {}, gguf, tok, 262144, self.root / "models", d)   # (--no-vision)
            self.assertNotIn("vision", json.loads(path.read_text(encoding="utf-8")))

    def test_a_start_of_it(self):
        d, gguf = self.files()
        exe = self.root / "maya-ik2"
        exe.touch()
        tok = d / "maya-ik2" / "tokenizer"
        tok.mkdir(parents=True)
        (tok / "vocab.json").write_text("{}", encoding="utf-8")
        cfg = {"exe": str(exe), "tokenizer": str(tok), "gpu": [0], "model_name": "inkling-2-mini",
               "installer": {"family": "ik2", "ik2_quant": Q.IK2_QUANT},
               "args": ["--model", str(gguf), "--max-context", "65536"]}
        path = self.root / "maya-inkling-2-mini.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        a = SimpleNamespace(parallel=None, port=None, host=None, api_key=None, gpus=None, gpu=None, yes=True)
        quiet = [patch.object(maya, n, return_value=None) for n in ("select_build_backend", "refresh_engine",
                                                                      "rename_programs", "refresh_mixed_builds")]
        for p in quiet:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(maya, "speed_calibration", side_effect=AssertionError("no calibration")), \
                patch.object(maya, "prefill_calibration", side_effect=AssertionError("no calibration")), \
                patch.object(maya, "link_calibration", side_effect=AssertionError("no calibration")), \
                patch.dict(os.environ, {"INVOCATION_ID": "x"}):
            cmd, env, info = maya.server_command(path, a)
        self.assertIn("serve", " ".join(cmd))
        self.assertEqual((info["model"], info["label"]), ("inkling-2-mini", "Inkling 2 Mini · unsloth UD-Q4_K_XL"))
        gguf.unlink()                                          # (its GGUF gone: said, not started)
        out = refusal(lambda: maya.server_command(path, a))
        self.assertIn("the model's GGUF file is missing", out)

    def test_bench_prompt(self):
        chat = maya.bench_chat("ik2", self.root)
        p = chat("Hello?")
        self.assertIn("Thinking effort level: 0<|end_message|>", p)
        self.assertTrue(p.endswith("<|message_user|><|content_text|>Hello?<|end_message|>"
                                   "<|message_model|><|content_text|>"), p[-120:])

    def test_nothing_to_tune_and_no_calibration(self):
        cfg = {"exe": "/b/maya-ik2", "args": ["--model", "/m/x.gguf"],
               "installer": {"family": "ik2"}}
        self.assertFalse(maya.needs_speed_calibration(cfg))
        path = self.root / "maya-inkling-2-mini.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        self.assertTrue(maya.calibrate_config(path))
        self.assertEqual(maya.exe_for_target("maya-ik2"), maya.IK2_EXE)
        self.assertIn("maya-ik2", maya.ENGINE_TARGETS)
        self.assertEqual(maya.bench_exe(cfg), maya.IK2_EXE)


if __name__ == "__main__":
    unittest.main()
