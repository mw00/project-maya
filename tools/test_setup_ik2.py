"""tools/test_setup_ik2.py - Inkling-2-Mini (maya-ik2) in the setup: its names and setup ids,
its own defaults (its author's), the pinned manifest, the one-line refusals on the hosts Inkling 2
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

    def test_ablate_direction_entry(self):
        """The refusal direction: Maya's repo, folder refusal-directions/; its revision and sha256 stay PLACEHOLDERs until
        the file is uploaded (then the setup reads main and checks the file's shape), a real pin is used as pinned."""
        man = json.loads(maya.IK2_MANIFEST.read_text(encoding="utf-8"))["ablate"]
        name = "Inkling-2-Mini-refusal-direction.gguf"
        self.assertEqual((man["repo"], man["folder"]), ("peasantsmith/Inkling-2-Mini-Maya-GGUF", "refusal-directions"))
        self.assertEqual(list(man["files"]), [name])
        self.assertEqual(maya.ABLATE_DIRECTIONS["ik2"], ("Inkling-2-Mini", man["repo"]))
        if man["revision"] == "PLACEHOLDER":
            self.assertIsNone(maya.ablate_pin("ik2", name))
        pinned = {"ablate": {"revision": "a" * 40, "files": {name: [9760, "b" * 64]}}}
        with patch.object(maya, "read_json", lambda p: pinned):
            self.assertEqual(maya.ablate_pin("ik2", name), ("a" * 40, 9760, "b" * 64))
            self.assertIsNone(maya.ablate_pin("glm", name))
        pinned["ablate"]["files"][name][1] = "PLACEHOLDER"
        with patch.object(maya, "read_json", lambda p: pinned):
            self.assertIsNone(maya.ablate_pin("ik2", name))

    def test_drafter_pinned(self):
        """Drafting: the model's own drafter from Maya's own repo at a pinned revision, folder drafter/, by size and
        sha256 (the ones docs/INKLING2.md lists) - the GGUF (Q8_0), on NVIDIA and on AMD; thinkingmachines'
        mtp.safetensors by name (an earlier setup's); each form one file."""
        repo, rev = "peasantsmith/Inkling-2-Mini-Maya-GGUF", "37027719327de5f574684b525ad88fe50c4ff222"
        doc = (maya.ROOT / "docs" / "INKLING2.md").read_text(encoding="utf-8")
        for amd_pc, by_name, name, size, sha in (
                (False, None, "dspark-Inkling-2-Mini-Q8_0.gguf", 402278560,
                 "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"),
                (True, None, "dspark-Inkling-2-Mini-Q8_0.gguf", 402278560,
                 "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"),
                (True, "mtp.safetensors", "mtp.safetensors", 635801416,
                 "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2")):
            m = maya.ik2_drafter_model(amd_pc, by_name)
            self.assertEqual((m["repo"], m["revision"], m["folder"]), (repo, rev, "drafter"))
            self.assertEqual(m["files"], {name: size})
            self.assertEqual(m["sha256"], {name: sha})
            self.assertAlmostEqual(m["download_gb"], size / 1e9)
            self.assertIn(sha, doc)
            self.assertEqual(maya.file_url(m, name), f"https://huggingface.co/{repo}/resolve/{rev}/drafter/{name}")
            self.assertEqual(maya.ik2_drafter_model(not amd_pc, name), m)       # (a form by its name)
        self.assertEqual(maya.ik2_drafter_model(), maya.ik2_drafter_model(False))

    def test_vision_pinned(self):
        """The pictures: the mmproj from Maya's own repo (vision/ at a pinned revision), by size and sha256 - unsloth's
        file byte for byte."""
        m = maya.ik2_vision_model()
        rev = "7ab64bac7090c9f77df551ac0fac98119494a2a5"
        self.assertEqual((m["repo"], m["revision"], m["folder"]), ("peasantsmith/Inkling-2-Mini-Maya-GGUF", rev, "vision"))
        self.assertEqual(m["files"], {"mmproj-Inkling-2-Mini-BF16.gguf": 641805760})
        self.assertEqual(m["sha256"]["mmproj-Inkling-2-Mini-BF16.gguf"],
                         "2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7")
        self.assertEqual(maya.file_url(m, "mmproj-Inkling-2-Mini-BF16.gguf"),
                         "https://huggingface.co/peasantsmith/Inkling-2-Mini-Maya-GGUF/resolve/"
                         f"{rev}/vision/mmproj-Inkling-2-Mini-BF16.gguf")

    def test_entry_and_urls(self):
        m = maya.ik2_entry()
        self.assertAlmostEqual(m["download_gb"], 17.63, places=1)
        gguf = maya.file_url(m, m["gguf"])
        self.assertTrue(gguf.startswith("https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/"), gguf)
        self.assertTrue(gguf.endswith("/Inkling-2-Mini-UD-Q4_K_XL.gguf"), gguf)
        self.assertNotIn("Mainline_Support", gguf)


class Refusals(unittest.TestCase):
    """One plain line, before anything is built or downloaded, wherever Inkling 2 does not run."""

    def test_the_lines(self):
        ok = [card(24)]
        cases = [
            ("metal", False, 0, [], "NVIDIA and AMD GPUs only"),
            ("hip", True, 0, [], "Linux only"),
            ("hip", False, 2, [], "on AMD runs on one GPU"),
            ("hip", False, 0, [amd("gfx1030", 16)], "RDNA 3, 3.5 or 4"),
            ("hip", False, 0, [amd("gfx1102", 8)], "12 GB of VRAM or more"),
            ("hip", False, 0, [amd("gfx1151", 0.5, unified_memory=True, memory_gb=20)], "APU with 24 GB"),
            ("cuda", True, 2, ok, "two are Linux only"),
            ("cuda", False, 3, ok, "one or two NVIDIA GPUs"),
            ("cuda", False, 2, [card(24), card(6, 1, arch="86")], "8 GB of VRAM or more on each"),
            ("cuda", True, 0, [card(6, arch="86")], "8 GB of VRAM or more"),   # (Windows: the same card rules)
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
            self.assertIsNone(F.unsupported("cuda", True, 1, cards))   # (Windows + NVIDIA: as on Linux)
        for cards in ([card(12, arch="86"), card(12, 1, arch="86")], [card(8, arch="70"), card(8, 1, arch="70")], []):
            self.assertIsNone(F.unsupported("cuda", False, 2, cards))   # (two cards: the split)
        # (AMD on Linux: RDNA 3 / 3.5 / 4 from 12 GB - the offload path under 24 - or an APU whose shared memory
        #  leaves 24 GB)
        for cards in ([amd("gfx1100", 24)], [amd("gfx1201", 32)], [amd("gfx1030", 16), amd("gfx1100", 24, 1)],
                      [amd("gfx1151", 0.5, unified_memory=True, memory_gb=168)], [amd("gfx1201", 16)],
                      [amd("gfx1101", 12)], []):
            self.assertIsNone(F.unsupported("hip", False, 1, cards))
            self.assertIsNone(F.unsupported("hip", False, 1, cards, 32.0))
        # (a 12 GB card with 8 GB of RAM: the experts it does not hold need the RAM)
        self.assertIn("RAM and VRAM together", F.unsupported("hip", False, 1, [amd("gfx1101", 12)], 8.0))

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
        for a, part in ((args(gpus="0,1,2"), "one or two NVIDIA GPUs"),):
            with patch.object(maya.S, "gpus", side_effect=AssertionError("no GPU read before the refusal")), \
                    patch.object(maya, "WIN", False):
                out = refusal(lambda: maya.check_pc(a, "ik2"))
            self.assertIn(part, out)
            self.assertEqual(len([ln for ln in out.splitlines() if "[X]" in ln]), 1)
        with patch.object(maya, "WIN", True):   # (Windows: AMD only - NVIDIA runs it as on Linux)
            self.assertIn("Linux only", refusal(lambda: maya.check_pc(args(backend="hip"), "ik2")))

    def test_windows_vram_measured(self):
        """Windows + NVIDIA: the card as CUDA gets it (an RTX 3050 8 GB: 7.0 GiB free), measured, kept per PC, card
        and driver, read back while a Maya engine holds the card; Linux keeps nvidia-smi's figure."""
        g = dict(card(8, arch="86", name="NVIDIA GeForce RTX 3050"), pci="00000000:01:00.0")
        pc = {"backend": "cuda", "gpus": [g]}
        probe = {maya.S._pci_key(g["pci"]): (int(7.0 * 2**30), int(6.7 * 2**30), int(8.0 * 2**30))}
        with patch.object(maya, "WIN", True):
            room, how = maya.ik2_engine_vram(pc, probe=probe, saved={}, busy={})
            self.assertEqual(how, "measured")
            self.assertAlmostEqual(room, 7.0 + F.MARGIN_GIB, places=3)
            self.assertLess(F.plan(room, 32768)["tier_gib"], F.plan(8.0, 32768)["tier_gib"])
            saved = {"ik2|" + maya.hy4_room_key(g): {"ik2_free_gib": 6.9}}
            self.assertEqual(maya.ik2_engine_vram(pc, probe={}, saved=saved, busy={0: 5000}),
                             (6.9 + F.MARGIN_GIB, "measured"))
            self.assertEqual(maya.ik2_engine_vram(pc, probe={}, saved={}, busy={}), (8.0, ""))
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.ik2_engine_vram(pc, probe=probe, saved={}, busy={}), (8.0, ""))

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
        # (Linux, two cards where the largest does not hold every expert: both, the larger first - the split)
        with patch.object(maya, "ok"), patch.object(maya, "WIN", False):
            got = maya.choose_gpus(args(), [card(12, 0, arch="86"), card(16, 1, arch="89"), card(8, 2)],
                                   maya.IK2_MIN_CC, "ik2")
            self.assertEqual([g["index"] for g in got], [1, 0])
            got = maya.choose_gpus(args(), [card(12, 0), card(12, 1)], maya.IK2_MIN_CC, "ik2", before=[1])
            self.assertEqual([g["index"] for g in got], [1])          # (one, as its config ran)
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
        p = F.plan(24.0, 262144, "f16")   # (F16: twice the cache - the experts leave first, the cache stays on the card)
        self.assertTrue(p["layout"] == "offload" and not p["kv_streamed"])
        self.assertEqual(F.recommended_context(24.0, "f16"), 131072)

    def test_the_trained_million_keeps_the_cache_on_the_card(self):
        """1M on a 24 GB card: the experts leave VRAM first (a token reads the whole cache, ~8 experts a layer) - the
        offload path with the whole cache in VRAM, no KV streaming (Jupiter V100 as a 24 GB card: a tier of 2.9 GiB,
        19 tok/s at 1M deep vs 0.93 streamed); an AMD card the same.  An AMD APU (no offload path): every expert stays,
        the cache streams."""
        p = F.plan(24.0, 1048576)
        self.assertEqual((p["layout"], p["kv_streamed"]), ("offload", False))
        self.assertGreaterEqual(p["tier_gib"], F.OFF_TIER_MIN_GIB)
        self.assertEqual(F.verdict(24.0, 64.0, 1048576)[0], "runs")
        self.assertIn("the whole cache stays on the card", F.verdict(24.0, 64.0, 1048576)[1])
        self.assertEqual(F.verdict(24.0, 8.0, 1048576)[0], "does not fit")    # (the experts need the RAM)
        h = F.plan(24.0, 1048576, apu=True)
        self.assertTrue(h["resident"] and h["kv_streamed"])
        self.assertEqual(h["kv_ram_gib"], 14.0)
        self.assertEqual(F.verdict(24.0, 64.0, 1048576, apu=True)[0], "streamed")
        self.assertEqual(F.recommended_context(24.0), 262144)
        self.assertEqual(F.recommended_context(32.0), 262144)
        self.assertTrue(F.plan(32.0, 524288)["resident"])
        self.assertEqual(F.plan(32.0, 1048576)["layout"], "offload")
        self.assertIn(1048576, F.CONTEXTS)

    def test_small_cards_stream_only_where_the_cache_does_not_fit(self):
        self.assertFalse(F.plan(12.0, 262144)["kv_streamed"])   # (v1.1.73's offload layout as it was)
        p = F.plan(8.0, 1048576)
        self.assertTrue(p["kv_streamed"] and not p["resident"])
        self.assertGreater(p["ram_gib"], 16.0)

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


class TwoCards(unittest.TestCase):
    def test_plan_two(self):
        """The first card alone where it holds every expert; else the split - every expert on two 12 GB cards at 32K,
        two 8 GB cards with the rest in RAM; the RAM as one card's offload path (the prompts stream the second's)."""
        self.assertEqual(F.plan_two(24.0, 24.0, 32768)["gpus"], 1)
        p = F.plan_two(12.0, 12.0, 32768)
        self.assertEqual((p["gpus"], p["layout"]), (2, "split"))
        self.assertAlmostEqual(p["tier_gib"] + p["gpu1_gib"], F.EXPERTS_GIB, places=1)
        self.assertEqual(p["ram_gib"], F.plan(12.0, 32768)["ram_gib"])
        q = F.plan_two(8.0, 8.0, 32768)
        self.assertEqual(q["layout"], "split+ram")
        self.assertLess(q["tier_gib"] + q["gpu1_gib"], F.EXPERTS_GIB)
        self.assertEqual(F.verdict_two(12.0, 12.0, 64.0, 32768)[0], "fast")
        self.assertEqual(F.verdict_two(8.0, 8.0, 64.0, 32768)[0], "runs")
        self.assertEqual(F.verdict_two(8.0, 8.0, 8.0, 32768)[0], "does not fit")


class Picker(unittest.TestCase):
    def test_its_row_last_and_picked(self):
        got = {}

        def pick(title, intro, opts, default, yes, *x):
            got.update(rows=[o[0] for o in opts], about={o[0]: o[2] for o in opts})
            return len(opts) - 1
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]):
            fam = maya.choose_family(args(model=None, model_dir=None, gguf_dir=None, draft=None), {})
        self.assertEqual((got["rows"][-1], fam), ("Inkling 2 Mini", "ik2"))
        about = got["about"]["Inkling 2 Mini"]
        for part in ("thinkingmachines' Inkling-2-Mini", "UD-Q4_K_XL", "18 GB",
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
            self.assertEqual(maya.ik2_drafter_step(va(), d, 24, 65536, "fp8"), d / "dspark-Inkling-2-Mini-Q8_0.gguf")
            self.assertEqual(maya.ik2_drafter_step(va(), d, 24, 65536, "fp8", True),
                             d / "dspark-Inkling-2-Mini-Q8_0.gguf")                                  # (AMD: the GGUF too)
            said = []
            with patch.object(maya, "warn", said.append):
                self.assertEqual(maya.ik2_drafter_step(va(), d, 24, 262144, "fp8"),
                                 d / "dspark-Inkling-2-Mini-Q8_0.gguf")
                self.assertEqual(said, [])                 # (a 24 GB card at 262,144: the engine fits it, FP8 rows)
                self.assertEqual(maya.ik2_drafter_step(va(), d, 16, 1048576, "fp8"),
                                 d / "dspark-Inkling-2-Mini-Q8_0.gguf")
            self.assertIn("does not fit beside the model", " ".join(said))

    def drafter_run(self, d, amd_pc=False, good=True, fetched=True):
        """ik2_drafter_step with the download stubbed: (the file it names, the downloads it asked for as
        (repo, revision, folder, names))."""
        calls = []

        def offer(a, what, dd, files, m):
            calls.append((m["repo"], m["revision"], m["folder"], [f.name for f in files]))
            for f in files if fetched else []:
                with open(f, "wb") as fh:
                    fh.truncate(m["files"][f.name])    # (a sparse file of the published size)
            return fetched
        with patch.object(maya, "offer_download", offer), patch.object(maya, "sha256_checked", lambda p, w: good), \
                patch.object(maya, "ok", lambda *a: None), patch.object(maya, "warn", lambda *a: None):
            return maya.ik2_drafter_step(self.va(), d, 24, 65536, "fp8", amd_pc), calls

    def test_drafter_fresh_from_mayas_repo(self):
        """A model folder without a drafter: one file from Maya's repo at the pinned revision, folder drafter/ - the
        GGUF, on NVIDIA and on AMD."""
        ours = ("peasantsmith/Inkling-2-Mini-Maya-GGUF", "37027719327de5f574684b525ad88fe50c4ff222", "drafter")
        for amd_pc, name in ((False, "dspark-Inkling-2-Mini-Q8_0.gguf"), (True, "dspark-Inkling-2-Mini-Q8_0.gguf")):
            d, gguf = self.files()
            for f in ("dspark-Inkling-2-Mini-Q8_0.gguf", "mtp.safetensors"):
                (d / f).unlink(missing_ok=True)
            got, calls = self.drafter_run(d, amd_pc)
            self.assertEqual(calls, [ours + ([name],)])
            self.assertEqual(got, d / name)

    def test_drafter_from_an_earlier_setup(self):
        """An earlier setup's mtp.safetensors beside the model: the setup fetches the GGUF (the engine's faster
        drafting path; NVIDIA and AMD alike) and the old file stays on the disk; where the GGUF is not downloaded the
        engine drafts with the old file (the same size and sha256 - another is not taken).  The GGUF already here:
        no download."""
        d, gguf = self.files()
        old, new = d / "mtp.safetensors", d / "dspark-Inkling-2-Mini-Q8_0.gguf"
        new.unlink(missing_ok=True)
        with open(old, "wb") as fh:
            fh.truncate(635801416)
        for amd_pc in (False, True):                                     # (the download declined: the old file)
            got, calls = self.drafter_run(d, amd_pc=amd_pc, fetched=False)
            self.assertEqual((got, [c[3] for c in calls]), (old, [[new.name]]))
        got, calls = self.drafter_run(d, good=False, fetched=False)      # (its size, another sha256: not taken)
        self.assertEqual((got, [c[3] for c in calls]), (None, [[new.name]]))
        got, calls = self.drafter_run(d)                                 # (the GGUF fetched from Maya's repo)
        self.assertEqual((got, [c[2:] for c in calls]), (new, [("drafter", [new.name])]))
        self.assertEqual(old.stat().st_size, 635801416)                  # (the old file stays)
        self.assertEqual(self.drafter_run(d), (new, []))                 # (again: nothing to download)
        self.assertEqual(self.drafter_run(d, amd_pc=True), (new, []))    # (AMD: the GGUF too)
        old.unlink()
        self.assertEqual(self.drafter_run(d, amd_pc=True), (new, []))
        new.unlink()
        with open(old, "wb") as fh:
            fh.truncate(1000)                                            # (another size: not the drafter)
        got, calls = self.drafter_run(d, fetched=False)
        self.assertEqual((got, [c[3] for c in calls]), (None, [[new.name]]))
        self.assertTrue(old.exists())

    def test_drafter_room(self):
        """tools/ik2_fit.py's drafter room (maya-ik2's draft_room): 24 GB cards hold it up to about 200K tokens with the
        FP8 cache, 32 GB at 262K; it grows 4 KiB a token (its attention's logits in shared memory, no scratch a token)."""
        self.assertTrue(F.draft_fits(24, 131072, "fp8"))
        self.assertTrue(F.draft_fits(24, 262144, "fp8"))          # (as the engine: its plan's sum, FP8 rows)
        self.assertTrue(F.draft_fits(12, 32768, "fp8"))           # (a small card: the tier's room)
        self.assertTrue(F.draft_fits(24, 524288, "fp8"))          # (the offload path: the tier's room)
        self.assertTrue(F.draft_fits(32, 1048576, "fp8"))         # (... its drafts read the last positions)
        self.assertFalse(F.draft_fits(16, 1048576, "fp8"))        # (the cache streams)
        self.assertTrue(F.draft_fits(32, 262144, "fp8"))
        self.assertTrue(F.draft_fits(0, 262144, "fp8"))          # (not known: the engine measures)
        # (the engine's measured room: the scratch of 1152 B a token and its 288 MiB beside every expert - 32 GB at
        #  720,896 tokens: "the drafter left out ... 0.03 GiB short"; the tool said it fits)
        self.assertFalse(F.draft_fits(32, 720896, "fp8"))
        self.assertFalse(F.draft_fits(31.73, 720896, "fp8"))
        self.assertTrue(F.draft_fits(32, 524288, "fp8"))
        self.assertIsNone(F.drafting_off_reason("70", 32, 262144, "fp8"))
        self.assertAlmostEqual(F.draft_gib(262144) - F.draft_gib(0), 262144 * 4096 / 2**30, places=6)

    def test_drafting_off_reason(self):
        """Every NVIDIA class (Volta and Turing too, since v1.1.75), the offload path, the two-card split and AMD keep
        drafting; a streamed cache or a tier without the drafter's room turn it off."""
        self.assertIsNone(F.drafting_off_reason("70", 32, 65536, "fp8"))            # (a V100: its own window kernels)
        self.assertIsNone(F.drafting_off_reason("75", 24, 65536, "fp8"))            # (an RTX 20)
        self.assertIsNone(F.drafting_off_reason("70", 12, 32768, "fp8"))            # (one small card: auto measures it)
        self.assertIsNone(F.drafting_off_reason("70", 12, 65536, "fp8", gpus2=True))    # (two cards, the split: measured)
        self.assertIsNone(F.drafting_off_reason("86", 12, 65536, "fp8", gpus2=True))
        self.assertIsNone(F.drafting_off_reason("86", 24, 65536, "fp8", gpus2=True))    # (the first holds every expert)
        self.assertIn("no room", F.drafting_off_reason("86", 16, 1048576, "fp8", gpus2=True))
        self.assertIsNone(F.drafting_off_reason("86", 8, 32768, "fp8"))      # (NVIDIA offload: auto measures it)
        self.assertIsNone(F.drafting_off_reason("86", 24, 524288, "fp8"))
        self.assertIn("streams", F.drafting_off_reason("86", 16, 1048576, "fp8"))
        self.assertIsNone(F.drafting_off_reason("86", 24, 65536, "fp8"))
        self.assertIsNone(F.drafting_off_reason("120", 32, 262144, "fp8"))
        self.assertIsNone(F.drafting_off_reason("gfx1151", 150, 262144, "fp8", hip=True, apu=True))
        self.assertIsNone(F.drafting_off_reason(None, 24, 65536, "fp8", hip=True))         # (AMD, every expert in VRAM)
        self.assertIsNone(F.drafting_off_reason(None, 16, 65536, "fp8", hip=True))         # (AMD under 24 GB: the tier's room)
        self.assertIsNone(F.drafting_off_reason(None, 24, 524288, "fp8", hip=True))        # (AMD, a long context)
        self.assertIn("streams", F.drafting_off_reason(None, 16, 1048576, "fp8", hip=True))
        self.assertIsNone(F.drafting_off_reason(None, 0, 65536, "fp8"))      # (not known)

    def run_setup(self, gpu, ram=64.0, backend="cuda", **kw):
        """set_up_ik2 with its steps stubbed: what it hands the drafter step and the config (drafting, drafter)."""
        d, gguf = self.files()
        pc = {"gpus": [gpu], "backend": backend}
        seen = {}
        stub = lambda name, ret: patch.object(maya, name, lambda *a, **k: ret)   # noqa: E731
        drafter = patch.object(maya, "ik2_drafter_step",
                               lambda *a, **k: seen.update(amd=a[-1]) or seen.setdefault("drafter", d / "mtp.safetensors"))
        cfg = patch.object(maya, "write_config_ik2", lambda a, *r: seen.update(drafting=a.drafting, drafter=r[-1]))
        said = []
        with patch.object(maya, "check_pc", return_value=pc), patch.object(maya, "mem_gb", return_value=(ram, 10.0)),                 patch.object(maya, "choose_context", lambda a, prev, opts, rec: 65536),                 stub("ik2_dir", d), stub("tools_step", d), stub("build_step", {}), stub("ik2_model_step", gguf),                 stub("ik2_tokenizer_step", d), stub("ik2_vision_step", None), stub("ablate_step", None), drafter, cfg,                 patch.object(maya.IK2_EXE.__class__, "exists", lambda self: True),                 patch.object(maya, "ok", said.append), patch.object(maya, "warn", lambda *a: None),                 patch.object(maya, "step", lambda *a: None), patch.object(maya, "hip_unified_memory", lambda g: False), patch.object(maya, "WIN", False):
            maya.set_up_ik2(self.va(**kw), {})
        return seen, " ".join(said)

    def test_setup_drafting_off_where_it_does_not_run(self):
        """Where the engine cannot draft (a streamed cache, a tier without the drafter's room): no drafter downloaded,
        drafting off, the reason's line said - on a V100 as on any card."""
        why = F.drafting_off_reason("86", 16, 1048576, "fp8")
        self.assertIn("streams", why)
        p = patch.object(maya.IK2F, "drafting_off_reason", lambda *a, **k: why)
        p.start()
        self.addCleanup(p.stop)
        for gpu in (card(16), card(32, arch="70", name="Tesla V100")):
            seen, said = self.run_setup(gpu, drafting="on")
            self.assertEqual(seen["drafting"], "off", gpu)
            self.assertIsNone(seen["drafter"], gpu)
            self.assertIn("drafting off", said)

    def test_setup_drafting_kept_where_it_runs(self):
        """NVIDIA (every expert in VRAM, or the offload layout: auto measures it there; Volta and Turing too) and AMD:
        the drafter is fetched and drafting stays auto."""
        for gpu, backend in ((card(24), "cuda"), (card(32, arch="120"), "cuda"), (card(8), "cuda"),
                             (amd("gfx1100", 24), "hip"), (card(32, arch="70", name="Tesla V100"), "cuda"),
                             (card(24, arch="75", name="RTX 2080 Ti"), "cuda")):
            seen, said = self.run_setup(gpu, backend=backend)
            self.assertIsNone(seen["drafting"], gpu)
            self.assertEqual(seen["drafter"].name, "mtp.safetensors", gpu)
            self.assertEqual(seen["amd"], backend == "hip", gpu)        # (the drafter step picks the form by it)
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
        self.assertEqual(v, {"exe": str(maya.VEXE), "mmproj": str(d / "mmproj-Inkling-2-Mini-BF16.gguf"), "gpu": True})
        with patch.object(maya, "vision_encoder_ready", lambda *a: False):
            self.assertIsNone(maya.ik2_vision_step(self.va(), {}, {}, d, d))

    def test_vision_fresh_downloads_from_our_repo(self):
        """(a) Nothing on the disk: the download is Maya's repo, the pinned revision, the vision/ folder."""
        d, gguf = self.files()
        seen = []
        def offer(a, what, dd, files, m):
            seen.append((m["repo"], m["revision"], m["folder"], [f.name for f in files]))
            files[0].write_bytes(b"x")
            return True
        with patch.object(maya, "offer_download", offer), patch.object(maya, "sha256_checked", lambda *a: True),                 patch.object(maya, "vision_encoder_ready", lambda *a: True):
            v = maya.ik2_vision_step(self.va(), {}, {}, d, d)
        self.assertEqual(seen, [("peasantsmith/Inkling-2-Mini-Maya-GGUF", "7ab64bac7090c9f77df551ac0fac98119494a2a5",
                                 "vision", ["mmproj-Inkling-2-Mini-BF16.gguf"])])
        self.assertEqual(v["mmproj"], str(d / "mmproj-Inkling-2-Mini-BF16.gguf"))

    def test_vision_old_unsloth_name_not_downloaded_again(self):
        """(b) unsloth's mmproj-BF16.gguf already here (same size and sha256): no download, the config names it; a
        copy of another size or sha256 is not taken."""
        d, gguf = self.files()
        man = maya.ik2_vision_model()
        size, sha = 641805760, man["sha256"]["mmproj-Inkling-2-Mini-BF16.gguf"]
        old = d / "mmproj-BF16.gguf"
        with open(old, "wb") as fh:
            fh.truncate(size)                          # (a sparse file of the published size)
        calls = []
        def offer(a, what, dd, files, m):
            calls.append([f.name for f in files])
            return False
        for good in (True, False):
            calls.clear()
            with patch.object(maya, "offer_download", offer), patch.object(maya, "sha256_checked", lambda p, w: good),                     patch.object(maya, "vision_encoder_ready", lambda *a: True):
                v = maya.ik2_vision_step(self.va(), {}, {}, d, d)
            if good:
                self.assertEqual(calls, [])
                self.assertEqual(v["mmproj"], str(old))
            else:
                self.assertEqual(calls, [["mmproj-Inkling-2-Mini-BF16.gguf"]])
                self.assertIsNone(v)

    def test_the_config_with_pictures(self):
        d, gguf = self.files()
        tok = d / "maya-ik2" / "tokenizer"
        tok.mkdir(parents=True)
        v = {"exe": "/b/maya-vision", "mmproj": str(d / "mmproj-Inkling-2-Mini-BF16.gguf"), "gpu": True}
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
