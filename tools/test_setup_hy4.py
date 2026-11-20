"""tools/test_setup_hy4.py - the setup's Hy4-Flash (maya.py): the one table that switches its sizes on
(tools/hy4/download.json), where it is offered (NVIDIA; AMD on Linux), the layout and the context worked out from the VRAM
before any download, the n-gram table shared between its folders, its pack and its config.  No GPU, no network (the
one question the setup asks Hugging Face - is a layout's file uploaded: maya.hy4_on_hub - is answered here); its
vision tower (the mmproj file: downloaded with the model unless --no-vision, packed when it is in the folder, off
where the VRAM has no room for a picture; Images on the Models page); the draft block of Maya's sizes (Tencent's MTP
block in a file of its own, one for every size: taken where it fits in the VRAM beside the model, packed, --mtp).

    python -m unittest tools.test_setup_hy4 -v
"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import re
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = json.loads((ROOT / "tools" / "hy4" / "download.json").read_text(encoding="utf-8"))   # (as committed)
MANIFEST = dict(TABLE, revision="")                # the table before anything is uploaded: no revision
XS = MANIFEST["quants"]["Hy4-Flash-Maya-XS"]
HUB_COMMIT = "c" * 40
_hub = patch.object(maya, "hy4_on_hub", lambda *a, **k: HUB_COMMIT)   # (the 8 GB-card file: uploaded, unless a test says)


def setUpModule():
    _hub.start()


def tearDownModule():
    _hub.stop()
TRUNK = "Hy4-Flash-Maya-XS-IQ2_XXS-00001-of-00002.gguf"
SMALL = "Hy4-Flash-Maya-XS-8GB-00001-of-00002.gguf"            # (the 8 GB-card trunk, as the setup saves it)
SMALL_HUB = SMALL + "_file"                                   # (its name on the Hub: not in the Hub's size list)
SMALL_WAS = "Hy4-Flash-Maya-XS-8GB-IQ2_XXS_S-00001-of-00002.gguf"   # (a name it had here before)
OLD_TRUNK, OLD_NGRAM = "Hy4-Flash-Maya-XS-00001-of-00002.gguf", "Hy4-Flash-Maya-XS-00002-of-00002.gguf"
NGRAM = "Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf"


class Failed(Exception):
    pass


def fail(msg, hint=None):
    raise Failed(f"{msg} | {hint or ''}")


def args(**kw):
    base = dict(model=maya.HY4, hy4_quant=None, mm3_quant=None, ds4_quant=None, qwen_quant=None, model_dir=None,
                models_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, check=False, port=None, host=None,
                api_key=None, env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None, repack=False,
                parallel=None)
    base.update(kw)
    return SimpleNamespace(**base)


def card(gib, index=0):
    return {"index": index, "name": "GPU", "arch": "86", "vram_gb": float(gib)}


def manifest(revision="0" * 40, hub=True, only=("Hy4-Flash-Maya-XS",), **quants):
    """The table with a revision (published) and other sizes' entries replaced - as the tests of the table's switches
    want it unless they say otherwise: only Maya-XS with files (only=None: every size as committed), its 8 GB-card
    layout one that is asked from the Hub (hub; as committed it is at the table's revision)."""
    man = json.loads(json.dumps(MANIFEST))
    man["revision"] = revision
    for name in man["quants"]:
        if only is not None and name not in only:
            man["quants"][name] = {"files": {}}
    if hub:
        man["quants"]["Hy4-Flash-Maya-XS"]["layouts"]["8gb"]["published"] = "hub"
    man["quants"].update(quants)
    return man


def with_manifest(man):
    real = maya.read_json
    return patch.object(maya, "read_json", lambda p: man if Path(p) == maya.HY4_MANIFEST else real(p))


class Manifest(unittest.TestCase):
    def test_the_table(self):
        self.assertEqual(MANIFEST["name"], "Hy4-Flash")
        self.assertEqual(MANIFEST["repo"], "peasantsmith/Hy4-Flash-Maya-GGUF")
        self.assertIn("LICENSE.hy4-flash", MANIFEST["license"])
        self.assertTrue((ROOT / "LICENSE.hy4-flash").is_file())
        self.assertEqual(sorted(MANIFEST["quants"]), sorted(maya.HY4_QUANTS.values()))   # (every size has its entry)
        files = dict(XS["files"], **XS["layouts"]["8gb"]["files"], **MANIFEST["ngram"])
        self.assertEqual(len(files), 3)
        for name, (size, sha) in files.items():
            self.assertTrue(name.endswith((".gguf", ".gguf_file")) and size > 10**10 and len(sha) == 64, name)
        # the 8 GB-card trunk: .gguf_file on the Hub, a .gguf here - the names it had here before still taken
        self.assertEqual(list(XS["layouts"]["8gb"]["files"]), ["Maya-XS-8GB/" + SMALL_HUB])
        self.assertEqual(maya.hy4_local_name("Maya-XS-8GB/" + SMALL_HUB), SMALL)
        self.assertEqual(maya.hy4_local_name("Maya-XS/" + TRUNK), TRUNK)
        self.assertEqual(MANIFEST["was"][SMALL], [SMALL_WAS, "Hy4-Flash-Maya-XS-00001-of-00002.gguf"])
        self.assertEqual(len(MANIFEST["ngram"]), 1)                    # (one n-gram table for every size and layout)
        for plan in (XS["plan"], XS["layouts"]["8gb"]["plan"]):
            self.assertEqual(sorted(map(int, plan["need_gib"])), sorted(set(map(int, plan["need_gib"])) & set(maya.HY4_CONTEXTS)))
        self.assertNotIn("mmproj", json.dumps(MANIFEST["quants"]))     # (neither is read by the engine yet)

    def test_names_and_defaults(self):
        self.assertEqual(maya.HY4, "Hy4-Flash")
        self.assertEqual(maya.HY4_QUANTS, {"maya-xs": "Hy4-Flash-Maya-XS", "maya-s": "Hy4-Flash-Maya-S",
                                           "maya-m": "Hy4-Flash-Maya-M", "maya-l": "Hy4-Flash-Maya-L"})
        self.assertEqual(maya.HY4_CONTEXTS, [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576])   # (up to the model's 1M)
        self.assertEqual((maya.HY4_MIN_CC, maya.HY4_TESTED_CONTEXT, maya.HY4_MODEL_NAME), (70, 32768, "hy4-flash"))
        self.assertEqual((maya.HY4_SAMPLING, maya.HY4_EFFORT), ({"temperature": 0.9, "top_p": 1.0}, "high"))
        self.assertIn("hy4", maya.FAMILIES)
        self.assertEqual(maya.FAMILY_TITLE["hy4"], "Hy4-Flash")
        self.assertIn("maya-hy4", maya.ENGINE_PROCESSES)
        self.assertNotIn("hy4", maya.PARALLEL_FAMILIES)
        for t in maya.HY4_PACK_TOOLS:
            self.assertTrue((ROOT / t).is_file(), t)

    def test_exe(self):
        for win, name in ((False, "maya-hy4"), (True, "maya-hy4.exe")):
            with patch.object(maya, "WIN", win):
                maya.select_build_backend("cuda")
                self.assertEqual((maya.HY4_EXE.name, maya.HY4_EXE.parent.name), (name, "build"))
                self.assertEqual(maya.exe_for_target("maya-hy4"), maya.HY4_EXE)
                self.assertEqual(maya.bench_exe({"exe": "/b/maya-hy4", "args": ["--pack", "/p"]}), maya.HY4_EXE)
        maya.select_build_backend("cuda")

    def test_entry(self):
        m = maya.hy4_entry("maya-xs")
        self.assertEqual(list(m["files"]), [TRUNK, NGRAM])            # (the trunk first, the n-gram table last)
        self.assertEqual((m["folder"], m["dir"], m["layout"], m["ngram"]), ("Maya-XS", "Hy4-Flash-Maya-XS", "standard", NGRAM))
        self.assertNotIn("sources", m)
        self.assertAlmostEqual(m["download_gb"], (44786173984 + 63360616160) / 1e9)
        self.assertEqual(m["memory_gb"], 52)
        small = maya.hy4_entry("maya-xs", "8gb")
        self.assertEqual((small["folder"], small["dir"]), ("Maya-XS-8GB", "Hy4-Flash-Maya-XS-8GB"))
        self.assertEqual(small["files"][SMALL], 43565100064)
        self.assertEqual(small["hub_names"], {SMALL: SMALL_HUB})
        self.assertEqual(small["earlier"][SMALL][0], SMALL_WAS)
        self.assertNotIn("hub_names", m)
        self.assertEqual(small["source_folders"], {NGRAM: "Maya-XS"})   # (the table from the size's own folder)
        with with_manifest(manifest("abc")):
            small = maya.hy4_entry("maya-xs", "8gb")
            self.assertEqual(maya.file_url(small, SMALL), "https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/"
                                                          f"resolve/main/Maya-XS-8GB/{SMALL_HUB}")   # (or the commit the Hub named: Fit)
            self.assertEqual(maya.file_url(small, NGRAM), "https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/"
                                                          f"resolve/abc/Maya-XS/{NGRAM}")
        with with_manifest(manifest()), patch.object(maya, "fail", fail), \
                self.assertRaisesRegex(Failed, "not available yet"):      # (a size the table has no files for)
            maya.hy4_entry("maya-s")


class Published(unittest.TestCase):
    """One table switches the sizes on: a size is offered once "revision" is set and it has files with sha256."""

    def test_nothing_until_the_revision_is_pinned(self):
        with with_manifest(MANIFEST):
            self.assertEqual([k for k in maya.HY4_QUANTS if maya.hy4_published(k)], [])

    def test_as_committed(self):
        """The table of this release: Maya-XS, S, M and L on, at one pinned commit of the repo; the 8 GB-card layout
        at the same commit (the Hub is not asked)."""
        self.assertRegex(TABLE["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual([k for k in maya.HY4_QUANTS if maya.hy4_published(k)], ["maya-xs", "maya-s", "maya-m", "maya-l"])
        q = TABLE["quants"]
        self.assertFalse(any("published" in v for v in q.values()))   # (no size switched off)
        self.assertEqual({n[-1]: list(q["Hy4-Flash-Maya-" + n]["files"].items()) for n in ("S", "M", "L")}, {
            "S": [("Maya-S/Hy4-Flash-Maya-S-IQ2_S-00001-of-00002.gguf",
                   [52970756096, "80843b8df8e6ce4c1f4d768b42b717a6862ef1d91c06ae9083411b537d6b816e"])],
            "M": [("Maya-M/Hy4-Flash-Maya-M-IQ3_XXS-00001-of-00002.gguf",
                   [68391812128, "9ef12ab340a06b631069f99b9faccc9e99b686cf9d98b74bb872c4923e4732af"])],
            "L": [("Maya-L/Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf",
                   [89512229920, "5f47ce10c0cb6186ec34036cd5eae47b662b780933abca9f1db0db2770799c17"])]})
        self.assertEqual(TABLE["ngram"], {"Maya-XS/" + NGRAM: [
            63360616160, "5819578b52eb1cda0945a716e0575331406d976b6eebc74db8544b3d478ad244"]})
        # the memory they are for, as the model card says it, and the VRAM + RAM that hold each (the recommendation's)
        self.assertEqual([(q["Hy4-Flash-Maya-" + n]["for_gb"], q["Hy4-Flash-Maya-" + n]["memory_gb"]) for n in ("XS", "S", "M", "L")],
                         [([56, 64], 52), ([68, 80], 64), ([80, 96], 76), ([104, 128], 100)])
        # each size's plan: the engine's defaults, more VRAM for the parts that are not routed experts as the size grows
        plans = [q["Hy4-Flash-Maya-" + n]["plan"] for n in ("XS", "S", "M", "L")]
        for plan in plans:
            self.assertEqual((plan["chunk"], plan["reserve_gb"], sorted(map(int, plan["need_gib"]))),
                             (1024, 0.75, [4096, 8192, 32768]))
        for c in ("4096", "8192", "32768"):
            xs, sm, m, l = (plan["need_gib"][c] for plan in plans)
            self.assertTrue(xs < sm < m < l < xs + 1.5, c)
        self.assertEqual(plans[1]["need_gib"], {"4096": 8.82, "8192": 9.11, "32768": 9.97})
        self.assertEqual(plans[2]["need_gib"], {"4096": 9.84, "8192": 10.13, "32768": 10.99})
        self.assertEqual(plans[3]["need_gib"], {"4096": 9.97, "8192": 10.26, "32768": 11.12})
        self.assertNotIn("published", q["Hy4-Flash-Maya-XS"]["layouts"]["8gb"])
        with patch.object(maya, "hy4_on_hub", side_effect=AssertionError("the Hub was asked")):
            self.assertTrue(maya.hy4_published("maya-xs", "8gb"))
            small = maya.hy4_entry("maya-xs", "8gb")
        base = f"https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/resolve/{TABLE['revision']}/"
        self.assertEqual(maya.file_url(small, SMALL), base + "Maya-XS-8GB/" + SMALL_HUB)   # (saved as SMALL: a .gguf)
        self.assertEqual(maya.file_url(small, NGRAM), base + "Maya-XS/" + NGRAM)
        for key, folder, trunk in (("maya-s", "Maya-S", "Hy4-Flash-Maya-S-IQ2_S-00001-of-00002.gguf"),
                                   ("maya-m", "Maya-M", "Hy4-Flash-Maya-M-IQ3_XXS-00001-of-00002.gguf"),
                                   ("maya-l", "Maya-L", "Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf")):
            e = maya.hy4_entry(key)
            self.assertEqual((list(e["files"]), e["dir"], e["layout"]), ([trunk, NGRAM], "Hy4-Flash-" + folder, "standard"))
            self.assertEqual(maya.file_url(e, trunk), base + folder + "/" + trunk)
            self.assertEqual(maya.file_url(e, NGRAM), base + "Maya-XS/" + NGRAM)   # (ONE table: the XS folder's)
            self.assertEqual(e["earlier"][trunk], [trunk.replace("-IQ2_S", "").replace("-IQ3_XXS", "").replace("-IQ4_XS", "")])
            self.assertEqual(maya.hy4_vision_entry(key)["name"], "mmproj-Hy4-Flash-Maya-XS.gguf")
            self.assertEqual(list(maya.hy4_layouts(q["Hy4-Flash-" + folder])), ["standard"])   # (no small-card layout)
        self.assertAlmostEqual(maya.hy4_entry("maya-s")["download_gb"], 116.33, places=1)
        self.assertAlmostEqual(maya.hy4_entry("maya-m")["download_gb"], 131.75, places=1)
        self.assertAlmostEqual(maya.hy4_entry("maya-l")["download_gb"], 152.87, places=1)
        # a folder filled from the repo's own M or L set has the table under that set's name: taken, not downloaded
        for was in ("Hy4-Flash-Maya-M-IQ3_XXS-00002-of-00002.gguf", "Hy4-Flash-Maya-L-IQ4_XS-00002-of-00002.gguf",
                    "Hy4-Flash-Maya-S-IQ2_S-00002-of-00002.gguf"):
            self.assertIn(was, TABLE["was"][NGRAM])
        m = maya.hy4_entry("maya-xs")
        self.assertEqual(maya.file_url(m, TRUNK), f"https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/resolve/"
                                                  f"{TABLE['revision']}/Maya-XS/{TRUNK}")
        self.assertEqual(maya.file_url(m, NGRAM), f"https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/resolve/"
                                                  f"{TABLE['revision']}/Maya-XS/{NGRAM}")

    def test_a_size_is_one_entry_away(self):
        """What switches a size on, shown on Maya-S's entry: the trunk's size and sha256, its plan, no "published":
        false, and the revision."""
        s = json.loads(json.dumps(TABLE["quants"]["Hy4-Flash-Maya-S"]))
        self.assertTrue(maya.hy4_published("maya-s"))                 # (as committed: on)
        self.assertEqual(maya.hy4_fit(s, [card(16)], used={0: 400})["layout"], "standard")
        self.assertEqual(maya.hy4_fit(s, [card(12)], used={0: 400})["layout"], "standard")   # (cards from 12 GB)
        rel = next(iter(s["files"]))
        off = dict(s, published=False)
        with with_manifest(manifest("d" * 40, only=None, hub=False, **{"Hy4-Flash-Maya-S": off})):
            self.assertFalse(maya.hy4_published("maya-s"))
            with patch.object(maya, "fail", fail), self.assertRaisesRegex(Failed, "Maya-S is not available yet"):
                maya.hy4_entry("maya-s")
        empty = dict(s, files={rel: [0, ""]})
        with with_manifest(manifest(only=None, hub=False, **{"Hy4-Flash-Maya-S": empty})):
            self.assertFalse(maya.hy4_published("maya-s"))            # (no size and sha256)
        with with_manifest(manifest("d" * 40, only=None, hub=False)):
            self.assertEqual([k for k in maya.HY4_QUANTS if maya.hy4_published(k)],
                             ["maya-xs", "maya-s", "maya-m", "maya-l"])
            e = maya.hy4_entry("maya-s")
            trunk = "Hy4-Flash-Maya-S-IQ2_S-00001-of-00002.gguf"
            self.assertEqual((list(e["files"]), e["dir"]), ([trunk, NGRAM], "Hy4-Flash-Maya-S"))
            self.assertEqual(maya.file_url(e, trunk), "https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/resolve/"
                                                      + "d" * 40 + "/Maya-S/" + trunk)
            self.assertEqual(e["earlier"][trunk], ["Hy4-Flash-Maya-S-00001-of-00002.gguf"])

    def test_switched_off_whatever_it_lists(self):
        s = {"published": False, "about": "x", "memory_gb": 80,
             "files": {"Maya-S/Hy4-Flash-Maya-S-00001-of-00002.gguf": [7, "a" * 64]}}
        with with_manifest(manifest(**{"Hy4-Flash-Maya-S": s})):
            self.assertFalse(maya.hy4_published("maya-s"))
            self.assertEqual(maya.hy4_layouts(s), {})
        s["published"] = True                                         # (the one-line switch)
        with with_manifest(manifest(**{"Hy4-Flash-Maya-S": s})):
            self.assertTrue(maya.hy4_published("maya-s"))

    def test_a_layout_asked_from_the_hub(self):
        with with_manifest(manifest()):
            self.assertTrue(maya.hy4_published("maya-xs", "8gb"))     # (there)
            for answer in (False, None):                              # (not there; no answer)
                with patch.object(maya, "hy4_on_hub", lambda *a, **k: answer):
                    self.assertFalse(maya.hy4_published("maya-xs", "8gb"))
                    self.assertTrue(maya.hy4_published("maya-xs"))    # (the size itself: by the table alone)
        asked = []
        with with_manifest(MANIFEST), patch.object(maya, "hy4_on_hub", lambda *a, **k: asked.append(a)):
            self.assertFalse(maya.hy4_published("maya-xs", "8gb"))    # (nothing uploaded: the Hub is not asked)
        self.assertEqual(asked, [])

    def test_the_hubs_answer(self):
        """hy4_on_hub: one HEAD request, the redirect's headers against the table's size and sha256."""
        import email.message
        import urllib.error
        import urllib.request
        _hub.stop()
        try:
            rel, (size, sha) = next(iter(XS["layouts"]["8gb"]["files"].items()))

            def answer(code, **headers):
                msg = email.message.Message()
                for k, v in headers.items():
                    msg[k.replace("_", "-")] = v

                class Opener:
                    def open(self, req, timeout=None):
                        seen.append((req.get_method(), req.full_url))
                        if code is None:
                            raise OSError("no route")
                        raise urllib.error.HTTPError(req.full_url, code, "x", msg, None)
                return patch.object(urllib.request, "build_opener", lambda *a: Opener())
            for want, how in ((HUB_COMMIT, answer(302, X_Linked_Size=str(size), X_Linked_Etag=f'"{sha}"',
                                                  X_Repo_Commit=HUB_COMMIT)),
                              (False, answer(302, X_Linked_Size=str(size + 1), X_Linked_Etag=f'"{sha}"')),
                              (False, answer(302, X_Linked_Size=str(size), X_Linked_Etag='"' + "0" * 64 + '"')),
                              (False, answer(404)), (None, answer(503)), (None, answer(None))):
                seen = []
                maya._HY4_HUB.clear()
                with how:
                    self.assertEqual(maya.hy4_on_hub("o/r", rel, size, sha), want)
                    self.assertEqual(maya.hy4_on_hub("o/r", rel, size, sha), want)
                self.assertEqual(seen, [("HEAD", f"https://huggingface.co/o/r/resolve/main/{rel}")])   # (asked once)
        finally:
            maya._HY4_HUB.clear()
            _hub.start()

    def test_with_a_revision(self):
        with with_manifest(manifest()):
            self.assertEqual([k for k in maya.HY4_QUANTS if maya.hy4_published(k)], ["maya-xs"])
            self.assertTrue(maya.hy4_published("maya-xs", "8gb"))
            self.assertFalse(maya.hy4_published("maya-xs", "4gb"))
        s = {"about": "x", "memory_gb": 80, "files": {"Maya-S/Hy4-Flash-Maya-S-00001-of-00002.gguf": [7, "a" * 64]}}
        with with_manifest(manifest(**{"Hy4-Flash-Maya-S": s})):      # (a size switched on by the table alone)
            self.assertEqual([k for k in maya.HY4_QUANTS if maya.hy4_published(k)], ["maya-xs", "maya-s"])
            self.assertEqual(maya.hy4_entry("maya-s")["dir"], "Hy4-Flash-Maya-S")
        s["files"] = {"Maya-S/x.gguf": [7, ""]}                       # (no sha256 yet: not offered)
        with with_manifest(manifest(**{"Hy4-Flash-Maya-S": s})):
            self.assertFalse(maya.hy4_published("maya-s"))

    def test_the_servers_view(self):
        """serve/model_pull.py lists maya.hy4_entry(k) of the published sizes, with their download_gb."""
        with with_manifest(manifest()):
            got = {k: maya.hy4_entry(k) for k in maya.HY4_QUANTS if maya.hy4_published(k)}
        self.assertEqual(list(got), ["maya-xs"])
        self.assertAlmostEqual(got["maya-xs"]["download_gb"], 108.1, places=1)

    def test_choice(self):
        a = args()
        self.assertEqual(maya.choose_hy4_quant(args(hy4_quant="maya-m"), {}), "maya-m")
        with with_manifest(MANIFEST):                                 # (none uploaded: the smallest, with --model-dir)
            self.assertEqual(maya.choose_hy4_quant(a, {}), "maya-xs")
            self.assertEqual(maya.choose_hy4_quant(a, {"installer": {"hy4_quant": "maya-s"}}), "maya-s")
        with with_manifest(manifest()):
            self.assertEqual(maya.choose_hy4_quant(a, {}), "maya-xs")  # (one uploaded: no question)
        s = {"about": "experts 3 bits", "memory_gb": 80, "files": {"Maya-S/s-00001-of-00002.gguf": [70 * 10**9, "a" * 64]}}
        seen = {}

        def choose(title, intro, rows, default, auto):
            seen.update(rows=rows, default=default)
            return default
        for ram, want in ((64.0, "Maya-XS"), (128.0, "Maya-S")):      # (by this PC's memory: fit_recommendation)
            with with_manifest(manifest(**{"Hy4-Flash-Maya-S": s})), patch.object(maya, "choose", choose), \
                    patch.object(maya, "mem_gb", return_value=(ram, ram)), \
                    patch.object(maya.S, "gpus", return_value=[card(16)]):
                picked = maya.choose_hy4_quant(a, {})
            self.assertEqual(Q.shown("hy4", picked), f"Hy4-Flash · {want}")
            self.assertEqual(seen["rows"][0][1], "recommended")
            self.assertTrue(all(r[0].startswith("Hy4-Flash · ") for r in seen["rows"]))
        # the table as committed: XS, S, M and L, the one recommended by this PC's RAM + VRAM first; each row says the
        # PCs the size is for as the model card does (no speeds)
        for ram, vram, want in ((32.0, 8, "XS"), (64.0, 8, "XS"), (64.0, 12, "XS"), (64.0, 16, "S"), (48.0, 24, "S"),
                                (64.0, 24, "M"), (96.0, 16, "M"), (96.0, 24, "L"), (128.0, 32, "L"), (128.0, 8, "XS")):
            with with_manifest(TABLE), patch.object(maya, "choose", choose), \
                    patch.object(maya, "mem_gb", return_value=(ram, ram)), \
                    patch.object(maya.S, "gpus", return_value=[card(vram)]):
                picked = maya.choose_hy4_quant(a, {})
            self.assertEqual(picked, "maya-" + want.lower(), (ram, vram))
            rows = {r[0].split(" (")[0]: r for r in seen["rows"]}
            self.assertEqual(sorted(rows), ["Hy4-Flash · Maya-L", "Hy4-Flash · Maya-M", "Hy4-Flash · Maya-S", "Hy4-Flash · Maya-XS",
                                            "Hy4-Flash · Q4_K_M"])
            self.assertEqual(seen["rows"][0][:2], (Q.shown("hy4", picked, maya.hy4_entry(picked)["download_gb"]), "recommended"))
            self.assertIn("for PCs with 56-64 GB of RAM + VRAM", rows["Hy4-Flash · Maya-XS"][2])
            self.assertIn("IQ3_XXS (3 bits) on layers 24-41", rows["Hy4-Flash · Maya-S"][2])
            self.assertIn("for PCs with 68-80 GB of RAM + VRAM", rows["Hy4-Flash · Maya-S"][2])
            self.assertIn("IQ3_XXS (3 bits)", rows["Hy4-Flash · Maya-M"][2])
            self.assertIn("for PCs with 80-96 GB of RAM + VRAM", rows["Hy4-Flash · Maya-M"][2])
            self.assertIn("for PCs with 104-128 GB of RAM + VRAM", rows["Hy4-Flash · Maya-L"][2])
            self.assertFalse(any("tok/s" in r[2] or "tokens/s" in r[2] for r in seen["rows"]))


class Picker(unittest.TestCase):
    def rows(self, a=None, prev=None, man=None):
        got = {}

        def pick(title, intro, opts, default, yes, *x):
            got.update(rows=[o[0] for o in opts], about={o[0]: o[2] for o in opts}, default=default)
            return default
        a = a or args(model=None)
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]), \
                with_manifest(man or MANIFEST):
            got["family"] = maya.choose_family(a, prev or {})
        return got

    def test_not_listed_until_published(self):
        got = self.rows()
        self.assertNotIn("Hy4-Flash", got["rows"])
        self.assertEqual(got["rows"][-1], "Qwen-3.8-Flash-Next")       # (the other rows as before)
        self.assertEqual(got["family"], "glm")

    def test_listed_once_published(self):
        got = self.rows(man=manifest())
        self.assertEqual(got["rows"][-1], "Hy4-Flash")
        about = got["about"]["Hy4-Flash"]
        for part in ("Tencent's Hy4-flash", "Hy4-Flash-Maya-XS (108 GB)", "NVIDIA from V100 on", "Linux or Windows",
                     "pictures (its vision tower", "thinking High or Off", "tool calls", "Tencent's own Q4_K_M (AngelSlim",
                     "MTP drafting"):
            self.assertIn(part, about)
        self.assertNotIn("Max", about)
        self.assertNotIn("Maya-S", about)                             # (only the sizes that are uploaded)
        self.assertIn("or AMD on Linux (Strix Halo / Gorgon Halo, Radeon cards)", about)
        self.assertNotIn("Mac", about)
        self.assertIn("under Windows experimental", about)
        for win in (False, True):                                     # (AMD: Linux, and Windows - experimental)
            with patch.object(maya, "WIN", win):
                self.assertIn("Hy4-Flash", self.rows(args(model=None, backend="hip"), man=manifest())["rows"])

    def test_the_setup_before(self):
        prev = {"exe": "/b/maya-hy4", "args": ["--pack", "/p"], "installer": {"family": "hy4"}}
        got = self.rows(prev=prev)                                    # (set up from a folder before the upload)
        self.assertEqual((got["rows"][-1], got["family"]), ("Hy4-Flash", "hy4"))

    def test_named(self):
        self.assertEqual(maya.choose_family(args(model=maya.HY4), {}), "hy4")
        self.assertEqual(maya.choose_family(args(model=None, hy4_quant="maya-xs"), {}), "hy4")
        self.assertEqual(maya.choose_family(args(model=maya.MM3), {}), "minimax")

    def test_model_option(self):
        for typed, key in (("hy4-flash-maya-xs", "maya-xs"), ("HY4-FLASH-MAYA-S", "maya-s"),
                           ("Hy4-Flash-Maya-L", "maya-l"), ("Hy4-Flash", None)):
            a = args(model=typed)
            self.assertIsNone(maya.model_option(a), typed)
            self.assertEqual((a.model, a.hy4_quant), (maya.HY4, key), typed)
        a = args(model="hy4-flash", hy4_quant="maya-m")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.hy4_quant), (maya.HY4, "maya-m"))
        why = maya.model_option(args(model="hy4-flash-maya-xs", hy4_quant="maya-s"))
        self.assertIn("--hy4-quant says maya-s", why)
        why = maya.model_option(args(model="hy5"))
        self.assertIn("not a model Maya knows", why)
        self.assertIn("hy4-flash-maya-xs", why)
        a = args(model="minimax-maya-xs")                             # (the others as before)
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.mm3_quant, a.hy4_quant), (maya.MM3, "gguf-xs", None))

    def test_set_up_goes_to_its_steps(self):
        with patch.object(maya, "choose_family", return_value="hy4"), patch.object(maya, "configs", return_value=[]), \
                patch.object(maya.S, "model_named") as named, \
                patch.object(maya, "set_up_hy4", return_value=Path("x")) as steps:
            self.assertEqual(maya.set_up(args(), {}), Path("x"))
        named.assert_called_once_with("Hy4-Flash")
        steps.assert_called_once()

    def test_parallel_is_refused(self):
        with patch.object(maya, "fail", fail), self.assertRaisesRegex(Failed, "Hy4-Flash serves one request at a time"):
            maya.check_parallel(args(parallel=2), "hy4")
        maya.check_parallel(args(), "hy4")                            # (not asked: nothing)


class Gate(unittest.TestCase):
    """NVIDIA (CUDA) on Linux and Windows, V100 on; AMD (HIP) on Linux; no Mac engine yet."""

    def check(self, backend="cuda", gpus=None, win=False):
        said = []
        with patch.object(maya, "fail", fail), patch.object(maya, "WIN", win), patch.object(maya, "step", lambda *a: None), \
                patch.object(maya.sys, "platform", "win32" if win else "linux"), \
                patch.object(maya.S, "gpus", return_value=gpus or []), patch.object(maya, "say", said.append), \
                patch.object(maya, "warn", said.append), patch.object(maya.S, "is_wsl", return_value=False), \
                patch.object(maya, "check_hip_pc", side_effect=lambda a, **k: said.append(("hip", k)) or
                             {"backend": "hip", "gpus": []}), \
                patch.object(maya, "check_metal_pc", side_effect=AssertionError("no Mac check for Hy4-Flash")), \
                patch.object(maya, "choose_gpus", side_effect=lambda a, found, cc, fam: said.append(cc) or fail("stop")):
            try:
                maya.check_pc(args(backend=backend), "hy4")
            except Failed as e:
                return str(e), said
        return "", said

    def test_amd_on_linux_goes_to_the_amd_check(self):
        why, said = self.check("hip")
        self.assertEqual(why, "")
        self.assertEqual(said, [("hip", {"text_only": False})])       # (pictures: the tower runs on HIP too)

    def test_amd_on_windows_experimental(self):
        """Windows + AMD: said once to be experimental, then the AMD check as on Linux (no stop, no extra question)."""
        why, said = self.check("hip", win=True)
        self.assertEqual(why, "")
        self.assertEqual(len(said), 2)
        self.assertIn("Hy4-Flash on an AMD GPU under Windows is experimental: Maya's team has not tested it there",
                      said[0])
        self.assertIn("docs/AMD_MAYA.md", said[0])
        self.assertEqual(said[1], ("hip", {"text_only": False}))

    def test_mac(self):
        why, _ = self.check("metal")
        self.assertIn("GLM-5.3-Flash only on a Mac (not Hy4-Flash)", why)

    def test_nvidia_from_v100(self):
        for win in (False, True):
            why, said = self.check(gpus=[dict(card(32), arch="70"), dict(card(8, 1), arch="61")], win=win)
            self.assertEqual(why.split(" | ")[0], "stop")              # (it reached the GPU choice)
            self.assertIn(70, said)                                    # (min_cc: V100 on)
            self.assertTrue(any("too old" in str(s) for s in said))    # (the Pascal card)

    def test_cmake_builds_it_for_cuda_and_hip(self):
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        at = text.index("add_executable(maya-hy4 ")
        self.assertIn("if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)", text[at - 60:at])
        # (its vision tower's BF16 type on HIP: the compat header, not CUDA's own)
        self.assertTrue((ROOT / "include" / "strata" / "hip_compat" / "cuda_bf16.h").exists())


def halo(index=0):
    """A Strix Halo / Gorgon Halo's Radeon as setup's amd_gpus lists it on Linux (its VRAM: the BIOS's small share)."""
    return {"index": index, "name": "AMD Radeon (gfx1151)", "vram_gb": 0.5, "arch": "gfx1151", "vendor": "amd"}


def radeon(gib=20.0, index=0):
    return {"index": index, "name": "AMD Radeon RX 7900 XT", "vram_gb": float(gib), "arch": "gfx1100", "vendor": "amd"}


class Amd(unittest.TestCase):
    """AMD cards: no nvidia-smi reading; a unified-memory APU's memory is the PC's, counted once, less the system's
    share - and no more than Linux lets the GPU take."""

    def pc(self, ram, limit=None, win=False):
        return [patch.object(maya, "mem_gb", return_value=(float(ram), float(ram) - 4)), patch.object(maya, "WIN", win),
                patch.object(maya, "amd_gpu_limit_gib", return_value=limit),
                patch.object(maya, "gpu_used_mib", side_effect=AssertionError("nvidia-smi is not these cards'"))]

    def on(self, ram, limit=None):
        stack = contextlib.ExitStack()
        for p in self.pc(ram, limit):
            stack.enter_context(p)
        return stack

    def test_cards(self):
        self.assertTrue(maya.hy4_amd([halo()]))
        self.assertTrue(maya.hy4_amd([{"index": 0, "arch": "gfx1100", "vram_gb": 20.0}]))
        self.assertFalse(maya.hy4_amd([card(32)]))
        self.assertFalse(maya.hy4_amd([]))

    def test_radeon_cards(self):
        with self.on(64):
            self.assertAlmostEqual(maya.hy4_usable_gib([radeon()]), 17.0)             # (3 GB a card left to the desktop)
            self.assertAlmostEqual(maya.hy4_usable_gib([radeon(index=i) for i in range(4)], {0: 9000}), 68.0)
            self.assertIsNone(maya.hy4_apu_note([radeon()], XS, "Hy4-Flash-Maya-XS"))

    def test_apu_memory_is_the_pcs(self):
        with self.on(187):                                             # (a 192 GB Gorgon Halo as Linux counts it)
            self.assertAlmostEqual(maya.hy4_usable_gib([halo()]), 187 - 18.7)
        with self.on(61):                                              # (a 64 GB one: 8 GB kept free at least)
            self.assertAlmostEqual(maya.hy4_usable_gib([halo()]), 53.0)
        with self.on(123, limit=62.0):                                 # (Linux's own limit for the GPU, where lower)
            self.assertAlmostEqual(maya.hy4_usable_gib([halo()]), 62.0)
        with self.on(123, limit=180.0):
            self.assertAlmostEqual(maya.hy4_usable_gib([halo()]), 123 - 12.3)

    def test_which_sizes_a_halo_holds_whole(self):
        q = TABLE["quants"]
        sizes = ("Hy4-Flash-Maya-XS", "Hy4-Flash-Maya-M", "Hy4-Flash-Maya-L")
        for ram, whole in ((61, 1), (92, 2), (123, 3), (187, 3)):      # (64 / 96 / 128 / 192 GB PCs as Linux counts them)
            with self.on(ram):
                got = [maya.hy4_apu_note([halo()], q[n], n) for n in sizes]
            self.assertEqual([g[0] for g in got], [i < whole for i in range(3)], ram)
            for (fits, line), n in zip(got, sizes):
                self.assertIn(n, line)
                self.assertIn("no expert is computed on the CPU" if fits else "computes the others on the CPU", line)

    def test_linux_limit_said(self):
        with self.on(123, limit=62.0):
            fits, line = maya.hy4_apu_note([halo()], TABLE["quants"]["Hy4-Flash-Maya-L"], "Hy4-Flash-Maya-L")
        self.assertFalse(fits)
        self.assertIn("Linux lets the GPU take 62 GB of this PC's memory", line)
        self.assertIn("ttm.pages_limit", line)

    def test_fit(self):
        with self.on(61):
            fit = maya.hy4_fit(XS, [halo()])
        self.assertEqual((fit["layout"], fit["context"], fit["vision"], fit["args"]), ("standard", 32768, True, []))
        self.assertAlmostEqual(fit["usable"], 53.0)
        with self.on(64):                                              # (a 20 GB Radeon: the standard layout, pictures)
            fit = maya.hy4_fit(XS, [radeon()])
        self.assertEqual((fit["layout"], fit["vision"]), ("standard", True))
        self.assertAlmostEqual(fit["usable"], 17.0)

    def test_windows_apu_is_its_share(self):
        """Windows: the GPU's share (HIP's figure) less the desktop's 3 GB - not the share and the RAM together: the
        engine keeps within the dedicated memory there."""
        win_halo = dict(halo(), vram_gb=96.0)
        with contextlib.ExitStack() as stack:
            for p in self.pc(31, win=True):
                stack.enter_context(p)
            self.assertAlmostEqual(maya.hy4_usable_gib([win_halo]), 93.0)
            fits, line = maya.hy4_apu_note([win_halo], TABLE["quants"]["Hy4-Flash-Maya-M"], "Hy4-Flash-Maya-M")
            self.assertTrue(fits, line)
            fits, line = maya.hy4_apu_note([dict(win_halo, vram_gb=64.0)], TABLE["quants"]["Hy4-Flash-Maya-L"],
                                           "Hy4-Flash-Maya-L")
        self.assertFalse(fits)
        self.assertIn("Windows gives the GPU 61 GB of this PC's memory", line)

    def test_limit_read_from_sysfs(self):
        with patch.object(maya.sys, "platform", "win32"):
            self.assertIsNone(maya.amd_gpu_limit_gib())


class Need(unittest.TestCase):
    def test_the_tables_own_rows(self):
        plan = XS["plan"]
        self.assertEqual([maya.hy4_need_gib(plan, c) for c in (4096, 8192, 32768)], [8.68, 8.97, 9.83])
        self.assertEqual(maya.hy4_need_gib(XS["layouts"]["8gb"]["plan"], 8192), 6.8)

    def test_the_cache_formula_reproduces_them(self):
        only = lambda plan, c: dict(plan, need_gib={str(c): plan["need_gib"][str(c)]})   # noqa: E731
        plan, small = XS["plan"], XS["layouts"]["8gb"]["plan"]
        self.assertAlmostEqual(maya.hy4_need_gib(only(plan, 4096), 8192), 8.97, delta=0.02)
        self.assertAlmostEqual(maya.hy4_need_gib(only(plan, 4096), 32768), 9.83, delta=0.02)
        self.assertAlmostEqual(maya.hy4_need_gib(only(plan, 8192), 32768), 9.83, delta=0.02)
        self.assertAlmostEqual(maya.hy4_need_gib(only(small, 4096), 8192), 6.8, delta=0.02)
        self.assertAlmostEqual(maya.hy4_need_gib(only(plan, 8192), 4096), 8.68, delta=0.02)   # (below the table: back)

    def test_in_use_without_a_loaded_engines_own(self):
        """hy4_used_mib (the Models page's reading): the setup's own (gpu_used_mib), less what a Maya engine loaded
        now holds itself - by nvidia-smi's figure for it, else the reading from before its start, else as it reads."""
        used = {0: 7400, 1: 300}
        self.assertEqual(maya.hy4_used_mib(used=used, own={}), used)                      # (no engine: the setup's)
        self.assertEqual(maya.hy4_used_mib(used=used, own={0: 6800.0}), {0: 600.0, 1: 300})
        self.assertEqual(maya.hy4_used_mib(used=used, own={0: 9000.0}), {0: 0, 1: 300})
        self.assertEqual(maya.hy4_used_mib({0: 571}, used=used, own={0: None}), {0: 571, 1: 300})   # (Windows)
        self.assertEqual(maya.hy4_used_mib({0: 9999}, used=used, own={0: None}), {0: 7400, 1: 300})
        self.assertEqual(maya.hy4_used_mib(None, used=used, own={0: None}), used)         # (neither: capped later)
        self.assertEqual(maya.hy4_used_mib(used={}, own={0: 5.0}), {})                    # (no reading at all)
        with patch.object(maya, "gpu_used_mib", return_value={0: 7000}), \
                patch.object(maya, "maya_engine_mib", return_value={0: 6500.0}):
            self.assertEqual(maya.hy4_used_mib(), {0: 500.0})
        # the same sum as the setup's from there on: at most 1 GB a card counts
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8)], maya.hy4_used_mib({0: 571}, used={0: 7400}, own={0: None})),
                               8 - 571 / 1024)
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8)], maya.hy4_used_mib(used={0: 7400}, own={0: None})), 7.0)

    def test_the_engines_own_vram(self):
        """maya_engine_mib: Maya's engines in nvidia-smi's process list, by card; None where it gives no figure."""
        def smi(out):
            def run(cmd, **k):
                what = cmd[1]
                return SimpleNamespace(stdout=out["gpu" if what.startswith("--query-gpu") else "apps"])
            return patch.object(maya.subprocess, "run", run)
        gpus = "0, GPU-aaa\n1, GPU-bbb\n"
        with smi({"gpu": gpus, "apps": "GPU-aaa, 11, 6500, /opt/maya/build/maya-hy4\nGPU-aaa, 12, 300, /usr/bin/Xorg\n"
                                       "GPU-bbb, 13, 1000, /opt/maya/build/maya-hy4\nGPU-bbb, 14, 200, maya-glm-helper\n"}):
            self.assertEqual(maya.maya_engine_mib(), {0: 6500.0, 1: 1200.0})
        with smi({"gpu": gpus, "apps": "GPU-aaa, 11, [N/A], C:\\Maya, v1\\build\\maya-hy4.exe\n"
                                       "GPU-aaa, 12, [N/A], C:\\Windows\\explorer.exe\n"}):
            self.assertEqual(maya.maya_engine_mib(), {0: None})
        with smi({"gpu": gpus, "apps": "GPU-aaa, 12, 300, firefox\n"}):
            self.assertEqual(maya.maya_engine_mib(), {})
        with patch.object(maya.subprocess, "run", side_effect=OSError("no nvidia-smi")):
            self.assertEqual(maya.maya_engine_mib(), {})

    def test_grows_with_the_context(self):
        for plan in (XS["plan"], XS["layouts"]["8gb"]["plan"]):
            need = [maya.hy4_need_gib(plan, c) for c in maya.HY4_CONTEXTS]
            self.assertEqual(need, sorted(need))
            self.assertLess(need[5] - need[0], 4.0)                       # (128K)
            self.assertAlmostEqual(need[-1] - need[5], 13 * (1048576 - 131072) * 1280 / 2**30, delta=0.05)   # (1M: F16 rows)
        self.assertEqual(maya.hy4_cache_bytes(4096, 1024), 29 * 4096 * 1152 + 13 * (4096 * 1152 + 1025 * 512 + 4096 * 1024)
                         + 4096 * 12288)
        self.assertEqual(maya.hy4_need_gib({}, 8192), 0.0)

    def test_usable(self):
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8)], {0: 512}), 7.5)
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8)], {0: 6000}), 7.0)        # (more: a program to close)
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8)], {}), 7.6)               # (no reading: a desktop's share)
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8), card(8, 1)], {0: 512, 1: 0}), 15.5)
        with patch.object(maya, "gpu_used_mib", return_value={0: 1024}):
            self.assertAlmostEqual(maya.hy4_usable_gib([card(12)]), 11.0)


class Fit(unittest.TestCase):
    """The layout and the context, from the VRAM alone - before any download."""

    def fit(self, gib, used=400, **kw):
        gpus = gib if isinstance(gib, list) else [card(gib)]
        return maya.hy4_fit(XS, gpus, used={g["index"]: used for g in gpus}, **kw)

    def test_kv_streaming_offers_longer_contexts_where_the_ram_holds_the_rows(self):
        for gib in (8, 12, 16, 24, 32):                    # (no RAM figure: the VRAM alone, as before)
            f = self.fit(gib, 0)
            self.assertNotIn("streamed", f)
            self.assertNotIn("--kv-streaming", f["args"])
            # ... and what fitted without streaming is never streamed, whatever the RAM
            r = self.fit(gib, 0, ram_gb=128)
            self.assertEqual([c for c in r["contexts"] if c not in r.get("streamed", [])], f["contexts"], gib)
            self.assertEqual(self.fit(gib, 0, ram_gb=32).get("streamed"), None, gib)   # (a small RAM: no rows there)
        f = self.fit(8, ram_gb=64)                         # (an 8 GB card: 32K before; 1M's pooled keys do not fit)
        self.assertEqual((f["layout"], f["streamed"], f["context"]), ("8gb", [65536, 131072, 262144, 524288], 32768))
        self.assertIn("error", self.fit(8, ram_gb=64, ctx_wanted=1048576))
        f = self.fit(8, ram_gb=64, ctx_wanted=262144)
        self.assertEqual((f["kv_streaming"], f["kv"], f["args"][-2:]), (True, "q8", ["--kv-streaming", "auto"]))
        self.assertAlmostEqual(f["kv_ram"], 13 * 262144 * 672 / 2**30, places=3)
        self.assertLessEqual(f["need"], f["usable"])
        f = self.fit(16, 0, ram_gb=64, ctx_wanted=1048576)  # (a 16 GB card: the model's whole reach, F16 rows in RAM)
        self.assertEqual((f["layout"], f["kv_streaming"], f["kv"], f["args"]), ("standard", True, "f16", ["--kv-streaming", "auto"]))
        self.assertAlmostEqual(f["kv_ram"], 13 * 1048576 * 1152 / 2**30, places=3)
        f = self.fit(32, 0, ram_gb=64, ctx_wanted=1048576)  # (a card that holds it all: not streamed)
        self.assertNotIn("kv_streaming", f)
        # the streamed cache: a ring of the global layers' rows instead of every position's
        full, st = maya.hy4_cache_bytes(1048576, 1024), maya.hy4_cache_bytes(1048576, 1024, resident=32768)
        self.assertEqual(full - st, 13 * ((1048576 - 32768) * 1152 - 32768 * 4) - (64 * 2052 * 1160 + 1048576 * 4 + 1024))
        self.assertEqual(maya.hy4_cache_bytes(32768, 1024, resident=32768), maya.hy4_cache_bytes(32768, 1024))

    def test_8gb_card(self):
        f = self.fit(8)                                               # (a card that also drives the desktop)
        # (16K and 32K fit with the cache in 8-bit rows - the engine's --kv auto takes them by itself: no --kv)
        self.assertEqual((f["layout"], f["context"], f["recommended"], f["chunk"], f["kv"]),
                         ("8gb", 32768, 32768, 256, "q8"))
        self.assertEqual(f["args"], ["--chunk", "256", "--reserve-gb", "0.5", "--vram-experts", "0"])
        self.assertEqual(f["contexts"], [4096, 8192, 16384, 32768])
        self.assertNotIn("contexts_q8", f)                           # (nothing for a screen to say about the rows)
        f = maya.hy4_fit(XS, [card(8)], used={})                      # (nvidia-smi gave no reading)
        self.assertEqual((f["layout"], f["context"], f["chunk"]), ("8gb", 32768, 256))

    def test_the_cache_rows_as_the_engine_chooses_them(self):
        """--kv auto on every layout: F16 rows where the context fits with them, 8-bit rows where it fits only with
        them, nothing where it fits neither way."""
        rows = maya.hy4_kv_rows
        self.assertEqual((rows(6.8, 6.6, 7.3), rows(7.6, 7.0, 7.3), rows(7.9, 7.4, 7.3)), ("f16", "q8", None))
        self.assertEqual((rows(7.6, 7.0, 7.3, "f16"), rows(6.8, 6.6, 7.3, "q8"), rows(7.6, 7.4, 7.3, "q8")),
                         (None, "q8", None))               # (a config's own --kv: only that one)
        self.assertEqual(maya.HY4_KV_Q8_ROW, 672)          # (the engine's kv_q8_row_bytes(512, 64))
        small = XS["layouts"]["8gb"]["plan"]
        # measured on an 8 GB card (dry starts of the engine); a context the table does not list: the F16 figure
        # less what the 8-bit rows save - which gives the measured ones back
        self.assertEqual([maya.hy4_need_gib(small, c, True) for c in (16384, 32768)], [7.0, 7.1])
        f16_only = {k: v for k, v in small.items() if k != "need_gib_q8"}
        self.assertAlmostEqual(maya.hy4_need_gib(f16_only, 16384, True), 7.0, delta=0.06)
        self.assertAlmostEqual(maya.hy4_need_gib(f16_only, 32768, True), 7.1, delta=0.06)
        self.assertAlmostEqual(maya.hy4_need_gib(small, 8192, True), 6.8 - 0.21, delta=0.02)
        for c in maya.HY4_CONTEXTS:                        # (8-bit rows never take more than F16, at any context)
            for plan in (small, XS["plan"]):
                self.assertLess(maya.hy4_need_gib(plan, c, True), maya.hy4_need_gib(plan, c))
        self.assertEqual(maya.hy4_fit_need_gib(small, 32768, True), 7.1)                       # (measured: no margin)
        self.assertAlmostEqual(maya.hy4_fit_need_gib(small, 65536, True) - maya.hy4_need_gib(small, 65536, True),
                               maya.HY4_FIT_MARGIN_GIB)

    def test_contexts_count_on_8_bit_rows_only_where_f16_does_not_fit(self):
        f = self.fit(8, ctx_wanted=8192)                   # (fits with F16 rows: the plan it had, nothing changes)
        self.assertEqual((f["kv"], f["need"], f["chunk"]), ("f16", 6.8, 256))
        self.assertEqual(self.fit(8, ctx_wanted=4096)["kv"], "f16")
        f = self.fit(8, ctx_wanted=16384)
        self.assertEqual((f["layout"], f["kv"], f["need"], f["vision"], f["vision_small"]), ("8gb", "q8", 7.0, True, False))
        f = self.fit(8, 1024)                              # (a desktop with 1 GB in use: 16K, not 32K)
        self.assertEqual((f["context"], f["contexts"], f["kv"], f["vision"]), (16384, [4096, 8192, 16384], "q8", True))
        self.assertIn("error", self.fit(8, 1024, ctx_wanted=32768))
        for gib in (12, 16, 24, 32):                       # (cards every context fits with F16 rows: as before)
            f = self.fit(gib)
            self.assertEqual((f["kv"], f["need"]), ("f16", 9.83), gib)
        f = self.fit(10)                                   # (the standard layout on a 10 GB card: 8K was its largest)
        self.assertEqual((f["layout"], f["contexts"], f["kv"]), ("standard", [4096, 8192, 16384, 32768], "q8"))
        # never a context that fits neither way
        for gib in (6.5, 7, 8, 9, 10, 11, 12, 16):
            for used in (0, 400, 1024):
                f = self.fit(gib, used)
                if "error" in f:
                    continue
                plan = (XS["layouts"][f["layout"]] if f["layout"] != "standard" else XS)["plan"]
                for c in maya.HY4_CONTEXTS:
                    f16, q8 = maya.hy4_fit_need_gib(plan, c) <= f["usable"], maya.hy4_fit_need_gib(plan, c, True) <= f["usable"]
                    self.assertEqual(c in f["contexts"], f16 or q8, (gib, used, c))

    def test_pictures_cost_no_vram(self):
        """The engine lends a picture's room from its prompt buffers (idle while a picture is read), so pictures are
        on wherever the model runs, at the tower's largest size, and take nothing from the plan: the layout, the
        contexts, the cache's rows, the chunk and the VRAM counted are those of a start without pictures."""
        f = self.fit(8, 400)                               # (7.6 GB: 7.1 at 32K in 8-bit rows - before, a small picture)
        self.assertEqual((f["layout"], f["context"], f["kv"], f["vision"], f["vision_small"]), ("8gb", 32768, "q8", True, False))
        f = self.fit(8, 400, ctx_wanted=8192)              # (6.8: F16 rows, the largest picture)
        self.assertEqual((f["kv"], f["vision"], f["vision_small"]), ("f16", True, False))
        f = self.fit(8, 1024, ctx_wanted=16384)            # (7.0 GB: the model's own parts fill it - before, text only)
        self.assertEqual((f["kv"], f["vision"], f["vision_small"]), ("q8", True, False))
        # (no 8-bit rows for a picture's sake any more: 7.9 GB at 32K in F16 fits 8.1 GB)
        f = self.fit(8.5, 400, ctx_wanted=32768, layout="8gb")
        self.assertEqual((f["kv"], f["vision"], f["vision_small"]), ("f16", True, False))
        for gib in (12, 16, 24, 32):                       # (F16 rows and the largest picture: as before)
            f = self.fit(gib, 400)
            self.assertEqual((f["layout"], f["kv"], f["vision"], f["vision_small"]), ("standard", "f16", True, False), gib)
        seen = 0
        for gib in (6.5, 7, 8, 8.5, 9, 9.5, 10, 11, 12, 16, 24):
            for used in (0, 400, 1024):
                for ctx in (None, 4096, 8192, 32768, 131072):
                    on, off = self.fit(gib, used, ctx_wanted=ctx), self.fit(gib, used, ctx_wanted=ctx, pictures=False)
                    self.assertEqual("error" in on, "error" in off, (gib, used, ctx))
                    if "error" in on:
                        self.assertEqual(on, off, (gib, used, ctx))
                        continue
                    self.assertEqual((on["vision"], on["vision_small"], off["vision"]), (True, False, False), (gib, used, ctx))
                    self.assertEqual({k: v for k, v in on.items() if k != "vision"},
                                     {k: v for k, v in off.items() if k != "vision"}, (gib, used, ctx))
                    seen += 1
        self.assertGreater(seen, 60)

    def test_chunk_512_where_it_fits(self):
        """A small-card layout's chunk is doubled where that fits beside the model's own parts with the rows the
        context takes - pictures or not: the larger chunk's buffers lend a picture its room as the smaller one's do."""
        small, up = XS["layouts"]["8gb"]["plan"], maya.HY4_CHUNK_UP_GIB
        seen = set()
        for gib, used, ctx in ((8, 400, None), (8, 400, 4096), (8.5, 300, None), (9, 500, None), (9, 500, 8192),
                               (8, 0, None), (9.5, 900, 4096), (8, 0, 4096), (9.5, 0, 4096), (10, 0, 8192),
                               (11, 300, 8192), (11, 300, 32768), (12, 0, 32768), (8, 1024, 16384)):
            f = self.fit(gib, used, ctx_wanted=ctx, layout="8gb")
            need = maya.hy4_fit_need_gib(small, f["context"], f["kv"] == "q8")
            self.assertEqual(f["chunk"], 512 if need + up <= f["usable"] else 256, (gib, used, ctx))
            self.assertEqual(f["args"][:2], ["--chunk", str(f["chunk"])])
            self.assertTrue(f["vision"] and not f["vision_small"], (gib, used, ctx))
            seen.add(f["chunk"])
        self.assertEqual(seen, {256, 512})
        # (an 8 GB card at 4K with a light desktop: 6.5 + 0.95 of 7.6 GB - before, the picture took that room)
        f = self.fit(8, 400, ctx_wanted=4096)
        self.assertEqual((f["chunk"], f["kv"], f["vision"], f["vision_small"]), (512, "f16", True, False))
        f = self.fit(9, 500, ctx_wanted=8192)              # (6.8 + 0.95 of 8.5 GB)
        self.assertEqual((f["chunk"], f["kv"], f["vision"], f["vision_small"]), (512, "f16", True, False))
        # (the larger chunk never moves the rows to 8-bit: 7.1 + 0.95 does not fit 7.6 GB at 32K)
        f = self.fit(8, 400)
        self.assertEqual((f["chunk"], f["kv"]), (256, "q8"))
        f = self.fit(8, 400, ctx_wanted=16384)             # (7.0 in 8-bit rows of 7.6 GB: the chunk stays)
        self.assertEqual((f["chunk"], f["kv"]), (256, "q8"))

    def test_chunk_512_with_no_pictures_asked(self):
        """--no-vision: no picture is ever read, so the room goes to the chunk as it did."""
        f = self.fit(9, 500, ctx_wanted=8192, pictures=False)             # (8.5 GB free: 6.8 + 0.95 fits)
        self.assertEqual((f["chunk"], f["kv"], f["vision"], f["vision_small"]), (512, "f16", False, False))
        f = self.fit(8, 400, pictures=False)                              # (7.6 GB free at 32K: it does not)
        self.assertEqual((f["chunk"], f["kv"], f["vision"]), (256, "q8", False))
        for gib in (12, 24):                               # (a large card: the engine's own chunk either way)
            self.assertEqual(self.fit(gib, pictures=False)["chunk"], self.fit(gib)["chunk"])

    def test_larger_cards(self):
        for gib in (12, 16, 24, 32):
            f = self.fit(gib)
            self.assertEqual((f["layout"], f["context"], f["chunk"], f["args"]), ("standard", 32768, 1024, []), gib)
            self.assertIn(65536, f["contexts"])                       # (offered; the setup recommends up to 32K)
        self.assertEqual(self.fit(24)["contexts"], maya.HY4_CONTEXTS)

    def test_6gb_card(self):
        f = self.fit(6, 0)
        self.assertNotIn("layout", f)
        self.assertIn("needs about 6.5 GB of free VRAM", f["error"])
        self.assertIn("this card has 6.0 GB free", f["error"])
        self.assertIn("more VRAM", f["hint"])

    def test_two_cards(self):
        f = self.fit([card(8), card(8, 1)])
        self.assertEqual((f["layout"], f["context"], f["args"]), ("standard", 32768, []))
        self.assertAlmostEqual(f["usable"], 16 - 2 * 400 / 1024)
        self.assertIn("2 cards together have", self.fit([card(3), card(3, 1)])["error"])

    def test_context_asked(self):
        f = self.fit(24, ctx_wanted=131072)
        self.assertEqual((f["layout"], f["context"], f["recommended"]), ("standard", 131072, 32768))
        f = self.fit(8, ctx_wanted=65536)                             # (does not fit: the largest that does)
        self.assertIn("a context of 65536 tokens needs about", f["error"])
        self.assertEqual(f["hint"], "the largest context that fits here: --context 32768")
        f = self.fit(9, 0)                                            # (the standard layout fits at 4K / 8K ...)
        self.assertEqual((f["layout"], f["context"]), ("standard", 8192))
        f = self.fit(9, 0, ctx_wanted=32768)                          # (... a longer context: the small-card layout)
        self.assertEqual((f["layout"], f["context"]), ("8gb", 32768))

    def test_a_folders_layout(self):
        f = self.fit(9, 500, layout="standard")                       # (its trunk is the standard one: 8.5 GB free)
        self.assertIn("needs about 8.7 GB", f["error"])
        self.assertEqual(self.fit(24, layout="8gb")["layout"], "8gb")
        man, body = stand_ins()                                       # (small files under the trunks' names)
        xs = man["quants"]["Hy4-Flash-Maya-XS"]
        with tempfile.TemporaryDirectory() as d, with_manifest(man):
            self.assertIsNone(maya.hy4_layout_in(Path(d), "maya-xs"))
            for layout, files in (("standard", xs["files"]), ("8gb", xs["layouts"]["8gb"]["files"])):
                (Path(d) / OLD_TRUNK).write_bytes(body[next(iter(files))])   # (the name both had: told apart by the size)
                self.assertEqual(maya.hy4_layout_in(Path(d), "maya-xs"), layout)
            (Path(d) / OLD_TRUNK).unlink()
            (Path(d) / SMALL).write_bytes(body[next(iter(xs["layouts"]["8gb"]["files"]))])   # (under today's name)
            self.assertEqual(maya.hy4_layout_in(Path(d), "maya-xs"), "8gb")

    def test_m_and_l_as_committed(self):
        """Maya-M and Maya-L: the standard layout by their own plans - on a card from 12 GB, the contexts that fit;
        a card of 8 GB is told so before any download (they have no small-card layout)."""
        for name, at4k in (("M", 9.84), ("L", 9.97)):
            q = TABLE["quants"]["Hy4-Flash-Maya-" + name]
            fit = maya.hy4_fit(q, [card(12)], used={0: 400})
            self.assertEqual((fit["layout"], fit["args"], fit["context"], fit["vision"]), ("standard", [], 32768, True))
            self.assertAlmostEqual(maya.hy4_fit(q, [card(12)], 4096, used={0: 400})["need"], at4k)
            self.assertEqual(maya.hy4_fit(q, [card(32)], used={0: 0})["contexts"][-1], 1048576)
            small = maya.hy4_fit(q, [card(8)], used={0: 400})
            self.assertRegex(small["error"], rf"needs about {at4k:.1f} GB of free VRAM .* this card has 7\.6 GB free")
            self.assertGreater(maya.hy4_fit(q, [card(8), card(8, 1)], used={0: 400, 1: 0})["context"], 4096)

    def test_a_size_without_a_small_layout(self):
        q = {"files": XS["files"], "plan": XS["plan"]}
        self.assertIn("needs about 8.7 GB", maya.hy4_fit(q, [card(8)], used={0: 0})["error"])

    def test_the_step(self):
        said = []
        note = lambda *a: said.append(" ".join(map(str, a)))   # noqa: E731
        quiet = [patch.object(maya, "fail", fail), patch.object(maya, "ok", note),
                 patch.object(maya, "warn", note), patch.object(maya, "say", note)]
        with contextlib.ExitStack() as st:
            for p in quiet:
                st.enter_context(p)
            st.enter_context(patch.object(maya, "gpu_used_mib", return_value={0: 400}))
            f = maya.hy4_fit_step(args(), {"gpus": [card(8)]}, "maya-xs")
            self.assertEqual((f["layout"], f["context"]), ("8gb", 32768))
            self.assertTrue(any("8gb layout" in s and "chunks of 256" in s for s in said))
            # (the cache's rows are the engine's own choice at every start: nothing on screen names them)
            self.assertTrue(any(s.startswith("context: 32768 tokens (recommended") for s in said))
            self.assertFalse([s for s in said if re.search(r"8.bit|16.bit|F16|q8|--kv", s)], said)
            # (the model reaches 1M: one plain line on what keeps this card below it)
            self.assertEqual(len([s for s in said if "a context above 32K needs more VRAM than this card has free" in s]), 1)
            del said[:]
            f = maya.hy4_fit_step(args(), {"gpus": [card(8)]}, "maya-xs", prev_ctx=4096)   # (the setup before's)
            self.assertEqual(f["context"], 4096)
            self.assertTrue(any(s.startswith("context: 4096 tokens") for s in said))
            self.assertFalse([s for s in said if re.search(r"8.bit|16.bit|F16|q8|--kv", s)], said)
            f = maya.hy4_fit_step(args(context=65536), {"gpus": [card(24)]}, "maya-xs")
            self.assertEqual((f["layout"], f["context"]), ("standard", 65536))
            self.assertTrue(any("not been run end to end yet" in s for s in said))
            with self.assertRaisesRegex(Failed, "needs about 6.5 GB of free VRAM"):
                maya.hy4_fit_step(args(), {"gpus": [card(6)]}, "maya-xs")
            with self.assertRaisesRegex(Failed, "--context 32768"):
                maya.hy4_fit_step(args(context=131072), {"gpus": [card(8)]}, "maya-xs")
            with with_manifest(manifest()), self.assertRaisesRegex(Failed, "not available yet"):
                maya.hy4_fit_step(args(), {"gpus": [card(24)]}, "maya-s")

    def test_a_size_too_big_names_itself_and_the_smaller_one(self):
        """A size the card does not hold even at 4K: the line names the size, and the hint the largest smaller size
        that runs there with its exact command (none where no smaller one fits, or for the smallest)."""
        with patch.object(maya, "fail", fail), patch.object(maya, "ok", lambda *a: None), \
                patch.object(maya, "warn", lambda *a: None), patch.object(maya, "say", lambda *a: None), \
                patch.object(maya, "gpu_used_mib", return_value={0: 400}):
            with self.assertRaises(Failed) as e:
                maya.hy4_fit_step(args(), {"gpus": [card(8)]}, "maya-s")
            said = str(e.exception)
            self.assertIn("Hy4-Flash · Maya-S needs about", said)
            self.assertIn("this card has 7.6 GB free", said)
            self.assertIn("the smaller Hy4-Flash · Maya-XS fits this card (its 8 GB-card file): "
                          f"{maya.ME} --setup --model hy4-flash-maya-xs - or a card with more VRAM", said)
            self.assertEqual(maya.hy4_smaller_fit("maya-l", [card(8)]), ("maya-xs", "8gb"))
            self.assertIsNone(maya.hy4_smaller_fit("maya-xs", [card(8)]))          # (the smallest: none smaller)
            for quant in ("maya-s", "maya-xs"):                # (a card no size fits: the plain hint alone)
                with self.assertRaises(Failed) as e:
                    maya.hy4_fit_step(args(), {"gpus": [card(6)]}, quant)
                self.assertNotIn("the smaller", str(e.exception))
                self.assertIn("a card with more VRAM", str(e.exception))
            with with_manifest(manifest(revision="", only=None, hub=False)), self.assertRaises(Failed) as e:
                maya.hy4_fit_step(args(), {"gpus": [card(8)]}, "maya-l")      # (nothing uploaded: none offered)
            self.assertIn("Hy4-Flash · Maya-L needs about", str(e.exception))
            self.assertNotIn("the smaller", str(e.exception))

    def test_the_8gb_file_on_the_hub_or_not(self):
        """Both ways by the Hub's answer alone: there - used (from the commit the Hub named); not there - a plain
        line for a small card, before any build or download; a larger card is never asked; nor a folder with it."""
        said, asked = [], []
        note = lambda *a: said.append(" ".join(map(str, a)))   # noqa: E731
        man, body = stand_ins(manifest(revision="abc"))

        def hub(answer):
            def on_hub(repo, rel, size, sha, revision="main"):
                asked.append(rel)
                maya._HY4_HUB[(repo, rel, revision)] = answer
                return answer
            return patch.object(maya, "hy4_on_hub", on_hub)
        with contextlib.ExitStack() as st, tempfile.TemporaryDirectory() as tmp:
            for p in (patch.object(maya, "fail", fail), patch.object(maya, "ok", note), patch.object(maya, "warn", note),
                      patch.object(maya, "say", note), patch.object(maya, "gpu_used_mib", return_value={0: 400}),
                      with_manifest(man)):
                st.enter_context(p)
            st.callback(maya._HY4_HUB.clear)
            small, large = {"gpus": [card(8)]}, {"gpus": [card(24)]}
            with hub(False):
                with self.assertRaisesRegex(Failed, "the 8 GB-card file is not published yet .*7.6 GB of free VRAM.*"
                                                    "about 8.7 GB.*used by itself once it is on Hugging Face"):
                    maya.hy4_fit_step(args(), small, "maya-xs")
                self.assertEqual(asked, ["Maya-XS-8GB/" + SMALL_HUB])
                self.assertEqual(maya.hy4_fit_step(args(), large, "maya-xs")["layout"], "standard")
                self.assertEqual(len(asked), 1)                       # (a card the standard file fits: not asked)
                d = Path(tmp) / "x"                                   # (a folder that has the 8 GB-card trunk)
                d.mkdir()
                (d / SMALL).write_bytes(body["Maya-XS-8GB/" + SMALL_HUB])
                self.assertEqual(maya.hy4_fit_step(args(model_dir=str(d)), small, "maya-xs")["layout"], "8gb")
                self.assertEqual(maya.hy4_fit_step(args(), small, "maya-xs", have=lambda lay: d)["layout"], "8gb")
                self.assertEqual(len(asked), 1)
            with hub(None), self.assertRaisesRegex(Failed, "Hugging Face did not answer"):
                maya.hy4_fit_step(args(), small, "maya-xs")
            with hub(HUB_COMMIT):
                self.assertEqual(maya.hy4_fit_step(args(), small, "maya-xs")["layout"], "8gb")
                m = maya.hy4_entry("maya-xs", "8gb")
                self.assertEqual(maya.file_url(m, SMALL), "https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/"
                                                          f"resolve/{HUB_COMMIT}/Maya-XS-8GB/{SMALL_HUB}")
                self.assertEqual(maya.file_url(m, NGRAM), "https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/"
                                                          f"resolve/abc/Maya-XS/{NGRAM}")
        with with_manifest(MANIFEST), hub(False), patch.object(maya, "fail", fail), patch.object(maya, "ok", note), \
                patch.object(maya, "gpu_used_mib", return_value={0: 400}):
            n = len(asked)                                            # (nothing uploaded at all: --model-dir's way)
            self.assertEqual(maya.hy4_fit_step(args(), {"gpus": [card(8)]}, "maya-xs")["layout"], "8gb")
            self.assertEqual(len(asked), n)


def stand_ins(man=None):
    """The table with small stand-ins for the files (the same names): (the manifest, {repo path: bytes})."""
    man = json.loads(json.dumps(man or MANIFEST))
    body = {}

    def small(files, seed):
        for i, name in enumerate(list(files)):
            body[name] = bytes([seed + i]) * (1000 + seed)
            files[name] = [len(body[name]), hashlib.sha256(body[name]).hexdigest()]
    xs = man["quants"]["Hy4-Flash-Maya-XS"]
    small(xs["files"], 1)
    small(xs["layouts"]["8gb"]["files"], 2)
    small(man["ngram"], 3)
    return man, body


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.said = []
        self.patches = [patch.object(maya, "fail", fail), patch.object(maya, "step", lambda *a: None),
                        patch.object(maya, "warn", self.said.append), patch.object(maya, "ok", self.said.append),
                        patch.object(maya, "say", self.said.append), patch.object(maya, "rotational", return_value=False),
                        patch.object(maya, "configs", return_value=[])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def put(self, d, body, *paths):
        d.mkdir(parents=True, exist_ok=True)
        for p in paths:
            (d / maya.hy4_local_name(p)).write_bytes(body[p])   # (as the setup saves it)

    def test_folders(self):
        a = args()
        self.assertEqual(maya.hy4_dir(a, self.models, {}, "maya-xs"), self.models / "Hy4-Flash-Maya-XS")
        self.assertEqual(maya.hy4_dir(a, self.models, {}, "maya-xs", "8gb"), self.models / "Hy4-Flash-Maya-XS-8GB")
        own = Path(self.tmp.name) / "mine"
        self.assertEqual(maya.hy4_dir(args(model_dir=str(own)), self.models, {}, "maya-xs"), own.resolve())
        own.mkdir()
        prev = {"installer": {"hy4_quant": "maya-xs", "layout": "8gb", "model_dir": str(own)}}
        self.assertEqual(maya.hy4_dir(a, self.models, prev, "maya-xs", "8gb"), own)       # (the setup before's)
        self.assertEqual(maya.hy4_dir(a, self.models, prev, "maya-xs"), self.models / "Hy4-Flash-Maya-XS")
        places = maya.hy4_places(self.models, NGRAM)
        self.assertIn(self.models / "Hy4-Flash-Maya-XS" / NGRAM, places)
        self.assertIn(self.models / "Hy4-Flash-Maya-XS-8GB" / NGRAM, places)
        self.assertIn(self.models / "Hy4-Flash-Maya-L" / NGRAM, places)

    def test_unpublished_takes_the_folders_files_by_sha256(self):
        man, body = stand_ins()
        trunk, ngram = list(man["quants"]["Hy4-Flash-Maya-XS"]["files"])[0], list(man["ngram"])[0]
        d = Path(self.tmp.name) / "mine"
        with with_manifest(man):
            with self.assertRaisesRegex(Failed, "not published yet .*--model hy4-flash-maya-xs --model-dir"):
                maya.hy4_model_step(args(), d, "maya-xs", "standard", self.models)
            # the trunk alone: the file that is missing and its size, not the command just run
            self.put(d, body, trunk)
            with self.assertRaises(Failed) as e:
                maya.hy4_model_step(args(), d, "maya-xs", "standard", self.models)
            self.assertIn("missing: the n-gram table " + Path(ngram).name + " (", str(e.exception))
            self.assertNotIn("--model-dir", str(e.exception))
            self.put(d, body, trunk, ngram)
            files = maya.hy4_model_step(args(), d, "maya-xs", "standard", self.models)
            self.assertEqual([f.name for f in files], [TRUNK, NGRAM])
            self.assertEqual(set(json.loads((d / maya.VERIFIED).read_text())), {TRUNK, NGRAM})   # (remembered)
            self.assertTrue(any("never loaded into RAM" in s for s in self.said))
            (d / TRUNK).write_bytes(b"\xff" + body[trunk][1:])        # (one byte changed, the size kept)
            with self.assertRaisesRegex(Failed, "sha256 does not match"):
                maya.hy4_model_step(args(), d, "maya-xs", "standard", self.models)

    def test_download_list_and_the_shared_ngram_table(self):
        man, body = stand_ins(manifest())
        xs = man["quants"]["Hy4-Flash-Maya-XS"]
        trunk, small, ngram = list(xs["files"])[0], list(xs["layouts"]["8gb"]["files"])[0], list(man["ngram"])[0]
        asked = []

        def offer(a, quant, d, shards, m):
            asked.append((quant, [s.name for s in shards if maya.file_incomplete(m, s)],
                          [maya.file_url(m, s.name) for s in shards]))
            return False
        with with_manifest(man), patch.object(maya, "offer_download", offer):
            std = self.models / "Hy4-Flash-Maya-XS"
            self.assertIsNone(maya.hy4_model_step(args(), std, "maya-xs", "standard", self.models))
            quant, missing, urls = asked.pop()
            self.assertEqual(missing, [TRUNK, NGRAM])                 # (the trunk and the n-gram table: nothing else)
            self.assertTrue(quant.startswith("Hy4-Flash · Maya-XS"))
            self.assertEqual([u.split("/resolve/")[1] for u in urls],
                             ["0" * 40 + "/Maya-XS/" + TRUNK, "0" * 40 + "/Maya-XS/" + NGRAM])
            self.assertFalse(any("mmproj" in u or "draft" in u.lower() for u in urls))
            # the standard size is here; the 8 GB layout's folder then needs its trunk only: the table is linked
            self.put(std, body, trunk, ngram)
            self.assertEqual(len(maya.hy4_model_step(args(), std, "maya-xs", "standard", self.models)), 2)
            low = self.models / "Hy4-Flash-Maya-XS-8GB"
            self.assertIsNone(maya.hy4_model_step(args(), low, "maya-xs", "8gb", self.models))
            quant, missing, urls = asked.pop()
            self.assertEqual(missing, [SMALL])
            self.assertEqual([u.split("/resolve/")[1] for u in urls],
                             ["main/Maya-XS-8GB/" + SMALL_HUB, "0" * 40 + "/Maya-XS/" + NGRAM])
            self.assertTrue((low / NGRAM).samefile(std / NGRAM))      # (one copy on the disk)
            self.assertTrue(any("linked from" in s for s in self.said))
            self.put(low, body, small)
            self.assertEqual(len(maya.hy4_model_step(args(), low, "maya-xs", "8gb", self.models)), 2)
            self.assertEqual(asked, [])

    def test_files_under_their_earlier_names(self):
        # a folder filled before the names carried the quant type: linked to today's names by size and sha256, no download
        man, body = stand_ins(manifest())
        xs = man["quants"]["Hy4-Flash-Maya-XS"]
        trunk, small, ngram = list(xs["files"])[0], list(xs["layouts"]["8gb"]["files"])[0], list(man["ngram"])[0]
        self.assertEqual((man["was"][TRUNK], man["was"][SMALL], man["was"][NGRAM][:3]),
                         ([OLD_TRUNK], [SMALL_WAS, OLD_TRUNK], [OLD_NGRAM, "Hy4-Flash-Maya-M-00002-of-00002.gguf",
                                                                "Hy4-Flash-Maya-L-00002-of-00002.gguf"]))
        std, low = self.models / "Hy4-Flash-Maya-XS", self.models / "Hy4-Flash-Maya-XS-8GB"
        for d, t in ((std, trunk), (low, small)):
            d.mkdir(parents=True)
            (d / OLD_TRUNK).write_bytes(body[t])
        (std / OLD_NGRAM).write_bytes(body[ngram])
        with with_manifest(man), patch.object(maya, "offer_download", side_effect=AssertionError("asked to download")):
            files = maya.hy4_model_step(args(), std, "maya-xs", "standard", self.models)
            self.assertEqual([f.name for f in files], [TRUNK, NGRAM])
            self.assertTrue((std / TRUNK).samefile(std / OLD_TRUNK) and (std / NGRAM).samefile(std / OLD_NGRAM))
            # the 8 GB-card folder: its own trunk under the old name (the standard one's name: told apart by size),
            # the n-gram table from the standard folder under either name
            files = maya.hy4_model_step(args(), low, "maya-xs", "8gb", self.models)
            self.assertEqual([f.name for f in files], [SMALL, NGRAM])
            self.assertTrue((low / SMALL).samefile(low / OLD_TRUNK) and (low / NGRAM).samefile(std / OLD_NGRAM))
            self.assertEqual(maya.hy4_layout_in(low, "maya-xs"), "8gb")
        # a file of another size under the old name is not taken
        other = Path(self.tmp.name) / "other"
        other.mkdir()
        (other / OLD_TRUNK).write_bytes(body[small])
        with with_manifest(man), patch.object(maya, "offer_download", return_value=False):
            self.assertIsNone(maya.hy4_model_step(args(), other, "maya-xs", "standard", None))
        self.assertFalse((other / TRUNK).exists())

    def test_ngram_from_another_configs_folder(self):
        man, body = stand_ins(manifest())
        ngram = list(man["ngram"])[0]
        other = Path(self.tmp.name) / "elsewhere"
        self.put(other, body, ngram)
        cfg = Path(self.tmp.name) / "maya-hy4-flash-maya-xs.json"
        cfg.write_text(json.dumps({"exe": "maya-hy4", "args": [], "installer": {"family": "hy4", "model_dir": str(other)}}))
        d = self.models / "Hy4-Flash-Maya-XS"
        with with_manifest(man), patch.object(maya, "configs", return_value=[cfg]), \
                patch.object(maya, "offer_download", return_value=False):
            self.assertIsNone(maya.hy4_model_step(args(), d, "maya-xs", "standard", self.models))
        self.assertTrue((d / NGRAM).samefile(other / NGRAM))

    def test_spinning_disk(self):
        man, body = stand_ins()
        d = Path(self.tmp.name) / "mine"
        self.put(d, body, *list(man["quants"]["Hy4-Flash-Maya-XS"]["files"]), *list(man["ngram"]))
        with with_manifest(man), patch.object(maya, "rotational", return_value=True):
            maya.hy4_model_step(args(), d, "maya-xs")
        self.assertTrue(any("spinning hard disk" in s and "n-gram table" in s for s in self.said))

    def test_pack_command(self):
        d = Path(self.tmp.name) / "m"
        d.mkdir()
        files = [d / TRUNK, d / NGRAM]
        for f in files:
            f.write_bytes(b"x")
        ran = []

        def run(cmd, *a, **kw):
            ran.append(cmd)
            pack = Path(cmd[cmd.index("--out") + 1])
            (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
            for n in ("hy4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
                (pack / n).write_text("x")
        with patch.object(maya, "run", run):
            pack = maya.hy4_pack_step(args(), d, files)
            self.assertEqual(pack, d / "maya-hy4")
            cmd = ran[0]
            self.assertEqual(cmd[1:], [str(maya.ROOT / "tools" / "hy4" / "pack_gguf.py"), "--model", str(d), "--out",
                                       str(pack), "--mmproj", "none", "--mtp", "none", "--drafter", "none",
                                       "--files", TRUNK, NGRAM])
            self.assertNotIn("--engram-eps", cmd)
            self.assertEqual(json.loads((pack / maya.PACK_SOURCES).read_text())["tools"],
                             maya.S.source_hash(maya.HY4_PACK_TOOLS))
            maya.hy4_pack_step(args(), d, files)                      # (built: not again)
            self.assertEqual(len(ran), 1)
            maya.hy4_pack_step(args(repack=True), d, files)           # (--repack: again)
            self.assertEqual(len(ran), 2)
            (pack / "engram.bin").unlink()                            # (a part missing: again)
            maya.hy4_pack_step(args(), d, files)
            self.assertEqual(len(ran), 3)


class Config(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pack = self.root / "Hy4-Flash-Maya-XS" / "maya-hy4"
        self.pack.mkdir(parents=True)
        self.said = []
        self.patches = [patch.object(maya, "ROOT", self.root), patch.object(maya, "say", lambda *a: None),
                        patch.object(maya, "ok", self.said.append), patch.object(maya, "step", lambda *a: None),
                        patch.object(maya, "write_run_script", lambda p, port: self.root / "run.sh")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def write(self, a=None, old=None, gpus=(0,), fit=None, ctx=32768, backend=None):
        path = self.root / "maya-hy4-flash-maya-xs.json"
        if old is not None:
            path.write_text(json.dumps(old))
        pc = {"gpus": [{"index": i} for i in gpus], **({"backend": backend} if backend else {})}
        p = maya.write_config_hy4(a or args(), pc, {"lib_dirs": ["/cuda/lib"]}, self.pack, ctx, Path("/models"),
                                  self.pack.parent, quant="maya-xs", fit=fit)
        self.assertEqual(p, path)                                     # (maya_quants.config_stem: one config a size)
        return json.loads(p.read_text())

    def test_amd_config_says_its_backend(self):
        self.assertEqual(self.write(backend="hip")["backend"], "hip")
        self.assertNotIn("backend", self.write(backend="cuda"))

    def test_fresh(self):
        c = self.write()
        self.assertNotIn("backend", c)
        self.assertEqual(c["args"], ["--pack", str(self.pack), "--max-context", "32768"])
        self.assertEqual((c["sampling"], c["reasoning_effort"]), ({"temperature": 0.9, "top_p": 1.0}, "high"))
        self.assertEqual((c["model_name"], c["gpu"], c["port"]), ("hy4-flash", [0], 8080))
        self.assertEqual(Path(c["exe"]).name, maya.HY4_EXE.name)
        self.assertEqual(c["tokenizer"], str(self.pack / "tokenizer"))
        self.assertEqual(c["lib_dirs"], ["/cuda/lib"])
        inst = c["installer"]
        self.assertEqual({k: inst[k] for k in ("family", "hy4_quant", "quant", "layout")},
                         {"family": "hy4", "hy4_quant": "maya-xs", "quant": "Hy4-Flash-Maya-XS", "layout": "standard"})
        self.assertEqual(inst["model_dir"], str(self.pack.parent))
        self.assertEqual(maya.config_family(c), "hy4")
        self.assertEqual((Q.config_key(c), Q.model_defaults(c)), ("maya-xs", Q.MODEL_DEFAULTS["hy4"]))
        for absent in ("--parallel", "--vision", "--dspark", "--mtp", "--draft", "--gpus", "--chunk", "--reserve-gb"):
            self.assertNotIn(absent, c["args"])
        self.assertNotIn("vision", c)
        self.assertNotIn("env", c)                                    # (nothing that loads the n-gram table into RAM)
        self.assertNotIn("backend", c)

    def test_small_card_layout(self):
        fit = maya.hy4_fit(XS, [card(8)], used={0: 400})
        c = self.write(fit=fit, ctx=fit["context"])
        self.assertEqual(c["args"], ["--pack", str(self.pack), "--max-context", "32768", "--chunk", "256",
                                     "--reserve-gb", "0.5", "--vram-experts", "0"])   # (no --kv: the engine's auto)
        self.assertEqual(c["installer"]["layout"], "8gb")

    def test_several_gpus_as_the_server_numbers_them(self):
        c = self.write(gpus=(2, 0, 3))
        self.assertEqual(c["gpu"], [2, 0, 3])                         # (the physical cards: the server masks the others)
        self.assertNotIn("--gpus", c["args"])                         # (the engine takes every card it is shown)

    def test_again_keeps_what_it_does_not_ask(self):
        old = {"args": ["--pack", "/old", "--max-context", "4096", "--cpu-threads", "12", "--reserve-gb", "1.5",
                        "--profile", "/p.bin", "--chunk", "128", "--vram-experts", "0.5", "--gpus", "0,1"],
               "host": "0.0.0.0", "api_key": "k", "cors_origins": ["*"], "mcp_servers": {"a": 1},
               "env": {"MAYA_X": "1"}, "sampling": {"temperature": 0.2}, "reasoning_effort": "max",
               "installer": {"family": "hy4", "layout": "standard"}}
        c = self.write(args(env=["B=2"]), old)
        a = c["args"]
        self.assertEqual(a[:4], ["--pack", str(self.pack), "--max-context", "32768"])
        self.assertEqual((a[a.index("--cpu-threads") + 1], a[a.index("--reserve-gb") + 1], a[a.index("--profile") + 1]),
                         ("12", "1.5", "/p.bin"))
        self.assertNotIn("--gpus", a)                                 # (the cards: the engine takes those it is shown)
        # (the standard layout: a prompt chunk and an expert share set by hand stay, as the VRAM kept free does)
        self.assertEqual((a[a.index("--chunk") + 1], a[a.index("--vram-experts") + 1]), ("128", "0.5"))
        self.assertEqual((c["host"], c["api_key"], c["cors_origins"], c["mcp_servers"]), ("0.0.0.0", "k", ["*"], {"a": 1}))
        self.assertEqual(c["env"], {"MAYA_X": "1", "B": "2"})
        self.assertEqual((c["sampling"], c["reasoning_effort"]), ({"temperature": 0.9, "top_p": 1.0}, "high"))
        self.assertTrue(any("kept from the config before" in s and "--cpu-threads 12" in s for s in self.said))

    def test_a_layouts_reserve_is_its_own(self):
        fit = maya.hy4_fit(XS, [card(8)], used={0: 400})
        old = {"args": ["--pack", "/old", "--reserve-gb", "1.5", "--cpu-threads", "6"], "installer": {"layout": "standard"}}
        a = self.write(old=old, fit=fit, ctx=8192)["args"]
        self.assertEqual((a[a.index("--reserve-gb") + 1], a[a.index("--cpu-threads") + 1]), ("0.5", "6"))
        self.assertEqual(a.count("--reserve-gb"), 1)
        old = {"args": ["--pack", "/old", "--reserve-gb", "0.5", "--chunk", "256"], "installer": {"layout": "8gb"}}
        self.assertNotIn("--reserve-gb", self.write(old=old)["args"])   # (back on the standard layout: the engine's)

    def test_host_and_key_given(self):
        c = self.write(args(host="0.0.0.0", api_key="s3", port=9001))
        self.assertEqual((c["host"], c["api_key"], c["port"]), ("0.0.0.0", "s3", 9001))


class Steps(unittest.TestCase):
    def test_from_a_folder_with_the_files(self):
        """--model hy4-flash-maya-xs --model-dir <folder>: check, fit, build, files, pack, config - in that order,
        the layout the folder's trunk is, nothing measured afterwards."""
        man, body = stand_ins()
        xs = man["quants"]["Hy4-Flash-Maya-XS"]
        said, order = [], []
        with tempfile.TemporaryDirectory() as tmp:
            root, d = Path(tmp) / "maya", Path(tmp) / "mine"
            root.mkdir()
            d.mkdir()
            (d / OLD_TRUNK).write_bytes(body[list(xs["layouts"]["8gb"]["files"])[0]])   # (the 8 GB layout's trunk, under the name it had before)
            (d / NGRAM).write_bytes(body[list(man["ngram"])[0]])

            def run(cmd, *a, **kw):
                order.append("pack")
                pack = Path(cmd[cmd.index("--out") + 1])
                (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
                for n in ("hy4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
                    (pack / n).write_text("x")
            pc = {"backend": "cuda", "gpus": [card(24)], "archs": [86], "nvcc": "nvcc"}
            with with_manifest(man), patch.object(maya, "ROOT", root), patch.object(maya, "fail", fail), \
                    patch.object(maya, "step", lambda *a: None), patch.object(maya, "say", lambda *a: said.extend(a)), \
                    patch.object(maya, "ok", said.append), patch.object(maya, "warn", said.append), \
                    patch.object(maya, "check_pc", lambda a, fam: order.append(f"pc {fam}") or pc), \
                    patch.object(maya, "gpu_used_mib", return_value={0: 300}), \
                    patch.object(maya, "mem_gb", return_value=(16.0, 12.0)), \
                    patch.object(maya, "tools_step", lambda a: order.append("tools") or Path("llama")), \
                    patch.object(maya, "build_step", lambda a, pc, llama, target: order.append(target) or {"lib_dirs": []}), \
                    patch.object(maya, "rotational", return_value=False), patch.object(maya, "configs", return_value=[]), \
                    patch.object(maya, "run", run), patch.object(maya, "write_run_script", lambda p, port: "run"), \
                    patch.object(maya, "speed_calibration", side_effect=AssertionError("nothing to measure")), \
                    patch.object(maya, "prefill_calibration", side_effect=AssertionError("nothing to measure")), \
                    patch.object(maya, "offer_download", side_effect=AssertionError("no download")):
                a = args(model="hy4-flash-maya-xs", model_dir=str(d), no_vision=True)
                self.assertIsNone(maya.model_option(a))
                cfg_path = maya.set_up_hy4(a, {})
                c = json.loads(cfg_path.read_text())
            self.assertEqual(order, ["pc hy4", "tools", "maya-hy4", "pack"])
            self.assertEqual(cfg_path.name, "maya-hy4-flash-maya-xs.json")
            self.assertEqual(c["installer"]["layout"], "8gb")             # (the folder's trunk, on a 24 GB card too)
            self.assertEqual(c["installer"]["model_dir"], str(d.resolve()))
            self.assertEqual(c["args"][:4], ["--pack", str(d.resolve() / "maya-hy4"), "--max-context", "32768"])
            self.assertEqual(c["args"][4:], ["--chunk", "512", "--reserve-gb", "0.5", "--vram-experts", "0"])
        text = "\n".join(map(str, said))
        self.assertIn("the model reads text only", text)              # (no mmproj file in the folder: said)
        self.assertIn("drafting: off - the draft block needs about", text)   # (a small-card layout: no room for it)
        self.assertIn("slower: parts of the model are then read from the SSD", text)   # (24 + 16 GB against 52)
        self.assertNotIn("Ollama", text)

    def test_nothing_to_tune_or_measure(self):
        cfg = {"exe": "/b/maya-hy4", "args": ["--pack", "/p", "--max-context", "8192"], "installer": {"family": "hy4"}}
        self.assertFalse(maya.needs_speed_calibration(cfg))
        self.assertFalse(maya.needs_prefill_calibration(cfg))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "maya-hy4-flash-maya-xs.json"
            p.write_text(json.dumps(cfg))
            said = []
            with patch.object(maya, "ok", said.append), patch.object(maya, "warn", said.append), \
                    patch.object(maya.subprocess, "Popen", side_effect=AssertionError("no engine is started")), \
                    patch.object(maya.subprocess, "run", side_effect=AssertionError("no engine is started")):
                self.assertTrue(maya.calibrate_config(p))
                self.assertFalse(maya.speed_calibration(p, "asked for"))
            self.assertEqual(json.loads(p.read_text()), cfg)            # (the config as it was)
            self.assertTrue(any("nothing to tune" in s for s in said))
            self.assertTrue(any("no speed calibration" in s for s in said))

    def test_bench_prompt_through_its_template(self):
        with tempfile.TemporaryDirectory() as d, patch.object(maya, "fail", fail):
            with self.assertRaisesRegex(Failed, "chat template is missing"):
                maya.bench_chat("hy4", Path(d))

    def test_options(self):
        out = io.StringIO()
        with patch.object(sys, "argv", ["maya.py", "--help"]), contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit):
            maya.main()
        text = " ".join(out.getvalue().split())
        self.assertIn("--hy4-quant {maya-xs,maya-s,maya-m,maya-l,q4_k_m}", text)
        self.assertIn("hy4-flash-maya-xs", text)
        self.assertIn("Hy4-Flash serve one request at a time", text)



# ---------------------------------------------------------------- the launcher, the dashboard and Tencent's own Q4_K_M
# (the PR's tests: its name and setup ids, its configs recognised, the pinned download, the context recommended by the
#  smallest GPU's VRAM, the config the setup writes, the engine target, the Models page's settings, the two thinking levels)
SHARDS = ("Hy4-flash-Q4_K_M-00001-of-00002.gguf", "Hy4-flash-Q4_K_M-00002-of-00002.gguf")
MMPROJ = "mmproj-Hy4-flash-bf16.gguf"


MTP_HUB = "mtp/Hy4-Flash-mtp-BF16.gguf_file"
MTP_FILE = "Hy4-Flash-mtp-BF16.gguf"
DFLY_FILE = "Hy4-Flash-Maya-XS-drafter.dfly2"


class Drafts(unittest.TestCase):
    """A Maya size drafts with Tencent's MTP block from its own file: the table's entry, its VRAM, the setup's choice
    (on where the block fits beside the model's own parts and a picture's room; --drafting off, and a setup again
    after it, without), the pack's command and the config."""

    def test_the_table(self):
        self.assertEqual(TABLE["mtp"], {MTP_HUB: [7486183072, "a49e5ae439de9a51c680b9110563106743e8d579d91e4d4cb514a"
                                                                "527fa908908"]})
        e = maya.hy4_mtp_entry()
        self.assertEqual((e["name"], e["folder"], e["hub_names"], e["repo"]),
                         (MTP_FILE, "mtp", {MTP_FILE: "Hy4-Flash-mtp-BF16.gguf_file"}, TABLE["repo"]))
        self.assertEqual(e["published"], bool(TABLE.get("mtp_revision")))
        if TABLE.get("mtp_revision"):
            self.assertRegex(TABLE["mtp_revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(maya.file_url(e, MTP_FILE),
                             f"https://huggingface.co/{TABLE['repo']}/resolve/{TABLE['mtp_revision']}/{MTP_HUB}")
        with with_manifest({k: v for k, v in TABLE.items() if k != "mtp"}):
            self.assertIsNone(maya.hy4_mtp_entry())
            self.assertEqual(maya.hy4_mtp_gib(), 0.0)

    def test_vram(self):
        # the engine's own count: the BF16 block rounded to Q8_0 (0.56 of its bytes) and a prompt chunk of rows
        self.assertAlmostEqual(maya.hy4_mtp_gib(), (7486183072 * 0.56 + 1024 * 3072 * 20) / 2**30, places=6)
        self.assertTrue(3.9 < maya.hy4_mtp_gib() < 4.0)
        base = maya.hy4_config_need_gib("maya-xs", "standard", 32768)
        self.assertAlmostEqual(maya.hy4_config_need_gib("maya-xs", "standard", 32768, mtp=True),
                               base + maya.hy4_mtp_gib(), places=6)
        self.assertEqual(maya.hy4_config_need_gib("maya-xs", "standard", 32768, mtp=False), base)

    def test_the_packs_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            self.assertIsNone(maya.hy4_pack_mtp_file(pack))
            (pack / "tensors.txt").write_text("file 0 a.gguf\nfile 1 b.gguf\nt x F32 0 0 4 1\n")
            self.assertIsNone(maya.hy4_pack_mtp_file(pack))
            (pack / "tensors.txt").write_text("file 0 a.gguf\nfile 1 b.gguf\nfile 2 %s\nt x F32 0 0 4 1\n" % MTP_FILE)
            self.assertEqual(maya.hy4_pack_mtp_file(pack), MTP_FILE)

    def set_up(self, gib, block=True, published=False, prev=None, downloads=None, dfly=False, gpus=1, dfly_gib=None,
               small=False, used=300, choose=None, revision="", room=None, **kw):
        """One setup of Maya-XS from a folder with stand-in files on `gpus` cards of `gib`: (config, what was said,
        the pack's command).  dfly: the size has a DFly2 drafter in the table, its file in the folder.  small: the
        folder has the 8 GB-card trunk; used: the MiB nvidia-smi reads in use on each card; choose: the setup's
        questions answered by it (else none is asked: --yes); revision: the table's (the drafter published); room:
        Windows, with CUDA's readings of each card (cuda_room_bytes' free, free with two contexts, total)."""
        man, body = stand_ins(dict(MANIFEST, revision=revision))
        blob = b"m" * 777
        man["mtp"] = {MTP_HUB: [len(blob), hashlib.sha256(blob).hexdigest()]}
        man["mtp_revision"] = "d" * 40 if published else ""
        xs = man["quants"]["Hy4-Flash-Maya-XS"]
        said, ran = [], []
        root, d = Path(self.tmp.name) / "maya", Path(self.tmp.name) / "mine"
        root.mkdir(exist_ok=True)
        d.mkdir(exist_ok=True)
        if small:
            (d / SMALL).write_bytes(body[list(xs["layouts"]["8gb"]["files"])[0]])
        else:
            (d / TRUNK).write_bytes(body[list(xs["files"])[0]])
        (d / NGRAM).write_bytes(body[list(man["ngram"])[0]])
        if block:
            (d / MTP_FILE).write_bytes(blob)
        if dfly:
            dblob = b"f" * 555
            xs["drafter"] = {"Maya-XS/" + DFLY_FILE: [len(dblob), hashlib.sha256(dblob).hexdigest()]}
            (d / DFLY_FILE).write_bytes(dblob)

        def run(cmd, *a, **k):
            ran.append(cmd)
            pack = Path(cmd[cmd.index("--out") + 1])
            (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
            for n in ("engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
                (pack / n).write_text("x")
            m, f = cmd[cmd.index("--mtp") + 1], cmd[cmd.index("--drafter") + 1]
            (pack / "hy4.txt").write_text("x\n" + ("" if f == "none" else "dfly_layers 5\n"))
            (pack / "tensors.txt").write_text("file 0 a\nfile 1 b\n" + "".join(
                "file %d %s\n" % (2 + i, n) for i, n in enumerate(x for x in (m, f) if x != "none")))

        def offer(a, what, dd, files, m):
            (downloads if downloads is not None else []).append((what, [f.name for f in files], maya.file_url(m, files[0].name)))
            if downloads is None:
                raise AssertionError("no download")
            files[0].write_bytes(blob)
            return True
        pc = {"backend": "cuda", "gpus": [card(gib, i) for i in range(gpus)], "archs": [86], "nvcc": "nvcc"}
        gib_f = (lambda q: dfly_gib) if dfly_gib is not None else maya.hy4_dfly2_gib
        st = contextlib.ExitStack()
        if room is not None:                           # (Windows, CUDA's readings of the cards: WindowsRoom)
            for i, g in enumerate(pc["gpus"]):
                g["pci"] = f"00000000:0{i + 1}:00.0"
            for p in (patch.object(maya, "WIN", True), patch.object(maya, "maya_engine_mib", return_value={}),
                      patch.object(maya, "cuda_room_bytes",
                                   return_value={f"0000:0{i + 1}:00.0": room for i in range(gpus)})):
                st.enter_context(p)
        with st, with_manifest(man), patch.object(maya, "ROOT", root), patch.object(maya, "fail", fail), \
                patch.object(maya, "hy4_dfly2_gib", gib_f), \
                patch.object(maya, "step", lambda *a: None), patch.object(maya, "say", lambda *a: said.extend(a)), \
                patch.object(maya, "ok", said.append), patch.object(maya, "warn", said.append), \
                patch.object(maya, "check_pc", lambda a, fam: pc), \
                patch.object(maya, "gpu_used_mib", return_value={i: used for i in range(gpus)}), \
                patch.object(maya, "choose", choose or (lambda q, i, o, default, yes, *r: default)), \
                patch.object(maya, "mem_gb", return_value=(64.0, 60.0)), \
                patch.object(maya, "tools_step", lambda a: Path("llama")), \
                patch.object(maya, "build_step", lambda a, pc, llama, target: {"lib_dirs": []}), \
                patch.object(maya, "rotational", return_value=False), patch.object(maya, "configs", return_value=[]), \
                patch.object(maya, "run", run), patch.object(maya, "write_run_script", lambda p, port: "run"), \
                patch.object(maya, "hy4_mtp_gib", lambda chunk=1024: 3.95),                 patch.object(maya, "offer_download", offer):
            a = args(model="hy4-flash-maya-xs", model_dir=str(d), no_vision=True, **({"yes": False} if choose else {}),
                     **kw)
            maya.model_option(a)
            c = json.loads(maya.set_up_hy4(a, prev or {}).read_text())
        return c, "\n".join(map(str, said)), ran[-1] if ran else None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_on_where_the_block_fits(self):
        c, text, cmd = self.set_up(32)
        self.assertEqual(cmd[cmd.index("--mtp") + 1], MTP_FILE)
        self.assertIn("--mtp", c["args"])
        self.assertEqual((c["installer"]["drafting"], c["installer"]["draft_block"], c["installer"]["layout"]),
                         ("on", True, "standard"))
        self.assertIn("drafting: Tencent's MTP block drafts tokens and the model checks every one", text)
        self.assertNotIn("the same answers", text)
        self.assertNotIn("--draft", [x for x in c["args"] if x != "--mtp"])   # (the engine's own draft length)
        # --drafting off: the block stays in the pack (the Models page's switch), no --mtp; a setup again keeps the
        # choice; --drafting on drafts again
        c, text, cmd = self.set_up(32, prev=c, drafting="off")
        self.assertEqual(cmd[cmd.index("--mtp") + 1], MTP_FILE) if cmd else None
        self.assertNotIn("--mtp", c["args"])
        self.assertEqual((c["installer"]["drafting"], c["installer"]["draft_block"]), ("off", True))
        self.assertIn("drafting: off (--drafting on", text)
        c, text, _ = self.set_up(32, prev=c)
        self.assertNotIn("--mtp", c["args"])
        c, text, _ = self.set_up(32, prev=c, drafting="on")
        self.assertIn("--mtp", c["args"])

    def test_a_config_from_before_the_block(self):
        # (a config written when these files had no drafting: its "off" was no choice)
        prev = {"args": ["--pack", "/p", "--max-context", "32768"],
                "installer": {"family": "hy4", "hy4_quant": "maya-xs", "layout": "standard", "drafting": "off"}}
        c, _, _ = self.set_up(32, prev=prev)
        self.assertIn("--mtp", c["args"])

    def test_off_where_it_does_not_fit(self):
        # a 12 GB card: the model's own parts at 32K and the block together are more than the card has - the block is
        # neither asked for nor packed
        c, text, cmd = self.set_up(12, published=True)
        self.assertEqual(cmd[cmd.index("--mtp") + 1], "none")
        self.assertNotIn("--mtp", c["args"])
        self.assertEqual(c["installer"]["drafting"], "off")
        self.assertNotIn("draft_block", c["installer"])
        self.assertIn("drafting: off - the draft block needs about", text)
        self.assertIn("GB free on this PC's cards", text)

    def test_dfly2_table(self):
        # every Maya size has its own DFly2 drafter in the table (.dfly2 in its folder of the repo, at the table's
        # commit); its VRAM is the file's bytes and a little scratch
        for q in ("maya-xs", "maya-s", "maya-m", "maya-l"):
            e = maya.hy4_dfly2_entry(q)
            self.assertEqual((e["repo"], e["revision"], e["published"]), (TABLE["repo"], TABLE["revision"], True))
            self.assertRegex(e["name"], r"^Hy4-Flash-Maya-(XS|S|M|L)-drafter\.dfly2$")
            self.assertTrue(e["folder"].startswith("Maya-"))
            self.assertTrue(0.6 < maya.hy4_dfly2_gib(q) < 0.9)
        self.assertIsNone(maya.hy4_dfly2_entry(maya.HY4_Q4))
        self.assertEqual(maya.hy4_dfly2_gib(maya.HY4_Q4), 0.0)
        base = maya.hy4_config_need_gib("maya-xs", "standard", 32768)
        self.assertAlmostEqual(maya.hy4_config_need_gib("maya-xs", "standard", 32768, dfly2=True),
                               base + maya.hy4_dfly2_gib("maya-xs"), places=6)

    def test_dfly2_where_only_it_fits(self):
        # a 12 GB card: no room for the block, room for the DFly2 drafter - packed, --dfly2, drafting on
        c, text, cmd = self.set_up(12, published=True, dfly=True)
        self.assertEqual((cmd[cmd.index("--mtp") + 1], cmd[cmd.index("--drafter") + 1]), ("none", DFLY_FILE))
        self.assertIn("--dfly2", c["args"])
        self.assertNotIn("--mtp", c["args"])
        self.assertEqual((c["installer"]["drafting"], c["installer"]["dfly2"]), ("on", True))
        self.assertNotIn("draft_block", c["installer"])
        self.assertIn("drafting: Tencent's DFly2 drafter drafts tokens and the model checks every one", text)
        self.assertIn("(the MTP block: the draft block needs about", text)
        # --drafting off: the drafter stays in the pack, no --dfly2, a setup again keeps it off; --drafting on again
        c, text, _ = self.set_up(12, published=True, dfly=True, prev=c, drafting="off")
        self.assertNotIn("--dfly2", c["args"])
        self.assertIn("drafting: off (--drafting on, or Drafting on the Models page, drafts with the DFly2 drafter)",
                      text)
        c, _, _ = self.set_up(12, published=True, dfly=True, prev=c)
        self.assertNotIn("--dfly2", c["args"])
        c, _, _ = self.set_up(12, published=True, dfly=True, prev=c, drafting="on")
        self.assertIn("--dfly2", c["args"])
        # --drafting mtp: the block alone - off on this card, said
        c, text, cmd = self.set_up(12, published=True, dfly=True, drafting="mtp")
        self.assertEqual(cmd[cmd.index("--drafter") + 1], "none")
        self.assertNotIn("--dfly2", c["args"])
        self.assertIn("drafting: off - the draft block needs about", text)

    def test_both_fit(self):
        # where both fit, both are packed and the size's rule drafts until the speed calibration measures (once, at a
        # start); a setup again keeps what it measured; --no-speed-calibration: the rule, recorded as chosen
        c, text, cmd = self.set_up(32, dfly=True)
        self.assertEqual((cmd[cmd.index("--mtp") + 1], cmd[cmd.index("--drafter") + 1]), (MTP_FILE, DFLY_FILE))
        rule = maya.HY4_DRAFT_RULE["maya-xs"]
        self.assertIn("--dfly2" if rule == "dfly2" else "--mtp", c["args"])
        self.assertNotIn("--mtp" if rule == "dfly2" else "--dfly2", c["args"])
        self.assertEqual({k: c["installer"].get(k) for k in ("draft_choice", "drafter", "draft_block", "dfly2")},
                         {"draft_choice": "auto", "drafter": rule, "draft_block": True, "dfly2": True})
        self.assertIn("both fit here", text)
        self.assertTrue(maya.hy4_both_drafters(c) and maya.needs_speed_calibration(c))
        c["installer"]["speed_calibration"] = {"date": "d", "kept": "mtp" if rule == "dfly2" else "dfly2"}
        self.assertFalse(maya.needs_speed_calibration(c))
        c2, text, _ = self.set_up(32, dfly=True, prev=c)
        self.assertEqual(c2["installer"]["drafter"], c["installer"]["speed_calibration"]["kept"])
        self.assertEqual(c2["installer"]["speed_calibration"], c["installer"]["speed_calibration"])
        self.assertIn("the faster of the two on this PC", text)
        c3, _, _ = self.set_up(32, dfly=True, no_speed_calibration=True)
        self.assertEqual(c3["installer"]["speed_calibration"]["skipped"], "chosen")
        self.assertFalse(maya.needs_speed_calibration(c3))
        # two cards: the drafter is not taken, nothing to measure
        c4, _, cmd = self.set_up(32, dfly=True, gpus=2)
        self.assertEqual(cmd[cmd.index("--drafter") + 1], "none")
        self.assertFalse(maya.needs_speed_calibration(c4))

    def test_by_name(self):
        # --drafting mtp / dfly2: that one alone, packed alone, nothing measured
        c, _, cmd = self.set_up(32, dfly=True, drafting="dfly2")
        self.assertEqual((cmd[cmd.index("--mtp") + 1], cmd[cmd.index("--drafter") + 1]), ("none", DFLY_FILE))
        self.assertIn("--dfly2", c["args"])
        self.assertNotIn("--mtp", c["args"])
        self.assertFalse(maya.needs_speed_calibration(c))
        c, _, cmd = self.set_up(32, dfly=True, drafting="mtp")
        self.assertEqual((cmd[cmd.index("--mtp") + 1], cmd[cmd.index("--drafter") + 1]), (MTP_FILE, "none"))
        self.assertIn("--mtp", c["args"])
        self.assertEqual(c["installer"]["draft_choice"], "mtp")
        self.assertFalse(maya.needs_speed_calibration(c))

    def test_speed_calibration(self):
        # the A/B: MTP block first, then DFly2; the faster is kept, a tie keeps the size's rule
        c, _, _ = self.set_up(32, dfly=True)
        p = Path(self.tmp.name) / "maya-hy4-flash-maya-xs.json"
        for o, d, want in ((30.0, 36.0, "dfly2"), (36.0, 30.0, "mtp"), (30.0, 30.3, maya.HY4_DRAFT_RULE["maya-xs"])):
            p.write_text(json.dumps(c))
            seen = []

            def run(mtp, dfly, **kw):
                seen.append(("--mtp" in mtp["args"], "--dfly2" in mtp["args"], "--dfly2" in dfly["args"],
                             "--mtp" in dfly["args"]))
                return {"other": {"tok_s": o, "tokens": 512}, "dflash2": {"tok_s": d, "tokens": 512}}
            with patch.object(maya.CD, "run", run), patch.object(maya, "busy_skip", return_value=False), \
                    patch.object(maya, "say", lambda *a: None), patch.object(maya, "ok", lambda *a: None):
                self.assertTrue(maya.speed_calibration(p, "asked for"))
            out = json.loads(p.read_text())
            self.assertEqual(seen, [(True, False, True, False)])
            self.assertEqual((out["installer"]["drafter"], out["installer"]["speed_calibration"]["kept"]), (want, want))
            self.assertIn("--dfly2" if want == "dfly2" else "--mtp", out["args"])
            self.assertNotIn("--mtp" if want == "dfly2" else "--dfly2", out["args"])
            self.assertFalse(maya.needs_speed_calibration(out))

    def test_dfly2_not_where_it_cannot_run(self):
        # no room for the drafter either, or two cards (the engine runs it with its layers on one GPU): not taken, said
        c, text, cmd = self.set_up(12, published=True, dfly=True, dfly_gib=5.0)
        self.assertEqual(cmd[cmd.index("--drafter") + 1], "none")
        self.assertNotIn("--dfly2", c["args"])
        self.assertIn("; the DFly2 drafter needs about 5.0 GB", text)
        c, text, cmd = self.set_up(12, published=True, dfly=True, gpus=2)
        self.assertEqual(cmd[cmd.index("--drafter") + 1], "none")
        self.assertNotIn("--dfly2", c["args"])
        if "--mtp" not in c["args"]:
            self.assertIn("the DFly2 drafter runs on one GPU for now (this PC: 2)", text)

    def test_dfly2_where_the_engine_loads_it(self):
        # the engine loads the drafter only where its room changes nothing but the experts' places: the setup counts
        # it on the fit's figure with a doubled chunk's buffers, and with the margin where that figure is measured -
        # an 8 GB card driving a desktop at 8K (the engine: 8-bit rows beside it) does not take it
        g, q, card8 = maya.hy4_dfly2_gib("maya-xs"), maya.hy4_quant_table("maya-xs"), [{"index": 0, "vram_gb": 8.0}]
        f8 = maya.hy4_fit(q, card8, 8192, used={0: 552}, ram_gb=64)
        self.assertEqual((f8["layout"], f8["kv"]), ("8gb", "f16"))
        self.assertAlmostEqual(maya.hy4_dfly2_need_gib("maya-xs", f8), f8["need"] + g + maya.HY4_FIT_MARGIN_GIB)
        self.assertGreater(maya.hy4_dfly2_need_gib("maya-xs", f8), f8["usable"])
        f5 = maya.hy4_fit(q, card8, 5120, used={0: 552}, ram_gb=64)   # (a context the plan does not list)
        up = maya.HY4_CHUNK_UP_GIB if f5["chunk"] > maya.hy4_layouts(q)["8gb"]["plan"]["chunk"] else 0.0
        self.assertAlmostEqual(maya.hy4_dfly2_need_gib("maya-xs", f5), f5["need"] + up + g)
        self.assertAlmostEqual(maya.hy4_dfly2_need_gib("maya-xs", dict(f8, chunk=2 * f8["chunk"])),
                               f8["need"] + maya.HY4_CHUNK_UP_GIB + g + maya.HY4_FIT_MARGIN_GIB)
        # at 4K the larger chunk fits, the drafter beside it does not, beside the plan's chunk it does: with the
        # drafter counted the chunk stays the plan's (the drafts are the larger gain), without it it doubles
        base = maya.hy4_layouts(q)["8gb"]["plan"]["chunk"]
        self.assertEqual(maya.hy4_fit(q, card8, 4096, used={0: 552}, ram_gb=64)["chunk"], 2 * base)
        f4 = maya.hy4_fit(q, card8, 4096, used={0: 552}, ram_gb=64, drafter=g)
        self.assertEqual(f4["chunk"], base)
        self.assertLessEqual(maya.hy4_dfly2_need_gib("maya-xs", f4), f4["usable"])
        self.assertEqual(maya.hy4_fit(q, card8, 4096, used={0: 0}, ram_gb=64, drafter=g)["chunk"], base)

    def test_the_file(self):
        # not uploaded yet and not in the folder: no download, said; uploaded: asked from Maya's repo at its commit,
        # under its name there, saved under the name the pack reads
        c, text, cmd = self.set_up(32, block=False)
        self.assertEqual(cmd[cmd.index("--mtp") + 1], "none")
        self.assertNotIn("--mtp", c["args"])
        self.assertIn("the draft block's file is not in the model's folder (it is not published yet)", text)
        got = []
        c, text, cmd = self.set_up(32, block=False, published=True, downloads=got)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][1], [MTP_FILE])
        self.assertEqual(got[0][2], f"https://huggingface.co/{TABLE['repo']}/resolve/{'d' * 40}/{MTP_HUB}")
        self.assertIn("one file for every size", got[0][0])
        self.assertEqual(cmd[cmd.index("--mtp") + 1], MTP_FILE)
        self.assertIn("--mtp", c["args"])

    def test_linked_from_another_size(self):
        man, _ = stand_ins()
        blob = b"m" * 777
        man["mtp"] = {MTP_HUB: [len(blob), hashlib.sha256(blob).hexdigest()]}
        models = Path(self.tmp.name) / "models"
        (models / "Hy4-Flash-Maya-XS").mkdir(parents=True)
        (models / "Hy4-Flash-Maya-XS" / MTP_FILE).write_bytes(blob)
        d = models / "Hy4-Flash-Maya-L"
        d.mkdir()
        with with_manifest(man), patch.object(maya, "fail", fail), patch.object(maya, "ok", lambda *a: None), \
                patch.object(maya, "say", lambda *a: None), patch.object(maya, "configs", return_value=[]), \
                patch.object(maya, "offer_download", side_effect=AssertionError("no download")):
            f = maya.hy4_mtp_step(args(), d, "maya-l", models)
            self.assertEqual(f, d / MTP_FILE)
            self.assertEqual(f.read_bytes(), blob)
            self.assertIsNone(maya.hy4_mtp_step(args(), d, maya.HY4_Q4, models))   # (Tencent's own set: its block inside)
            (d / MTP_FILE).write_bytes(b"x" * 777)                                 # (damaged: the size is right)
            with self.assertRaises(Failed):
                maya.hy4_mtp_step(args(), d, "maya-l", models)


class FastDrafting(unittest.TestCase):
    """The opt-in faster drafting (--fast-drafting on): where the DFly2 drafter has no room beside the cache's F16
    rows but has it beside 8-bit rows, the setup offers the same context with --kv q8 - never by itself (the answers
    are not the default's), worked out from the card's free VRAM."""
    G = maya.hy4_dfly2_gib("maya-xs")                  # (the committed table's drafter: its real size)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def set_up(self, used, **kw):                      # (an 8 GB card at 8K, the 8 GB-card trunk and the drafter)
        return Drafts.set_up(self, 8, block=False, dfly=True, dfly_gib=self.G, small=True, used=used, context=8192,
                             revision=TABLE["revision"], **kw)

    def test_where_it_applies(self):
        q, card8 = maya.hy4_quant_table("maya-xs"), [card(8)]

        def fit(used, ctx=8192, cards=card8):
            f = maya.hy4_fit(q, cards, ctx, used={g["index"]: used for g in cards}, ram_gb=64, drafter=self.G)
            return f, maya.hy4_fast_drafting_fit("maya-xs", cards, f, used={g["index"]: used for g in cards}, ram_gb=64)
        # a desktop's ~0.4 GB in use: F16 rows fit, the drafter beside them does not, beside 8-bit rows it does
        f, fast = fit(400)
        self.assertEqual((f["kv"], maya.hy4_dfly2_need_gib("maya-xs", f) > f["usable"]), ("f16", True))
        self.assertEqual((fast["kv"], fast["context"], fast["layout"], fast["fast_drafting"], fast["f16_room"]),
                         ("q8", 8192, "8gb", True, False))
        self.assertEqual(fast["args"][-2:], ["--kv", "q8"])
        self.assertEqual(fast["chunk"], maya.hy4_layouts(q)["8gb"]["plan"]["chunk"])   # (the drafter's room first)
        self.assertLessEqual(maya.hy4_dfly2_need_gib("maya-xs", fast), fast["usable"])
        self.assertNotIn("--kv", f["args"])                                            # (the default: the engine's auto)
        # where these sums find the drafter room beside the F16 rows too (a free card; 4K): marked - the setup asks
        # nothing there (the default claims the drafter), and takes it only by name
        self.assertTrue(fit(76)[1]["f16_room"])
        self.assertTrue(fit(400, 4096)[1]["f16_room"])
        # not where 8-bit rows do not make the room either (more in use), at a context whose default rows are 8-bit
        # already, or on two cards
        self.assertIsNone(fit(552)[1])
        self.assertEqual(fit(400, 16384)[0]["kv"], "q8")
        self.assertIsNone(fit(400, 16384)[1])
        self.assertIsNone(fit(400, cards=[card(8), card(8, 1)])[1])
        # the standard layout on larger cards: the MTP block or the drafter drafts there with F16 rows
        for gib in (12, 16, 24):
            f = maya.hy4_fit(q, [card(gib)], None, used={0: 400}, ram_gb=64, drafter=self.G)
            self.assertIsNone(maya.hy4_fast_drafting_fit("maya-xs", [card(gib)], f, used={0: 400}, ram_gb=64), gib)
        self.assertIsNone(maya.hy4_fast_drafting_fit(maya.HY4_Q4, card8, f))            # (Tencent's own set: no drafter)
        # the rows asked for: only those counted and written (the engine's --kv)
        f8 = maya.hy4_fit(q, card8, 8192, used={0: 400}, ram_gb=64, kv="q8")
        self.assertEqual((f8["kv"], f8["args"][-2:]), ("q8", ["--kv", "q8"]))
        self.assertAlmostEqual(f8["need"], maya.hy4_fit_need_gib(maya.hy4_layouts(q)["8gb"]["plan"], 8192, True))

    def test_the_default_says_it(self):
        # --yes (and no flag): the default start - F16 rows, no drafter - and the drafting line names the faster one
        c, text, cmd = self.set_up(400)
        self.assertEqual(cmd[cmd.index("--drafter") + 1], "none")
        self.assertNotIn("--kv", c["args"])
        self.assertNotIn("--dfly2", c["args"])
        self.assertNotIn("fast_drafting", c["installer"])
        self.assertIn("with the cache in 8-bit rows it fits at 8K: faster drafting, slightly different wording than "
                      "the default (./maya.sh --setup --fast-drafting on)".replace("./maya.sh", maya.ME), text)
        # a free card: the drafter fits beside the F16 rows - the default drafts, nothing offered
        c, text, _ = self.set_up(76)
        self.assertEqual(("--dfly2" in c["args"], "--kv" in c["args"]), (True, False))
        self.assertNotIn("fast-drafting", text)

    def test_on(self):
        c, text, cmd = self.set_up(400, fast_drafting="on")
        self.assertEqual(cmd[cmd.index("--drafter") + 1], DFLY_FILE)
        self.assertEqual((c["args"][c["args"].index("--kv") + 1], "--dfly2" in c["args"]), ("q8", True))
        self.assertEqual(c["args"][c["args"].index("--chunk") + 1], "256")
        self.assertEqual((c["installer"]["fast_drafting"], c["installer"]["drafting"], c["installer"]["drafter"]),
                         (True, "on", "dfly2"))
        self.assertIn("faster drafting: on - the cache in 8-bit rows at this context, so the DFly2 drafter fits: "
                      "faster drafting, slightly different wording than the default", text)
        # a setup again keeps it (with drafting turned off on the Models page too: it is for the drafter); off goes back
        again, text, _ = self.set_up(400, prev={**c, "installer": {**c["installer"], "drafting": "off"}})
        self.assertEqual((again["args"][again["args"].index("--kv") + 1], "--dfly2" in again["args"]), ("q8", True))
        self.assertIn("faster drafting: on (as set up before)", text)
        off, _, _ = self.set_up(400, prev=c, fast_drafting="off")
        self.assertEqual(("--kv" in off["args"], "--dfly2" in off["args"], "fast_drafting" in off["installer"]),
                         (False, False, False))

    def test_asked(self):
        # asked where it applies, the default first; the answer taken
        seen = []

        def answer(pick):
            def choose(q, intro, options, default, yes, *r):
                seen.append((q, default, [o[0] for o in options], yes))
                return pick if q == "Faster drafting?" else default
            return choose
        c, _, _ = self.set_up(400, choose=answer(1))
        self.assertEqual(seen[-1][:2], ("Faster drafting?", 0))
        self.assertEqual(seen[-1][2][0], "Default: the cache in F16 rows, no drafting at this context")
        self.assertEqual(c["args"][c["args"].index("--kv") + 1], "q8")
        c, _, _ = self.set_up(400, choose=answer(0))
        self.assertNotIn("--kv", c["args"])
        seen.clear()
        self.set_up(76, choose=answer(1))                                              # (no trade here: not asked)
        self.assertNotIn("Faster drafting?", [s[0] for s in seen])

    def test_by_name(self):
        # where the setup's sums give the drafter room beside F16 rows too (a free card), the default claims it and the
        # engine may still find none (its start line says so): by name the 8-bit rows are taken, said so
        c, text, _ = self.set_up(76, fast_drafting="on")
        self.assertEqual((c["args"][c["args"].index("--kv") + 1], "--dfly2" in c["args"]), ("q8", True))
        self.assertIn("so the DFly2 drafter fits also where the engine finds no room for it beside F16 rows", text)
        # where even 8-bit rows leave it no room, or drafting is off: the default, said why
        c, text, _ = self.set_up(552, fast_drafting="on")
        self.assertEqual(("--kv" in c["args"], "--dfly2" in c["args"]), (False, False))
        self.assertIn("faster drafting: not set up - even with the cache in 8-bit rows the DFly2 drafter has no room",
                      text)
        c, text, _ = self.set_up(400, fast_drafting="on", drafting="off")
        self.assertEqual(("--kv" in c["args"], "--dfly2" in c["args"]), (False, False))
        self.assertIn("faster drafting: not set up with --drafting off", text)


MIB = 2**20


class WindowsRoom(unittest.TestCase):
    """Windows keeps part of every card from the programs on it, which nvidia-smi does not show: the setup counts the
    VRAM the engine's CUDA gets there (hy4_engine_room, measured on the PC: CUDA's free VRAM with a context held - the
    engine's own reading - plus what a context takes and what the driver keeps from CUDA), and only there."""
    G = maya.hy4_dfly2_gib("maya-xs")
    # CUDA's readings (free with one context, free with two, total), bytes: an RTX 3050 8 GB under Windows with 0.1 GB
    # in use (Uranus), a V100 32 GB under Linux (Sun's eGPU)
    URANUS = (7556038656, 7484735488, 8589410304)
    V100 = (33745534976, 33422573568, 34072559616)

    def pc(self, gib=8, n=1):
        return {"backend": "cuda", "gpus": [dict(card(gib, i), pci=f"00000000:0{i + 1}:00.0", driver="610.88")
                                            for i in range(n)]}

    def room(self, pc, probe, win=True, busy=None, saved=None):
        with patch.object(maya, "WIN", win), patch.object(maya, "maya_engine_mib", return_value=busy or {}):
            return maya.hy4_engine_room(pc, probe=probe, saved=saved)

    # an 8 GB layout's start at 8K (the setup's config on the RTX 3050) and what its engine said with --room-report
    CFG = {"exe": "C:/Maya/build/maya-hy4.exe", "gpu": [0], "backend": "cuda",
           "args": ["--pack", "p", "--max-context", "8192", "--chunk", "256", "--reserve-gb", "0.5",
                    "--vram-experts", "0"], "installer": {"hy4_quant": "maya-xs", "layout": "8gb"}}
    INFO = {"room_rows": "f16", "room_drafter": 0, "room_streamed": 0,
            "room_gpus": "0000:01:00.0,7556038656,805306368"}                     # (0.75 GiB spare)

    def test_first_guess_then_the_engines_figure(self):
        one = self.URANUS[0] / 2**30
        pc = self.pc()
        self.room(pc, {"0000:01:00.0": self.URANUS})
        g = pc["gpus"][0]
        self.assertEqual((g["engine_room_how"], round(g["cuda_free_gb"], 4)), ("estimated", round(one, 4)))
        self.assertAlmostEqual(g["engine_room_gb"], one + maya.HY4_FIT_CUDA_OWN_GIB)
        # the engine's first start on this card measures it: the setup's sums for that start (the 8 GB layout at 8K,
        # F16 rows, no drafter: 6.8 GiB) and what it had spare, against what the setup's CUDA probe sees
        with patch.object(maya, "WIN", True):
            # (the probe's reading is taken before the engine starts: Windows' figure drops as the engine fills it)
            card_ = maya.hy4_room_due(self.CFG, gpus=self.pc()["gpus"], probe={"0000:01:00.0": self.URANUS})
            self.assertEqual((card_["pci"], card_["cuda_probe"]), ("00000000:01:00.0", list(self.URANUS)))
            rec = maya.hy4_room_measure(self.CFG, card_, self.INFO, probe=AssertionError("not asked"))
            self.assertIsNone(maya.hy4_room_due(self.CFG, gpus=self.pc()["gpus"], probe=AssertionError("no")))
        need = maya.hy4_config_need_gib("maya-xs", "8gb", 8192, chunk=256, reserve_gb=0.5)
        self.assertEqual(need, 6.8)
        self.assertAlmostEqual(rec["own_gib"], need + 0.75 - one, places=3)
        self.assertEqual((rec["rows"], rec["drafter"], rec["context"]), ("f16", False, 8192))
        self.assertEqual(list(maya.hy4_room_load()), [maya.hy4_room_key(card_)])
        # the next setup counts it ("measured"); on another driver, or another card in the slot: the first guess
        pc = self.pc()
        got = self.room(pc, {"0000:01:00.0": self.URANUS})
        self.assertAlmostEqual(got[0], one + rec["own_gib"])
        self.assertEqual(pc["gpus"][0]["engine_room_how"], "measured")
        for other in ({"driver": "615.10"}, {"name": "NVIDIA GeForce RTX 4060"}, {"vram_gb": 12.0}):
            pc = {"backend": "cuda", "gpus": [dict(self.pc()["gpus"][0], **other)]}
            self.room(pc, {"0000:01:00.0": self.URANUS})
            self.assertEqual(pc["gpus"][0]["engine_room_how"], "estimated", other)
        # measured again under a new driver: the old driver's figure goes
        with patch.object(maya, "WIN", True):
            new = dict(card_, driver="615.10")
            self.assertIsNotNone(maya.hy4_room_measure(self.CFG, new, self.INFO))
        self.assertEqual(list(maya.hy4_room_load()), [maya.hy4_room_key(new)])

    def test_the_measured_start(self):
        probe = {"0000:01:00.0": self.URANUS}
        card_ = self.pc()["gpus"][0]

        def got(info=None, cfg=None, **kw):
            return maya.hy4_room_measure(cfg or self.CFG, card_, {**self.INFO, **(info or {})}, probe=kw.get("probe", probe),
                                         save=False)
        base = got()["own_gib"]
        # the rows and the drafter the engine took count in the sums (8-bit rows: their own; the drafter's room)
        self.assertAlmostEqual(got({"room_drafter": 1})["own_gib"], base + maya.hy4_dfly2_gib("maya-xs"), delta=0.002)
        self.assertLess(got({"room_rows": "q8"})["own_gib"], base)
        # a context the plan does not list: its sums without the fit's margin
        c6 = dict(self.CFG, args=[x if x != "8192" else "6144" for x in self.CFG["args"]])
        self.assertAlmostEqual(got(cfg=c6)["need_gib"], maya.hy4_config_need_gib("maya-xs", "8gb", 6144, chunk=256,
                                                                                reserve_gb=0.5) - maya.HY4_FIT_MARGIN_GIB,
                               places=3)
        # not kept: a start that streams its cache, a card the engine did not report, no probe reading, a figure out
        # of all reason, no plan for the size, a report the engine did not write
        for info, kw in (({"room_streamed": 1}, {}), ({"room_gpus": "0000:02:00.0,7556038656,805306368"}, {}),
                         ({}, {"probe": {}}), ({"room_gpus": "0000:01:00.0,7556038656,8053063680"}, {}),
                         ({"room_gpus": "0000:01:00.0,x,1"}, {}), ({"room_gpus": None}, {})):
            self.assertIsNone(got(info, **kw), info)
        self.assertIsNone(got(cfg=dict(self.CFG, installer={"hy4_quant": "nope"})))

    def test_uranus_measured(self):
        # the RTX 3050's engine at 8K (F16 rows, no drafter): 7.04 GiB free, its choices held with 0.49 GiB less - CUDA's
        # own share there 0.26 GiB in the setup's terms; with it the setup claims the drafter where the engine loads
        # it: at 4K and 5K, not at 6K and 8K (F16), and at 8K beside 8-bit rows (the question asked) - measured
        info = dict(self.INFO, room_gpus="0000:01:00.0,7556038656,531283968")
        rec = maya.hy4_room_measure(self.CFG, dict(self.pc()["gpus"][0], cuda_probe=list(self.URANUS)), info)
        self.assertAlmostEqual(rec["own_gib"], 0.258, delta=0.001)
        q, pc = maya.hy4_quant_table("maya-xs"), self.pc()
        self.room(pc, {"0000:01:00.0": self.URANUS})
        cards = pc["gpus"]
        self.assertEqual(cards[0]["engine_room_how"], "measured")
        claims = {}
        for ctx in (4096, 5120, 6144, 8192):
            f = maya.hy4_fit(q, cards, ctx, used={0: 76}, ram_gb=64, drafter=self.G)
            self.assertEqual((f["kv"], f["room_measured"]), ("f16", True))
            claims[ctx] = maya.hy4_dfly2_need_gib("maya-xs", f) <= f["usable"]
        self.assertEqual(claims, {4096: True, 5120: True, 6144: False, 8192: False})
        f8 = maya.hy4_fit(q, cards, 8192, used={0: 76}, ram_gb=64, drafter=self.G)
        fast = maya.hy4_fast_drafting_fit("maya-xs", cards, f8, used={0: 76}, ram_gb=64)
        self.assertEqual((fast["kv"], fast["f16_room"]), ("q8", False))
        # the first guess keeps the margins (the drafter's sums never set against this PC's engine)
        pc = self.pc()
        self.room(pc, {"0000:01:00.0": self.URANUS}, saved={})
        f4 = maya.hy4_fit(q, pc["gpus"], 4096, used={0: 76}, ram_gb=64, drafter=self.G)
        self.assertNotIn("room_measured", f4)
        self.assertAlmostEqual(maya.hy4_dfly2_need_gib("maya-xs", f4), 6.5 + self.G + maya.HY4_FIT_MARGIN_GIB)

    def test_due(self):
        gl, pr = self.pc(n=2)["gpus"], {"0000:01:00.0": self.URANUS, "0000:02:00.0": self.URANUS}
        with patch.object(maya, "WIN", True):
            self.assertIsNotNone(maya.hy4_room_due(self.CFG, gpus=gl, saved={}, probe=pr))
            self.assertEqual(maya.hy4_room_due(dict(self.CFG, gpu="1"), gpus=gl, saved={}, probe=pr)["index"], 1)
            self.assertIsNone(maya.hy4_room_due(self.CFG, gpus=gl, saved={maya.hy4_room_key(gl[0]): {"own_gib": 0.5}},
                                                probe=pr))
            self.assertIsNone(maya.hy4_room_due(self.CFG, gpus=gl, saved={}, probe={}))   # (no reading: no figure)
            for cfg in (dict(self.CFG, gpu=[0, 1]), dict(self.CFG, gpu=None), dict(self.CFG, gpu=[3]),
                        dict(self.CFG, exe="C:/Maya/build/maya-ds4.exe"), dict(self.CFG, backend="hip")):
                self.assertIsNone(maya.hy4_room_due(cfg, gpus=gl, saved={}, probe=pr), cfg)
            self.assertIsNone(maya.hy4_room_due(self.CFG, gpus=[dict(gl[0], unified_memory=True)], saved={}, probe=pr))
        with patch.object(maya, "WIN", False):
            self.assertIsNone(maya.hy4_room_due(self.CFG, gpus=gl, saved={}, probe=pr))

    def test_measured_on_windows_only(self):
        one, two, tot = self.URANUS
        pc = self.pc()
        got = self.room(pc, {"0000:01:00.0": self.URANUS})
        # (CUDA's 7.04 GiB free; a context 68 MiB and the driver's 0.5 MiB there, less than the sums count for them)
        self.assertLess((one - two) / 2**30 + 8.0 - tot / 2**30, maya.HY4_FIT_CUDA_OWN_GIB)
        self.assertAlmostEqual(got[0], one / 2**30 + maya.HY4_FIT_CUDA_OWN_GIB)
        self.assertEqual(pc["gpus"][0]["engine_room_gb"], got[0])
        # not measured: another OS, a card a Maya engine holds now, AMD, a unified-memory GPU, a card with no PCI id,
        # a device CUDA did not read
        for case in ({"win": False}, {"busy": {0: None}}):
            pc = self.pc()
            self.assertEqual(self.room(pc, {"0000:01:00.0": self.URANUS}, **case), {}, case)
            self.assertNotIn("engine_room_gb", pc["gpus"][0])
        for pc in ({"backend": "hip", "gpus": [dict(radeon(), pci="00000000:01:00.0")]},
                   {"backend": "cuda", "gpus": [dict(card(8), pci="00000000:01:00.0", unified_memory=True)]},
                   {"backend": "cuda", "gpus": [card(8)]}):
            self.assertEqual(self.room(pc, AssertionError("not asked")), {})
        self.assertEqual(self.room(self.pc(), {"0000:02:00.0": self.URANUS}), {})
        with patch.object(maya, "WIN", True), patch.object(maya, "maya_engine_mib", return_value={}), \
                patch.object(maya, "cuda_room_bytes", return_value={}):
            self.assertEqual(maya.hy4_engine_room(self.pc()), {})           # (no driver to ask: as before)

    def test_the_same_as_linux_counts(self):
        # the same sums on Linux (nothing kept there) give nvidia-smi's total less what is in use: CUDA's free VRAM
        # and a context's 308 MiB and the driver's 274 MiB (32768 MiB less CUDA's total), 0 MiB in use - more than
        # HY4_FIT_CUDA_OWN_GIB, so the measured part
        got = self.room(self.pc(32), {"0000:01:00.0": self.V100})
        self.assertAlmostEqual(got[0], 32.0, delta=0.01)

    def test_usable(self):
        room = dict(card(8), engine_room_gb=7.5)
        self.assertAlmostEqual(maya.hy4_usable_gib([room], {0: 76}), 7.5)                # (no more than its room)
        self.assertAlmostEqual(maya.hy4_usable_gib([room], {0: 900}), 8.0 - 900 / 1024)   # (less in VRAM in use)
        self.assertAlmostEqual(maya.hy4_usable_gib([room], {0: 4000}), 7.0)               # (in use: at most 1 GiB)
        self.assertAlmostEqual(maya.hy4_usable_gib([room], {}), 7.5)                      # (no reading)
        self.assertAlmostEqual(maya.hy4_usable_gib([dict(card(8), engine_room_gb=7.7)], {}), 7.6)
        self.assertAlmostEqual(maya.hy4_usable_gib([room, card(8, 1)], {0: 76, 1: 76}), 7.5 + 8.0 - 76 / 1024)
        # no room on the cards: the figures as before
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8)], {0: 76}), 8.0 - 76 / 1024)
        self.assertAlmostEqual(maya.hy4_usable_gib([card(8), card(8, 1)], {}), 16.0 - maya.HY4_FIT_DISPLAY_GIB)

    def test_uranus(self):
        # an idle 8 GB card under Windows: at 8K the setup no longer claims the drafter beside the F16 rows (the
        # engine does not load it there) and offers the faster drafting; at 4K it claims it (the engine loads it)
        q, pc = maya.hy4_quant_table("maya-xs"), self.pc()
        self.room(pc, {"0000:01:00.0": self.URANUS})
        cards = pc["gpus"]
        f8 = maya.hy4_fit(q, cards, 8192, used={0: 76}, ram_gb=64, drafter=self.G)
        self.assertEqual((f8["kv"], f8["chunk"]), ("f16", 256))
        self.assertGreater(maya.hy4_dfly2_need_gib("maya-xs", f8), f8["usable"])
        fast = maya.hy4_fast_drafting_fit("maya-xs", cards, f8, used={0: 76}, ram_gb=64)
        self.assertEqual((fast["kv"], fast["f16_room"]), ("q8", False))
        f4 = maya.hy4_fit(q, cards, 4096, used={0: 76}, ram_gb=64, drafter=self.G)
        self.assertLessEqual(maya.hy4_dfly2_need_gib("maya-xs", f4), f4["usable"])
        # without the room (before; or Linux with the same card): the drafter claimed at 8K
        f8 = maya.hy4_fit(q, [card(8)], 8192, used={0: 76}, ram_gb=64, drafter=self.G)
        self.assertLessEqual(maya.hy4_dfly2_need_gib("maya-xs", f8), f8["usable"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # (the engine's measured figures: a store of the test's own, empty to begin with)
        store = patch.object(maya, "HY4_ROOM_STORE", Path(self.tmp.name) / "data" / "vram-room.json")
        store.start()
        self.addCleanup(store.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_setup(self):
        # the setup on that card: at 8K no --dfly2, the drafting line says why and the faster drafting is asked
        seen = []

        def choose(q, intro, options, default, yes, *r):
            seen.append(q)
            return default
        c, text, _ = Drafts.set_up(self, 8, block=False, dfly=True, dfly_gib=self.G, small=True, used=76, context=8192,
                                   revision=TABLE["revision"], room=self.URANUS, choose=choose)
        self.assertNotIn("--dfly2", c["args"])
        self.assertIn("Faster drafting?", seen)
        self.assertIn("VRAM: Windows keeps part of this card from the programs on it - the setup counts 7.5 of its "
                      "8.0 GB, from CUDA's own reading", text)
        self.assertIn("the DFly2 drafter needs about 0.7 GB beside the model's own parts (7.7 GB with it at this "
                      "context, 7.5 GB free)", text)
        self.assertIn("(at a context of 4K it fits: ", text)
        self.assertEqual(c["installer"]["engine_room_gb"]["0"][1], 8.0)
        # at 4K: claimed and written
        c, text, _ = Drafts.set_up(self, 8, block=False, dfly=True, dfly_gib=self.G, small=True, used=76, context=4096,
                                   revision=TABLE["revision"], room=self.URANUS)
        self.assertIn("--dfly2", c["args"])
        self.assertIn("drafting: Tencent's DFly2 drafter drafts tokens", text)


def hy4_cfg(d: Path, mtp: bool = True, n_mtp: int = 1, vision: bool = False) -> dict:
    pack = d / "Hy4-flash" / "maya-hy4"
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    (pack / "hy4.txt").write_text(f"arch hy_v4_flash\nn_layer 42\nn_mtp {n_mtp}\ncontext_length 1048576\n"
                                  f"model_dir {pack.parent}\n" + ("vision.dim 1152\n" if vision else ""))
    (pack / "tensors.txt").write_text("".join(f"file {i} {n}\n" for i, n in enumerate(SHARDS)) + "t x F32 0 0 4 1\n")
    return {"exe": "/b/build/maya-hy4", "args": ["--pack", str(pack), "--max-context", "65536"] + (["--mtp"] if mtp else []),
            "model_name": "hy4-flash", "gpu": [0, 1], "tokenizer": str(pack / "tokenizer"),
            "installer": {"family": "hy4", "quant": "Hy4-flash", "hy4_quant": "q4_k_m"}}


class Names(unittest.TestCase):
    def test_shown_and_ids(self):
        self.assertEqual(Q.shown("hy4", "q4_k_m", 168.35), "Hy4-Flash · Q4_K_M (168 GB)")
        self.assertEqual(Q.config_stem("hy4", "q4_k_m"), "maya-hy4-flash")
        for name in ("hy4-flash-q4_k_m", "HY4-Q4_K_M"):
            self.assertEqual(Q.resolve(name), ("hy4", "q4_k_m"), name)
        for name in ("hy4-flash", "Hy4-flash", "hy4"):   # (the model alone: its files are asked - Maya's sizes first)
            self.assertEqual(Q.resolve(name), ("hy4", None), name)
        self.assertEqual((maya.HY4, maya.HY4_MODEL_NAME, maya.HY4_SHOWN), ("Hy4-Flash", "hy4-flash", "Hy4-Flash"))

    def test_family(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = hy4_cfg(Path(t))
            self.assertEqual(Q.family_of(cfg), "hy4")
            cfg.pop("installer")                       # (by its engine too: maya-hy4's --pack is not DeepSeek's)
            self.assertEqual(Q.family_of(cfg), "hy4")
            c = hy4_cfg(Path(t) / "b")
            self.assertEqual((Q.config_key(c), Q.config_label(c)), ("q4_k_m", "Hy4-Flash · Q4_K_M"))
            self.assertEqual(Q.model_defaults(c), ({"temperature": 0.9, "top_p": 1.0}, "high"))
            self.assertEqual(maya.setup_title(c), "Hy4-Flash")

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
        self.assertEqual(label, "Hy4-Flash")
        self.assertIn("Q4_K_M", about)
        self.assertIn("pictures", about)
        self.assertEqual(maya.FAMILY_TITLE["hy4"], "Hy4-Flash")


class Setup(unittest.TestCase):
    def test_manifest_pinned(self):
        """The shards and the vision tower's file, pinned with their sha256; --no-vision's download: the shards."""
        m = maya.hy4_entry()
        self.assertEqual(m["repo"], "AngelSlim/Hy4-flash-GGUF")
        self.assertEqual(len(m["revision"]), 40)
        self.assertEqual(list(m["files"]), [SHARDS[0], NGRAM])       # (its trunk; the n-gram table: every set's one)
        self.assertEqual(list(m["theirs"]), [SHARDS[1]])              # (their own shard: taken where a folder has it)
        self.assertTrue(m["mtp"])
        for name, sha in m["sha256"].items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertGreater(m["files"][name], 50e9 if name in SHARDS else 1e9)
        self.assertAlmostEqual(m["download_gb"], 168.35, delta=0.1)
        # the vision tower: a download of its own (asked with the model; --no-vision: not) - AngelSlim's BF16 file for
        # their set, Maya's one Q8_0 file for every Maya size, each from its repo at its pinned commit
        v = maya.hy4_vision_entry()
        self.assertEqual((v["repo"], v["revision"], v["folder"], list(v["files"])), (m["repo"], m["revision"], "", [MMPROJ]))
        self.assertAlmostEqual(v["download_gb"], 1.04, delta=0.01)
        self.assertRegex(v["sha256"][MMPROJ], r"^[0-9a-f]{64}$")
        x = maya.hy4_vision_entry("maya-xs")
        self.assertEqual((x["repo"], x["revision"], x["folder"], x["name"]),
                         (TABLE["repo"], TABLE["revision"], "Maya-XS", "mmproj-Hy4-Flash-Maya-XS.gguf"))
        self.assertEqual(x["files"][x["name"]], 784524096)
        self.assertEqual(maya.hy4_vision_entry("maya-l")["name"], x["name"])   # (one file for every Maya size)

    def test_pack_with_the_vision_tower(self):
        """The pack step indexes the folder's mmproj file (--mmproj), stamps it with the shards, and is built again
        when the file comes or goes; without it, --mmproj none."""
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "Hy4-flash"
            d.mkdir()
            files = [d / n for n in SHARDS]
            for f in files:
                f.write_bytes(b"x")
            runs = []

            def fake_run(cmd, *x, **k):
                runs.append(cmd)
                out = Path(cmd[cmd.index("--out") + 1])
                (out / "tokenizer").mkdir(parents=True, exist_ok=True)
                for n in ("tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
                    (out / n).write_text("x")
                (out / "hy4.txt").write_text("arch hy_v4_flash\n" +
                                             ("vision.dim 1152\n" if cmd[cmd.index("--mmproj") + 1] != "none" else ""))
            a = type("A", (), {"repack": False})()
            with patch.object(maya, "run", fake_run), patch.object(maya, "step"), patch.object(maya, "ok"), \
                    patch.object(maya, "say"):
                pack = maya.hy4_pack_step(a, d, files)
                self.assertEqual(runs[-1][runs[-1].index("--mmproj") + 1], "none")
                self.assertFalse(maya.hy4_pack_has_vision(pack))
                (d / MMPROJ).write_bytes(b"v")         # (downloaded later: the pack is built again with it)
                self.assertEqual(maya.hy4_vision_file(d), d / MMPROJ)
                maya.hy4_pack_step(a, d, files, d / MMPROJ)
                self.assertEqual(len(runs), 2)
                cmd = runs[-1]
                self.assertEqual(cmd[cmd.index("--mmproj") + 1], MMPROJ)
                self.assertEqual(cmd[cmd.index("--files") + 1:], list(SHARDS))
                self.assertTrue(maya.hy4_pack_has_vision(pack))
                stamped = [r["name"] for r in json.loads((pack / maya.PACK_SOURCES).read_text())["files"]]
                self.assertEqual(stamped, list(SHARDS) + [MMPROJ])
                maya.hy4_pack_step(a, d, files, d / MMPROJ)      # (nothing changed: kept)
                self.assertEqual(len(runs), 2)
                maya.hy4_pack_step(a, d, files)                  # (the tower left out: built again without it)
                self.assertEqual((len(runs), runs[-1][runs[-1].index("--mmproj") + 1]), (3, "none"))

    def test_the_ngram_table_from_mayas_repo(self):
        """Tencent's own Q4_K_M: its trunk from AngelSlim's repo at the commit it is pinned at, its n-gram table the
        one file every Hy4 set shares - from Maya's repo at the table's commit, with that file's sha256."""
        m = maya.hy4_entry()
        repo, rev = m["sources"][NGRAM]
        self.assertEqual((repo, rev), (TABLE["repo"], TABLE["revision"]))
        self.assertEqual(repo, "peasantsmith/Hy4-Flash-Maya-GGUF")
        folder = m["source_folders"][NGRAM]
        url = maya.HF.format(repo=repo, revision=rev, path=f"{folder}/{NGRAM}")
        self.assertTrue(url.startswith("https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/resolve/"), url)
        self.assertIn(f"/{TABLE['revision']}/Maya-XS/{NGRAM}", url)
        self.assertNotIn(NGRAM, m.get("sources", {}).get(SHARDS[0], ()))
        self.assertNotIn(SHARDS[0], m["sources"])                     # (the trunk: the entry's own repo, AngelSlim's)
        size, sha = next(iter(TABLE["ngram"].values()))
        self.assertEqual((m["files"][NGRAM], m["sha256"][NGRAM]), (size, sha))
        xs = maya.hy4_entry("maya-xs")                               # (the same file as Maya's sizes download)
        self.assertEqual((xs["files"][NGRAM], xs["sha256"][NGRAM]), (size, sha))

    def test_their_own_ngram_shard_in_the_folder(self):
        """A folder that already has AngelSlim's whole set (--model-dir): both of its files are taken, nothing asked."""
        man = json.loads(json.dumps(TABLE))                           # (the table with small files of these sizes)
        q4 = man["others"]["q4_k_m"]
        q4["files"][SHARDS[0]][0], q4["theirs"][SHARDS[1]][0] = 16, 8
        with tempfile.TemporaryDirectory() as t, with_manifest(man):
            d = Path(t)
            m = maya.hy4_entry()
            (d / SHARDS[0]).write_bytes(b"0" * 16)
            (d / SHARDS[1]).write_bytes(b"0" * 8)
            a = type("A", (), {"model_dir": str(d), "yes": True, "download_model": False})()
            asked = []
            with patch.object(maya, "step"), patch.object(maya, "ok"), patch.object(maya, "say"), \
                    patch.object(maya, "warn"), patch.object(maya, "rotational", return_value=False), \
                    patch.object(maya, "sha256_checked", lambda f, sha: asked.append((f.name, sha)) or True), \
                    patch.object(maya, "offer_download", side_effect=AssertionError("nothing to download")):
                files = maya.hy4_model_step(a, d, maya.HY4_Q4, models=d.parent)
            self.assertEqual([f.name for f in files], list(SHARDS))
            self.assertEqual(asked, [(SHARDS[0], m["sha256"][SHARDS[0]]), (SHARDS[1], m["theirs"][SHARDS[1]][1])])

    def test_context_by_vram(self):
        """The cache sits on every GPU of a tensor-parallel group: the smallest GPU decides."""
        gpus = lambda *v: {"gpus": [{"vram_gb": x} for x in v]}   # noqa: E731
        self.assertEqual(maya.hy4_recommended_context(gpus(24, 16, 16)), 131072)   # (2.0 GB at 128K)
        self.assertEqual(maya.hy4_recommended_context(gpus(12)), 65536)
        self.assertEqual(maya.hy4_recommended_context(gpus(24, 6)), 32768)
        self.assertEqual(maya.hy4_recommended_context({"gpus": []}), 32768)
        self.assertIn(maya.HY4_DEFAULT_CONTEXT, maya.HY4_Q4_CONTEXTS)
        self.assertEqual(maya.HY4_Q4_CONTEXTS[-1], 262144)                         # (256K: offered for the Q4_K_M)
        # ... and up to the model's 1M where the smallest card is large enough for the cache (9.1 GB at 1M)
        q4 = lambda *v: maya.hy4_q4_contexts(gpus(*v)["gpus"])   # noqa: E731
        self.assertEqual((q4(16, 16), q4(24, 24), q4(48, 24), q4(48, 48)),
                         (maya.HY4_Q4_CONTEXTS, maya.HY4_Q4_CONTEXTS + [524288], maya.HY4_Q4_CONTEXTS + [524288],
                          maya.HY4_Q4_CONTEXTS + [524288, 1048576]))
        self.assertEqual(q4(), maya.HY4_Q4_CONTEXTS)
        self.assertEqual(maya.hy4_recommended_context(gpus(32, 32)), 262144)       # (4.4 GB in F16 rows)

    def test_config(self):
        """maya-hy4's own flags only (it stops at one it does not know), the model's sampling and level, its engine;
        a setup again keeps the engine settings set by hand."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            pack = t / "Hy4-flash" / "maya-hy4"
            (pack / "tokenizer").mkdir(parents=True)
            a = type("A", (), dict(port=None, env=[], host=None, api_key=None, no_vision=False))()
            with patch.object(maya, "ROOT", t), patch.object(maya, "write_run_script", lambda *x: t / "run"), \
                    patch.object(maya, "ok"), patch.object(maya, "say"):
                p = maya.write_config_hy4(a, {"gpus": [{"index": 0}, {"index": 3}]}, {"lib_dirs": ["/cuda"]}, pack,
                                          131072, t, pack.parent, True)
                self.assertEqual(p.name, "maya-hy4-flash.json")
                c = json.loads(p.read_text())
                self.assertEqual(c["args"], ["--pack", str(pack), "--max-context", "131072", "--mtp"])
                self.assertEqual(Path(c["exe"]).stem, "maya-hy4")     # (maya-hy4.exe on Windows)
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
            # --no-vision with the tower in the pack: --vision off (the Models page's Images turns it on); a setup
            # again without it does not keep the off
            (pack / "hy4.txt").write_text("arch hy_v4_flash\nvision.dim 1152\n")
            a.no_vision = True
            with patch.object(maya, "ROOT", t), patch.object(maya, "write_run_script", lambda *x: t / "run"), \
                    patch.object(maya, "ok"), patch.object(maya, "say"):
                c = json.loads(maya.write_config_hy4(a, {"gpus": [{"index": 0}]}, {}, pack, 65536, t, pack.parent,
                                                     False, vision=False).read_text())
                self.assertEqual(c["args"][c["args"].index("--vision") + 1], "off")
                a.no_vision = False
                c = json.loads(maya.write_config_hy4(a, {"gpus": [{"index": 0}]}, {}, pack, 65536, t, pack.parent,
                                                     False).read_text())
                self.assertNotIn("--vision", c["args"])

    def test_the_engine_target(self):
        """The launcher builds and starts maya-hy4 for it, and the GPU list knows the engine's process."""
        self.assertEqual(maya.exe_for_target("maya-hy4").stem, "maya-hy4")   # (maya-hy4.exe on Windows)
        self.assertIn("maya-hy4", maya.ENGINE_PROCESSES)
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(maya.bench_exe(hy4_cfg(Path(t))).stem, "maya-hy4")
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
        self.assertTrue(seen[0].startswith("Hy4-Flash runs best on all of them"), seen)

    def test_not_on_metal(self):
        a = type("A", (), {"backend": "metal"})()
        with patch.object(maya, "step"), self.assertRaises(SystemExit):
            maya.check_pc(a, "hy4")

    def test_amd_on_linux_goes_to_the_hip_check(self):
        a = type("A", (), {"backend": "hip"})()
        with patch.object(maya, "step"), patch.object(maya, "WIN", False), \
                patch.object(maya.sys, "platform", "linux"), \
                patch.object(maya, "check_hip_pc", return_value={"backend": "hip"}) as hip:
            self.assertEqual(maya.check_pc(a, "hy4"), {"backend": "hip"})
        hip.assert_called_once_with(a, text_only=False)


class Dashboard(unittest.TestCase):
    def test_models_page(self):
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = hy4_cfg(Path(t))
            self.assertEqual(M.choices(cfg), {"draft": ["mtp", "none"], "images": False,
                                              "ablate": {"available": False, "why": "not available for this model"}})
            s = M.settings_of(cfg)
            self.assertEqual((s["draft"], s["tp"], s["reserve_gb"], s["prefill_chunk"], s["images"]),
                             ("mtp", "auto", None, None, False))
            self.assertEqual(M.trained_context_ds4(M.pack_of(cfg)), 1048576)
            self.assertEqual(M.pack_reads(M.pack_of(cfg)), [Path(t) / "Hy4-flash" / n for n in SHARDS])
            out = M.apply(cfg, {"draft": "none", "tp": 0, "reserve_gb": 0.6, "prefill_chunk": 2048, "context": 131072})
            self.assertEqual(out["args"], ["--pack", str(M.pack_of(cfg)), "--max-context", "131072", "--tp", "0",
                                           "--reserve-gb", "0.6", "--chunk", "2048"])
            self.assertEqual(M.apply(cfg, {"context": 262144})["args"][:4],
                             ["--pack", str(M.pack_of(cfg)), "--max-context", "262144"])
            back = M.apply(out, {"draft": "mtp", "tp": "auto", "reserve_gb": None, "prefill_chunk": None})
            self.assertEqual(back["args"], ["--pack", str(M.pack_of(cfg)), "--max-context", "131072", "--mtp"])
            for bad in ({"kv": "int8"}, {"parallel": 2}, {"images": True}, {"expert_cache": 100}):
                with self.assertRaises(ValueError, msg=bad):
                    M.apply(cfg, bad)
            with self.assertRaises(ValueError):
                M.apply(hy4_cfg(Path(t) / "b", n_mtp=0), {"draft": "mtp"})
            with self.assertRaises(ValueError):
                M.apply(cfg, {"draft": "dfly2"})                 # (no DFly2 drafter in this pack)

    def test_models_page_dfly2(self):
        """A pack with the DFly2 drafter (hy4.txt: dfly_layers): DFly2 offered beside MTP; one drafter's flag at a time."""
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = hy4_cfg(Path(t), mtp=False)
            with (M.pack_of(cfg) / "hy4.txt").open("a") as f:
                f.write("dfly_layers 5\n")
            self.assertEqual(M.choices(cfg)["draft"], ["mtp", "dfly2", "none"])
            self.assertEqual(M.settings_of(cfg)["draft"], "none")
            on = M.apply(cfg, {"draft": "dfly2"})
            self.assertEqual(on["args"][-1], "--dfly2")
            self.assertEqual(M.settings_of(on)["draft"], "dfly2")
            mtp = M.apply(on, {"draft": "mtp"})
            self.assertEqual((mtp["args"][-1], "--dfly2" in mtp["args"]), ("--mtp", False))
            self.assertNotIn("--dfly2", M.apply(on, {"draft": "none"})["args"])

    def test_models_page_images(self):
        """A pack with the vision tower: Images on by default (no flag), off writes --vision off, on drops it."""
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = hy4_cfg(Path(t), vision=True)
            self.assertTrue(M.choices(cfg)["images"])
            self.assertTrue(M.settings_of(cfg)["images"])
            off = M.apply(cfg, {"images": False})
            self.assertEqual(off["args"][-2:], ["--vision", "off"])
            self.assertFalse(M.settings_of(off)["images"])
            on = M.apply(off, {"images": True})
            self.assertNotIn("--vision", on["args"])
            self.assertTrue(M.settings_of(on)["images"])

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


class Q4xCheck(unittest.TestCase):
    """The CPU lane's 4-bit expert copies (maya-hy4's HY4_Q4X): the engine measures them once its experts are warm and
    remembers the verdict per PC (the data folder's hy4-cpu-lane.txt) - no start of their own; --speed-calibration
    measures again (HY4_Q4X_MEASURE=1); a verdict an earlier start's check wrote into the config's env is dropped."""

    def cfg(self, root: Path, **kw) -> dict:
        tok = root / "tokenizer"
        tok.mkdir(exist_ok=True)
        (tok / "chat_template.jinja").write_text("{{ messages }}")
        c = {"exe": str(root / "maya-hy4"), "tokenizer": str(tok), "gpu": [0],
             "args": ["--pack", str(root / "pack"), "--max-context", "8192"],
             "installer": {"family": "hy4", "hy4_quant": "maya-xs"}}
        c.update(kw)
        return c

    def test_an_earlier_checks_verdict_is_left_to_the_engine(self):
        """1.1.69's start-time check wrote HY4_Q4X 1 / 0 into the config (often measured on cold experts): dropped at
        the start, so the engine measures once warm and remembers it per PC, model and lane kernels.  A value of the
        user's own stays."""
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "maya-hy4-flash-maya-xs.json"
            for env, inst, kept in (({"HY4_Q4X": "1"}, {"q4x_calibration": {"result": "on"}}, False),
                                    ({"HY4_Q4X": "0"}, {"q4x_calibration": {"result": "off"}}, False),
                                    ({"HY4_Q4X": "0"}, {}, True),                                 # (set by hand)
                                    ({"HY4_Q4X": "1"}, {"q4x_calibration": {"result": "off"}}, True)):
                with self.subTest(env=env, inst=inst):
                    c = self.cfg(Path(t), env=dict(env))
                    c["installer"].update(inst)
                    p.write_text(json.dumps(c))
                    got = maya.q4x_config_migrate(p, maya.read_json(p))
                    self.assertEqual("HY4_Q4X" in (got.get("env") or {}), kept)
                    self.assertEqual("HY4_Q4X" in (maya.read_json(p).get("env") or {}), kept)
            glm = {"exe": "/b/maya", "args": ["--glm-pack", "/p"], "env": {"HY4_Q4X": "1"},
                   "installer": {"family": "glm", "q4x_calibration": {"result": "on"}}}
            self.assertEqual(maya.q4x_config_migrate(p, glm), glm)
        self.assertFalse(hasattr(maya, "needs_q4x_calibration"))          # (no start loads the model for it any more)

    def test_the_server_names_the_verdicts_file(self):
        import serve.server as SV
        with tempfile.TemporaryDirectory() as t:
            env = SV.child_env(self.cfg(Path(t)))
        self.assertEqual(env["HY4_Q4X_FILE"], str(SV.ROOT / "data" / "hy4-cpu-lane.txt"))
        self.assertNotIn("HY4_Q4X_FILE", SV.child_env({"exe": "/b/maya", "args": ["--glm-pack", "/p"]}))

    def test_the_verdict(self):
        self.assertEqual(maya.q4x_verdict("maya-hy4: CPU lane: the 4-bit copies kept - an expert run 60 us"), "on")
        self.assertEqual(maya.q4x_verdict("maya-hy4: CPU lane: the 4-bit copies dropped - an expert run"), "off")
        self.assertEqual(maya.q4x_verdict("INFO CPU lane: a call of 16 experts ... - the file's rows kept"), "off")
        self.assertEqual(maya.q4x_verdict("INFO CPU lane: no 4-bit copies of its experts (2.0 GB of RAM free"), "off")
        self.assertEqual(maya.q4x_verdict("INFO CPU lane: ... 4-bit copies - the most routed experts re-coded"),
                         "unmeasured")
        self.assertIsNone(maya.q4x_verdict("INFO every routed expert in VRAM"))

    def run_check(self, root: Path, verdict_line: str):
        p = root / "maya-hy4-flash-maya-xs.json"
        p.write_text(json.dumps(self.cfg(root)))
        seen = {}

        class Engine:
            def __init__(self, cmd, stdin=None, stdout=None, stderr=None, env=None, **kw):
                seen["env"], seen["cmd"] = env, cmd
                stderr.write(verdict_line + "\n")
                self.stdin = io.StringIO()
                self.stdout = iter(["READY\n", "T 5\n", "DONE 64 30 1 1 length\n", "DONE 256 30 1 1 length\n"])

            def wait(self, timeout=None):
                return 0

        class Tok:
            @staticmethod
            def from_dir(d):
                return SimpleNamespace(encode=lambda s, parse_special=False: [1, 2, 3])

        said = []
        with patch.object(maya.subprocess, "Popen", Engine), patch.object(maya, "busy_skip", return_value=False), \
                patch.object(maya.time, "sleep"), patch.object(maya, "ok", said.append), \
                patch.object(maya, "warn", said.append), patch.object(maya, "say"), \
                patch.object(maya, "ROOT", root), patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(Tokenizer=Tok)}), \
                patch.object(maya, "bench_chat", return_value=lambda q: q):
            got = maya.q4x_calibration(p)
        return got, maya.read_json(p), seen, said

    def test_measured_again_and_left_to_the_engine(self):
        for line, want in (("maya-hy4: CPU lane: the 4-bit copies kept - x", True),
                           ("maya-hy4: CPU lane: the 4-bit copies dropped - x", False)):
            with self.subTest(want=want), tempfile.TemporaryDirectory() as t:
                got, c, seen, _ = self.run_check(Path(t), line)
                self.assertIs(got, want)
                self.assertEqual(seen["env"]["HY4_Q4X"], "2")          # (the engine's own measurement)
                self.assertEqual(seen["env"]["HY4_Q4X_MEASURE"], "1")  # (again: not the verdict it remembers)
                self.assertTrue(seen["env"]["HY4_Q4X_FILE"].endswith("hy4-cpu-lane.txt"))
                self.assertEqual(seen["cmd"][1], "--serve")
                self.assertNotIn("HY4_Q4X", c.get("env") or {})        # (the engine remembers it, not the config)
                self.assertEqual(c["installer"]["q4x_calibration"]["result"], "on" if want else "off")
                self.assertIs(maya.q4x_config_migrate(Path(t) / "maya-hy4-flash-maya-xs.json", c), c)

    def test_nothing_to_decide_or_unfinished(self):
        with tempfile.TemporaryDirectory() as t:
            got, c, _, _ = self.run_check(Path(t), "INFO every routed expert in VRAM")
            self.assertIsNone(got)
            self.assertNotIn("HY4_Q4X", c.get("env") or {})            # (the engine's default stays)
            self.assertEqual(c["installer"]["q4x_calibration"]["result"], "n/a")
        with tempfile.TemporaryDirectory() as t:
            got, c, _, said = self.run_check(Path(t), "INFO CPU lane: ... 4-bit copies - re-coded")
            self.assertIsNone(got)
            self.assertNotIn("HY4_Q4X", c.get("env") or {})
            self.assertEqual(c["installer"]["q4x_calibration"]["result"], "unmeasured")
            self.assertTrue(any("did not finish" in s for s in said))


if __name__ == "__main__":
    unittest.main()
