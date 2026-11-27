"""tools/test_setup_ling.py - Ling-3.2-air in the setup (maya.py set_up_ling and its steps): its names and ids, its
own defaults (inclusionAI's generation_config.json), the download table (a size is offered only once its files are
listed and pinned), the model step (sha256-checked), the tokenizer with the model's own chat template, the config
(one a size, the user's choices kept with a .bak, disk restore points off), the picker's rows and the catalog.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import maya  # noqa: E402
import ling_fit as LINGF  # noqa: E402
import maya_quants as Q  # noqa: E402

S_NAME = "Ling-3.2-air-Maya-S-IQ2_XXS.gguf"
S_BYTES = 36912008640
S_SHA = "9fed99fc3beff053b96f8ef1cc40bb1b1cf926e201a121803d38f7927f60dd74"
M_NAME = "Ling-3.2-air-Maya-M-IQ2_S.gguf"
M_BYTES = 45971705280
M_SHA = "1b43c9956d6ac3706744b0f043113a9b4d9269d51f77a56fb506d1332c9f4a88"
L_FILES = {"Ling-3.2-air-Maya-L-IQ3_S-00001-of-00002.gguf": [49602151040, "aeed7796e48c713c6b1d5efd8c0db8c231fb8a0908542be881da6f0cea330def"],
           "Ling-3.2-air-Maya-L-IQ3_S-00002-of-00002.gguf": [9581612064, "799e308aa227dc1faaa52a414d67c87b0d9adccd387c1c265b8eeb2b7a4842c8"]}
V_NAME = "Ling-3.2-air-mmproj-BF16.gguf"
V_BYTES = 869332480
V_SHA = "4a6c6063070714617b62093baba9dfb040e4bee1f9019be6c81b6a16880117bb"


def args(**kw):
    base = dict(model=maya.LING, ling_quant=None, hy4_quant=None, mm3_quant=None, ds4_quant=None, qwen_quant=None,
                model_dir=None, models_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, check=False,
                port=None, host=None, api_key=None, env=[], no_vision=False, context=None, gpus=None, gpu=None,
                nvcc=None, repack=False, parallel=None, drafting=None, force_fit=False)
    base.update(kw)
    return SimpleNamespace(**base)


def manifest(revision="", l_files=None):
    """The table as it is, with another commit (a published one) or files for a size not uploaded yet."""
    man = json.loads(maya.LING_MANIFEST.read_text(encoding="utf-8"))
    man["revision"] = revision
    if l_files is not None:
        man["quants"]["Ling-3.2-air-Maya-L"]["files"] = l_files
    return man


def with_manifest(man):
    """maya.py and tools/ling_fit.py both read the table: the same one for both."""
    return (mock.patch.object(LINGF, "download_manifest", lambda: man),
            mock.patch.object(maya, "read_json", lambda p, *a, **k: man if Path(p) == maya.LING_MANIFEST
                              else REAL_READ(p, *a, **k)))


REAL_READ = maya.read_json


class Names(unittest.TestCase):
    def test_choose_family_and_title(self):
        self.assertEqual(maya.choose_family(args(), {}), "ling")
        self.assertEqual(maya.setup_title({"installer": {"family": "ling"}}), "Ling-3.2-air")

    def test_ids_and_names(self):
        self.assertEqual(Q.resolve("ling"), ("ling", None))
        self.assertEqual(Q.resolve("Ling-3.2-air"), ("ling", None))
        for name in ("ling-3.2-air-maya-s", "Ling-3.2-air-Maya-S", "LING-3.2-AIR-MAYA-S"):
            self.assertEqual(Q.resolve(name), ("ling", "maya-s"), name)
        self.assertIsNone(Q.resolve("ling-maya-s"))
        self.assertIsNone(Q.resolve("ling-3.2-air-maya-xs"))   # (no such size)
        self.assertEqual(Q.shown("ling", "maya-s", 36.91), "Ling-3.2-air · Maya-S (37 GB)")
        self.assertEqual(Q.shown("ling"), "Ling-3.2-air")
        self.assertEqual(Q.full_name("ling", "maya-s"), "Ling-3.2-air-Maya-S")
        self.assertEqual(Q.config_stem("ling", None), "maya-ling-3.2-air")
        self.assertEqual(Q.config_stem("ling", "maya-s"), "maya-ling-3.2-air-maya-s")
        self.assertEqual(maya.LING_QUANTS, {"maya-s": "Ling-3.2-air-Maya-S", "maya-m": "Ling-3.2-air-Maya-M",
                                            "maya-l": "Ling-3.2-air-Maya-L"})

    def test_label_and_key(self):
        cfg = {"exe": "/b/maya-ling", "installer": {"family": "ling", "ling_quant": "maya-s",
                                                     "quant": "Ling-3.2-air-Maya-S"}}
        self.assertEqual(Q.config_label(cfg), "Ling-3.2-air · Maya-S")
        self.assertEqual(Q.config_key(cfg), "maya-s")
        own = {"exe": "/b/maya-ling", "installer": {"family": "ling", "quant": "local"}}
        self.assertEqual((Q.config_label(own), Q.config_key(own)), ("Ling-3.2-air", None))

    def test_its_own_defaults(self):
        """inclusionAI/Ling-3.2-air's generation_config.json: do_sample, temperature 1.0, top_k 20, top_p 0.95 - no
        min-p, no penalties; thinking on (its template's default).  Every size and a GGUF of the user's own the same."""
        want = ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "high")
        for inst in ({"family": "ling", "ling_quant": "maya-s"}, {"family": "ling", "ling_quant": "maya-l"},
                     {"family": "ling", "quant": "local"}):
            self.assertEqual(Q.model_defaults({"exe": "/b/maya-ling", "installer": inst}), want)
        self.assertEqual(Q.model_defaults({"exe": "build/maya-ling", "args": ["--gguf", "x.gguf"]}), want)
        man = LINGF.download_manifest()
        self.assertEqual((man["sampling"], man["reasoning_effort"]), want)

    def test_model_option(self):
        a = args(model="ling-3.2-air-maya-s")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.ling_quant), (maya.LING, "maya-s"))
        a = args(model="ling-3.2-air-maya-m")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.ling_quant), (maya.LING, "maya-m"))
        a = args(model="ling-3.2-air-maya-l")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.ling_quant), (maya.LING, "maya-l"))
        for i in ("ling-3.2-air-maya-s", "ling-3.2-air-maya-m", "ling-3.2-air-maya-l"):
            self.assertIn(i, maya.named_quants())
        # (the other models' ids: all of them, in their order)
        self.assertEqual([i for i in maya.named_quants() if not i.startswith("ling-")],
                         [i for f, _, _, i in Q.QUANTS if f != "ling"])


class Table(unittest.TestCase):
    def test_the_table(self):
        m = LINGF.download_manifest()
        self.assertEqual((m["name"], m["license"], m["repo"]),
                         ("Ling-3.2-air", "MIT", "peasantsmith/Ling-3.2-air-Maya-GGUF"))
        self.assertEqual(list(m["quants"]), ["Ling-3.2-air-Maya-S", "Ling-3.2-air-Maya-M", "Ling-3.2-air-Maya-L"])
        self.assertEqual(m["quants"]["Ling-3.2-air-Maya-S"]["files"], {S_NAME: [S_BYTES, S_SHA]})
        self.assertEqual(m["quants"]["Ling-3.2-air-Maya-M"], {"about": "IQ2_S", "files": {M_NAME: [M_BYTES, M_SHA]}})
        self.assertRegex(m["revision"], r"^[0-9a-f]{40}$")   # (the Hub repo's commit that has S, M and vision/)
        self.assertEqual(m["quants"]["Ling-3.2-air-Maya-L"], {"about": "IQ3_S", "files": L_FILES})
        self.assertEqual(m["vision"]["revision"], m["revision"])
        for q, spec in m["quants"].items():            # (a file's name carries the model, the size and its quant type)
            for name, (size, sha) in spec["files"].items():
                self.assertRegex(name, r"^" + q + r"-[A-Z0-9_]+(-\d{5}-of-\d{5})?\.gguf$")
                self.assertRegex(sha, r"^[0-9a-f]{64}$")
                self.assertGreater(size, 1 << 30)
        self.assertEqual(LINGF.listed(), ["Ling-3.2-air-Maya-S", "Ling-3.2-air-Maya-M", "Ling-3.2-air-Maya-L"])

    def test_vision_pinned(self):
        """The pictures: the mmproj from Maya's own repo (vision/ at the table's commit), by size and sha256."""
        m = maya.ling_vision_model()
        rev = LINGF.download_manifest()["revision"]
        self.assertEqual((m["repo"], m["revision"], m["folder"]), ("peasantsmith/Ling-3.2-air-Maya-GGUF", rev, "vision"))
        self.assertEqual((m["files"], m["sha256"], m["format"]), ({V_NAME: V_BYTES}, {V_NAME: V_SHA}, "mmproj"))
        self.assertEqual(maya.file_url(m, V_NAME),
                         f"https://huggingface.co/peasantsmith/Ling-3.2-air-Maya-GGUF/resolve/{rev}/vision/{V_NAME}")

    def test_offered_only_once_pinned(self):
        """A size is downloadable once the table names the commit that has its files; until then it is set up from a
        folder that has them, and has no row on screen."""
        if not LINGF.download_manifest().get("revision"):
            self.assertFalse(maya.ling_published("maya-s"))
        self.assertTrue(maya.ling_listed("maya-s"))
        self.assertTrue(maya.ling_listed("maya-m") and maya.ling_listed("maya-l"))
        for k in ("maya-xs", None):
            self.assertFalse(maya.ling_listed(k))
            self.assertFalse(maya.ling_published(k))
        p1, p2 = with_manifest(manifest("0" * 40))
        with p1, p2:
            self.assertTrue(maya.ling_published("maya-s"))
            self.assertTrue(maya.ling_published("maya-m"))
            self.assertTrue(maya.ling_published("maya-l"))
            e = maya.ling_entry("maya-l")   # (two files: the first shard is the one the engine opens)
            self.assertEqual((list(e["files"]), e["gguf"]), (["Ling-3.2-air-Maya-L-IQ3_S-00001-of-00002.gguf", "Ling-3.2-air-Maya-L-IQ3_S-00002-of-00002.gguf"], "Ling-3.2-air-Maya-L-IQ3_S-00001-of-00002.gguf"))
            e = maya.ling_entry("maya-s")
            self.assertEqual((e["repo"], e["revision"], e["files"], e["sha256"], e["gguf"]),
                             ("peasantsmith/Ling-3.2-air-Maya-GGUF", "0" * 40, {S_NAME: S_BYTES}, {S_NAME: S_SHA}, S_NAME))
            self.assertAlmostEqual(e["download_gb"], 36.91, places=2)
            self.assertEqual(maya.hf_url(e["repo"], e["revision"], e["folder"], S_NAME),
                             f"https://huggingface.co/peasantsmith/Ling-3.2-air-Maya-GGUF/resolve/{'0' * 40}/{S_NAME}")
            self.assertIn("Ling-3.2-air-Maya-S (37 GB), Ling-3.2-air-Maya-M (46 GB), Ling-3.2-air-Maya-L (59 GB)", maya.ling_about())

    def test_sizes_from_the_files(self):
        """The planner sizes a listed size by its file, a GGUF of the user's own by the file on the disk; a size with
        no file has no figure of its own."""
        s = LINGF.quant_spec("Ling-3.2-air-Maya-S")
        self.assertAlmostEqual(s["experts_gib"], S_BYTES / LINGF.GIB - LINGF.DENSE_GIB, places=1)
        m = LINGF.quant_spec("Ling-3.2-air-Maya-M")
        self.assertAlmostEqual(m["experts_gib"], M_BYTES / LINGF.GIB - LINGF.DENSE_GIB, places=1)
        self.assertAlmostEqual(LINGF.quant_spec("Ling-3.2-air-Maya-L")["experts_gib"],
                               (49602151040 + 9581612064) / LINGF.GIB - LINGF.DENSE_GIB, places=1)
        self.assertEqual(LINGF.quant_spec("Ling-3.2-air-Maya-XS"), LINGF.quant_spec(None))
        self.assertAlmostEqual(LINGF.quant_spec("local", gguf_gib=50.0)["experts_gib"], 50.0 - LINGF.DENSE_GIB, places=1)
        # (a hint never names a size that does not exist)
        hint = LINGF.smaller_quant_hint([8.0], 32.0, 32768, "f16", "local", gguf_gib=60.0)
        self.assertNotIn("Maya-L", hint)


class Picker(unittest.TestCase):
    def rows(self, prev=None):
        got = {}

        def pick(title, intro, opts, default, yes, *x):
            got.update(rows=[o[0] for o in opts], about={o[0]: o[2] for o in opts}, default=default)
            return default
        with mock.patch.object(maya, "choose", pick), mock.patch.object(maya.S, "gpus", return_value=[]):
            fam = maya.choose_family(args(model=None), prev or {})
        return got, fam

    def test_no_row_until_a_size_is_published(self):
        p1, p2 = with_manifest(manifest(""))
        with p1, p2:
            got, _ = self.rows()
        self.assertNotIn("Ling-3.2-air", got["rows"])
        self.assertEqual(got["rows"][-1], "Inkling 2 Mini")

    def test_its_row_once_published_and_picked(self):
        p1, p2 = with_manifest(manifest("0" * 40))
        with p1, p2:
            got, _ = self.rows()
            self.assertEqual(got["rows"][-2:], ["Ling-3.2-air", "Inkling 2 Mini"])
            for part in ("inclusionAI's Ling-3.2-air", "Ling-3.2-air-Maya-S (37 GB)", "Linux and Windows", "thinking on or off",
                         "one answer at a time"):
                self.assertIn(part, got["about"]["Ling-3.2-air"])
            self.assertIn("Ling-3.2-air-Maya-M (46 GB)", got["about"]["Ling-3.2-air"])
            self.assertIn("Ling-3.2-air-Maya-L (59 GB)", got["about"]["Ling-3.2-air"])
            got, fam = self.rows({"exe": "/b/maya-ling", "installer": {"family": "ling"}})
            self.assertEqual((got["rows"][got["default"]], fam), ("Ling-3.2-air", "ling"))

    def test_the_other_rows_are_as_before(self):
        p1, p2 = with_manifest(manifest(""))
        with p1, p2:
            before, _ = self.rows()
        p1, p2 = with_manifest(manifest("0" * 40))
        with p1, p2:
            after, _ = self.rows()
        self.assertEqual([r for r in after["rows"] if r != "Ling-3.2-air"], before["rows"])
        for r in before["rows"]:
            self.assertEqual(after["about"][r], before["about"][r])

    def test_the_size_chosen(self):
        pc = {"gpus": [{"index": 0, "vram_gb": 32.0}]}
        self.assertEqual(maya.ling_choose_quant(args(ling_quant="maya-s"), pc, {}), "maya-s")
        self.assertIsNone(maya.ling_choose_quant(args(model_dir="/x"), pc, {}))
        before = {"exe": "/b/maya-ling", "installer": {"family": "ling", "ling_quant": "maya-s"}}
        self.assertEqual(maya.ling_choose_quant(args(), pc, before), "maya-s")
        own = {"exe": "/b/maya-ling", "installer": {"family": "ling", "quant": "local", "model_dir": "/x"}}
        self.assertIsNone(maya.ling_choose_quant(args(), pc, own))
        p1, p2 = with_manifest(manifest("0" * 40))
        with p1, p2, mock.patch.object(maya, "mem_gb", lambda: (128.0, 100.0)):   # (the largest that fits: Maya-L)
            self.assertEqual(maya.ling_choose_quant(args(), pc, {}), "maya-l")
        with p1, p2, mock.patch.object(maya, "mem_gb", lambda: (24.0, 20.0)):     # (Maya-M's room, not Maya-L's)
            self.assertEqual(maya.ling_choose_quant(args(), pc, {}), "maya-m")
        small = {"gpus": [{"index": 0, "vram_gb": 8.0}]}
        with p1, p2, mock.patch.object(maya, "mem_gb", lambda: (36.0, 30.0)):     # (room for Maya-S only: Maya-S)
            self.assertEqual(maya.ling_choose_quant(args(), small, {}), "maya-s")
        p1, p2 = with_manifest(manifest(""))
        with p1, p2, self.assertRaises(SystemExit):    # (nothing downloadable and nothing named: said, with the way)
            maya.ling_choose_quant(args(), pc, {})


class Steps(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.said = []
        for name in ("ok", "say", "step", "warn"):
            p = mock.patch.object(maya, name, lambda *a, **kw: self.said.append(" ".join(map(str, a))))
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(maya, "rotational", lambda d: False)
        p.start()
        self.addCleanup(p.stop)

    def test_windows_nvidia_passes_step_one_amd_refused(self):
        """v1.1.83: a Windows PC with an NVIDIA card goes on to step 2 (no "Linux only" refusal); AMD on Windows is
        refused in one line before anything."""
        class Step2(Exception):
            pass

        def choose(*a, **kw):
            raise Step2

        for backend, gpus in (("cuda", [{"vram_gb": 8.0, "cc": 86}]), ("hip", [{"vram_gb": 24.0}])):
            with mock.patch.object(maya, "WIN", True), \
                    mock.patch.object(maya, "check_pc", lambda a, *x, **k: {"backend": backend, "gpus": gpus}), \
                    mock.patch.object(maya, "mem_gb", lambda: (64.0, 60.0)), \
                    mock.patch.object(maya, "ling_avx2_available", lambda: True), \
                    mock.patch.object(maya, "ling_choose_quant", choose), \
                    mock.patch.object(maya, "fail", lambda *m: (_ for _ in ()).throw(SystemExit(" ".join(m)))):
                if backend == "cuda":
                    with self.assertRaises(Step2):
                        maya.set_up_ling(args(), {})
                else:
                    with self.assertRaises(SystemExit) as cm:
                        maya.set_up_ling(args(), {})
                    self.assertIn("AMD", str(cm.exception.code))

    @unittest.skipUnless(sys.platform == "win32", "Windows' CPU feature bit")
    def test_avx2_check_asks_windows(self):
        self.assertEqual(maya.ling_avx2_available(), maya.S.cpu_info()[1])

    def small_table(self, revision=""):
        """The table with a small file in the size's place: its real size and sha256."""
        body = b"GGUF" + b"\1" * 252
        man = manifest(revision)
        man["quants"]["Ling-3.2-air-Maya-S"]["files"] = {S_NAME: [len(body), hashlib.sha256(body).hexdigest()]}
        return man, body

    def test_model_step_checks_the_sha256(self):
        man, body = self.small_table()
        d = self.root / "m"
        d.mkdir()
        p1, p2 = with_manifest(man)
        with p1, p2:
            (d / S_NAME).write_bytes(body)
            self.assertEqual(maya.ling_model_step(args(), d, "maya-s"), d / S_NAME)
            (d / S_NAME).write_bytes(b"GGUF" + b"\2" * 252)   # (the same size, other bytes)
            with self.assertRaises(SystemExit):
                maya.ling_model_step(args(), d, "maya-s")

    def test_model_step_before_and_after_the_upload(self):
        d = self.root / "m"
        d.mkdir()
        man, body = self.small_table()
        p1, p2 = with_manifest(man)
        with p1, p2, mock.patch.object(maya, "offer_download") as offer:
            with self.assertRaises(SystemExit):        # (not pinned yet: nothing is downloaded, the way is said)
                maya.ling_model_step(args(), d, "maya-s")
            offer.assert_not_called()
        man, body = self.small_table("0" * 40)
        p1, p2 = with_manifest(man)

        def fetch(a, label, folder, files, m):
            self.assertEqual((m["repo"], m["revision"]), ("peasantsmith/Ling-3.2-air-Maya-GGUF", "0" * 40))
            files[0].write_bytes(body)
            return True
        with p1, p2, mock.patch.object(maya, "offer_download", fetch):
            self.assertEqual(maya.ling_model_step(args(), d, "maya-s"), d / S_NAME)
        with p1, p2, mock.patch.object(maya, "offer_download", lambda *a: False):
            (d / S_NAME).unlink()
            self.assertIsNone(maya.ling_model_step(args(), d, "maya-s"))   # (declined)

    def test_model_step_with_the_users_own_file(self):
        d = self.root / "own"
        d.mkdir()
        with self.assertRaises(SystemExit):
            maya.ling_model_step(args(), d, None)
        (d / "Ling-3.2-air-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF" + b"\0" * 60)
        (d / "Ling-3.2-air-Q8_0-00002-of-00002.gguf").write_bytes(b"GGUF" + b"\0" * 28)
        (d / "Ling-3.2-air-mmproj-BF16.gguf").write_bytes(b"GGUF")
        got = maya.ling_model_step(args(), d, None)
        self.assertEqual(got.name, "Ling-3.2-air-Q8_0-00001-of-00002.gguf")
        self.assertAlmostEqual(maya.ling_gguf_gib(got) * LINGF.GIB, 96.0)   # (both shards, not the vision file)

    def small_vision(self):
        """The table with a small file in the vision file's place: its real size and sha256."""
        body = b"GGUF" + b"\3" * 124
        man = manifest("0" * 40)
        man["vision"]["files"] = {V_NAME: [len(body), hashlib.sha256(body).hexdigest()]}
        return man, body

    def test_vision_fresh_downloads_from_our_repo(self):
        """Nothing on the disk: the vision file from Maya's repo, the pinned commit, folder vision/, beside the model;
        sha256-checked; --mmproj names it.  A declined download: text only."""
        d = self.root / "m"
        d.mkdir()
        (d / S_NAME).write_bytes(b"GGUF")
        man, body = self.small_vision()
        seen = []

        def offer(a, label, folder, files, m):
            seen.append((m["repo"], m["revision"], m["folder"], folder, [f.name for f in files]))
            files[0].write_bytes(body)
            return True
        p1, p2 = with_manifest(man)
        with p1, p2, mock.patch.object(maya, "offer_download", offer):
            self.assertEqual(maya.ling_vision_step(args(), d), d / V_NAME)
            self.assertEqual(seen, [("peasantsmith/Ling-3.2-air-Maya-GGUF", man["vision"]["revision"], "vision", d,
                                     [V_NAME])])
            self.assertEqual(maya.ling_vision_step(args(), d), d / V_NAME)   # (there now: not downloaded again)
            self.assertEqual(len(seen), 1)
        (d / V_NAME).unlink()
        with p1, p2, mock.patch.object(maya, "offer_download", lambda *a: False):
            self.assertIsNone(maya.ling_vision_step(args(), d))
        self.assertTrue(any("not downloaded" in x for x in self.said))

    def test_vision_off_with_no_vision(self):
        d = self.root / "m"
        d.mkdir()
        (d / V_NAME).write_bytes(b"GGUF")
        man, _ = self.small_vision()
        p1, p2 = with_manifest(man)
        with p1, p2, mock.patch.object(maya, "offer_download") as offer:
            self.assertIsNone(maya.ling_vision_step(args(no_vision=True), d))
            offer.assert_not_called()

    def test_vision_file_already_beside_the_model_is_kept(self):
        """An mmproj beside the model under another name (an earlier one, or the user's own): used as it is, nothing
        downloaded.  The table naming no vision file: the one beside the model, else text only."""
        d = self.root / "m"
        d.mkdir()
        (d / S_NAME).write_bytes(b"GGUF")
        man, _ = self.small_vision()
        p1, p2 = with_manifest(man)
        with p1, p2, mock.patch.object(maya, "offer_download") as offer:
            (d / "mmproj-BF16.gguf").write_bytes(b"GGUF mine")
            self.assertEqual(maya.ling_vision_step(args(), d), d / "mmproj-BF16.gguf")
            offer.assert_not_called()
        none = manifest("0" * 40)
        none["vision"]["files"] = {}
        p1, p2 = with_manifest(none)
        with p1, p2, mock.patch.object(maya, "offer_download") as offer:
            self.assertEqual(maya.ling_vision_step(args(), d), d / "mmproj-BF16.gguf")
            (d / "mmproj-BF16.gguf").unlink()
            self.assertIsNone(maya.ling_vision_step(args(), d))
            offer.assert_not_called()

    def test_vision_sha256_checked(self):
        """The table's file whole but with other bytes: removed, and the setup stops (the next run downloads it)."""
        d = self.root / "m"
        d.mkdir()
        man, body = self.small_vision()
        (d / V_NAME).write_bytes(b"GGUF" + b"\4" * 124)   # (the same size, other bytes)
        p1, p2 = with_manifest(man)
        with p1, p2, mock.patch.object(maya, "offer_download") as offer, self.assertRaises(SystemExit):
            maya.ling_vision_step(args(), d)
        offer.assert_not_called()
        self.assertFalse((d / V_NAME).exists())

    def test_tokenizer_with_the_models_own_template(self):
        d = self.root / "m"
        d.mkdir()
        gguf = d / S_NAME
        gguf.write_bytes(b"GGUF" + b"\0" * 60)
        runs = []

        def run(cmd, **kw):
            runs.append([str(c) for c in cmd])
            out = Path(cmd[cmd.index("--out") + 1]) / "tokenizer"
            out.mkdir(parents=True, exist_ok=True)
            (out / "vocab.json").write_text("{}", encoding="utf-8")
            (out / "chat_template.jinja").write_bytes(Path(cmd[cmd.index("--chat-template") + 1]).read_bytes())
            return SimpleNamespace(returncode=0)
        with mock.patch.object(maya, "run", run):
            tok = maya.ling_tokenizer_step(args(), d, gguf)
            self.assertEqual(tok, d / "maya-ling" / "tokenizer")
            cmd = runs[0]
            self.assertEqual(cmd[cmd.index("--chat-template") + 1], str(ROOT / "tools" / "ling" / "chat_template.jinja"))
            self.assertIn("--check", cmd)
            self.assertEqual((tok / "chat_template.jinja").read_bytes(), maya.LING_TEMPLATE.read_bytes())
            maya.ling_tokenizer_step(args(), d, gguf)   # (nothing changed: not exported again)
            self.assertEqual(len(runs), 1)
            (d / "chat_template.jinja").write_text("{# mine #}", encoding="utf-8")   # (the folder's own wins)
            tok = maya.ling_tokenizer_step(args(), d, gguf)
            self.assertEqual(len(runs), 2)
            self.assertEqual((tok / "chat_template.jinja").read_text(encoding="utf-8"), "{# mine #}")

    def test_the_template_is_the_official_text(self):
        """tools/ling/chat_template.jinja: a three-line licence comment, then inclusionAI's chat_template.jinja byte
        for byte (its sha256 as published with the model)."""
        body = maya.LING_TEMPLATE.read_bytes().split(b"\n", 3)
        self.assertTrue(body[0].startswith(b"{#-") and body[2].rstrip().endswith(b"-#}"))
        self.assertEqual(hashlib.sha256(body[3]).hexdigest(),
                         "dbb08479531af61d736c16f92a8271f66eced23c8f4b403c808352e5b2631875")

    def write(self, a, prev=None, key="maya-s", ctx=32768, drafting="auto", mmproj=True, gpus=1, meta=None, pc=None):
        td = self.root
        fake_root = td / "repo"
        (fake_root / "data").mkdir(parents=True, exist_ok=True)
        gguf = td / S_NAME
        gguf.write_bytes(b"GGUF")
        mm = td / "Ling-3.2-air-mmproj-BF16.gguf"
        mm.write_bytes(b"GGUF")
        tok = td / "tok"
        tok.mkdir(exist_ok=True)
        pc = pc or {"backend": "cuda", "gpus": [{"index": 3 + i, "vram_gb": 24.0, "cc": 86} for i in range(gpus)]}
        with mock.patch.object(maya, "ROOT", fake_root), \
                mock.patch.object(maya, "LING_EXE", fake_root / "build" / "maya-ling"), \
                mock.patch.object(maya, "LING_PROFILE", fake_root / "data" / "none.bin"), \
                mock.patch.object(maya, "write_run_script", return_value=fake_root / "run.sh"):
            p = maya.write_config_ling(a, pc, meta or {}, gguf, tok, ctx, "f16", drafting, td, td,
                                       mm if mmproj else None, maya.ling_table(key) or "local", prev or {}, key)
        return p, json.loads(p.read_text(encoding="utf-8"))

    def test_the_config(self):
        p1, c1 = self.write(args(port=8099), meta={"lib_dirs": ["/cuda/lib64"]})
        self.assertEqual(p1.name, "maya-ling-3.2-air-maya-s.json")
        self.assertEqual(c1["port"], 8099)
        self.assertEqual(c1["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(c1["reasoning_effort"], "high")
        self.assertEqual(c1["lib_dirs"], ["/cuda/lib64"])
        self.assertEqual(c1["model_name"], "ling-3.2-air")
        self.assertEqual(c1["gpu"], [3])
        inst = c1["installer"]
        self.assertEqual((inst["family"], inst["ling_quant"], inst["quant"], inst["drafting"]),
                         ("ling", "maya-s", "Ling-3.2-air-Maya-S", "auto"))
        ar = c1["args"]
        self.assertEqual(ar[:5], ["--serve", "--gguf", str(self.root / S_NAME), "--max-context", "32768"])
        self.assertEqual(ar[ar.index("--kv") + 1], "fp16")
        self.assertEqual(ar[ar.index("--drafting") + 1], "auto")
        self.assertIn("--mmproj", ar)
        self.assertIn("--cal-file", ar)
        self.assertNotIn("--gpus", ar)                 # (one GPU: the engine's own)
        self.assertNotIn("--parallel", ar)             # (maya-ling answers one request at a time)
        self.assertNotIn("--restore-dir", ar)          # (disk restore points stay off by default)
        self.assertNotIn("restore_dir", inst)
        self.assertNotIn("loop_guard", c1)             # (on by default, as for every model: no key needed)
        self.assertEqual(Q.config_label(c1), "Ling-3.2-air · Maya-S")
        self.assertEqual(Q.model_defaults(c1), (c1["sampling"], c1["reasoning_effort"]))

    def test_a_setup_again_keeps_the_users_choices(self):
        _, c1 = self.write(args(port=8099))
        p = self.root / "repo" / "maya-ling-3.2-air-maya-s.json"
        c1["args"] += ["--cpu-threads", "12", "--restore-dir", "/r"]
        c1.update(loop_guard=False, api_key="k", env={"LING_WATCHDOG_S": "9"},
                  sampling={"temperature": 0.2}, reasoning_effort="none")
        p.write_text(json.dumps(c1), encoding="utf-8")
        _, c2 = self.write(args(), ctx=65536, drafting="off", mmproj=False, gpus=3)
        ar = c2["args"]
        self.assertEqual((c2["port"], c2["api_key"], c2["loop_guard"]), (8099, "k", False))
        self.assertEqual(c2["env"], {"LING_WATCHDOG_S": "9"})
        self.assertEqual(ar[ar.index("--cpu-threads") + 1], "12")
        self.assertEqual(ar[ar.index("--restore-dir") + 1], "/r")   # (the user's own opt-in stays)
        self.assertEqual(ar[ar.index("--max-context") + 1], "65536")
        self.assertEqual(ar[ar.index("--drafting") + 1], "off")
        self.assertEqual(ar[ar.index("--gpus") + 1], "0,1,2")
        self.assertEqual(c2["gpu"], [3, 4, 5])
        self.assertNotIn("--mmproj", ar)
        # (the model's own sampling and level every setup, as for every model)
        self.assertEqual((c2["sampling"], c2["reasoning_effort"]),
                         ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "high"))
        self.assertEqual(len(list((self.root / "repo").glob("maya-ling-3.2-air-maya-s.json.bak-*"))), 1)

    def test_a_config_a_size_and_one_for_the_users_own_file(self):
        p1, _ = self.write(args())
        p2, c2 = self.write(args(), key=None)
        self.assertEqual((p1.name, p2.name), ("maya-ling-3.2-air-maya-s.json", "maya-ling-3.2-air.json"))
        self.assertTrue(p1.exists() and p2.exists())
        self.assertNotIn("ling_quant", c2["installer"])
        self.assertEqual(c2["installer"]["quant"], "local")

    def test_amd_linux_builds_the_hip_engine(self):
        # an AMD APU under Linux: the PC check is the HIP one (pictures on), the fit counts the APU's memory, and
        # step 4 compiles maya-ling for that PC (build_step on the HIP pc: compile_engine_hip's target)
        pc = {"backend": "hip", "archs": ["gfx1151"], "gpus": [{"index": 0, "name": "AMD Radeon 8065S",
                                                               "arch": "gfx1151", "vram_gb": 0.5, "vendor": "amd"}]}
        seen = {}

        class Built(Exception):
            pass

        def build(a, pc_, llama, target):
            seen["build"] = (pc_["backend"], target)
            raise Built

        def check(a, family="glm", *rest, **kw):
            seen["family"] = family
            return pc

        p1, p2 = with_manifest(manifest("0" * 40))
        with p1, p2, mock.patch.object(maya, "check_pc", check), mock.patch.object(maya, "WIN", False), \
                mock.patch.object(maya, "mem_gb", lambda: (187.0, 170.0)), \
                mock.patch.object(maya, "hy4_apu_gib", return_value=168.0), \
                mock.patch.object(maya, "ling_avx2_available", return_value=True), \
                mock.patch.object(maya, "tools_step", return_value=Path("/llama")), \
                mock.patch.object(maya, "build_step", build), self.assertRaises(Built):
            maya.set_up_ling(args(backend="hip", ling_quant="maya-s", models_dir=str(self.root)), {})
        self.assertEqual(seen, {"family": "ling", "build": ("hip", "maya-ling")})
        # on Windows an AMD PC is refused in one line, before anything is built
        with mock.patch.object(maya, "check_pc", check), mock.patch.object(maya, "WIN", True), \
                mock.patch.object(maya, "mem_gb", lambda: (64.0, 50.0)), \
                mock.patch.object(maya, "build_step", build), self.assertRaises(SystemExit):
            maya.set_up_ling(args(backend="hip", ling_quant="maya-s"), {})

    def test_amd_config_keeps_room_free(self):
        apu = {"backend": "hip", "gpus": [{"index": 0, "name": "AMD Radeon 8065S", "arch": "gfx1151", "vram_gb": 0.5,
                                           "vendor": "amd"}]}
        with mock.patch.object(maya, "mem_gb", lambda: (187.0, 170.0)):
            p, c = self.write(args(backend="hip"), pc=apu, meta={"lib_dirs": ["/opt/rocm/lib"]})
        self.assertEqual(c["backend"], "hip")
        self.assertEqual(c["args"][c["args"].index("--reserve-gb") + 1], "18.7")   # (max(8 GB, 10% of the RAM))
        p.unlink()                                      # (a fresh setup each: a setup again keeps the value before)
        card = {"backend": "hip", "gpus": [{"index": 0, "name": "Radeon RX 7900 XT", "arch": "gfx1100",
                                            "vram_gb": 20.0, "vendor": "amd"}]}
        p, c = self.write(args(backend="hip"), pc=card)
        self.assertEqual(c["args"][c["args"].index("--reserve-gb") + 1], "3.0")
        p.unlink()
        _, c = self.write(args())                       # (NVIDIA: as before, the engine's own default)
        self.assertNotIn("--reserve-gb", c["args"])

    def test_memory_for_the_fit(self):
        # NVIDIA: the cards' VRAM and the PC's RAM, as before
        nv = {"backend": "cuda", "gpus": [{"index": 0, "vram_gb": 32.0, "cc": 70}, {"index": 1, "vram_gb": 24.0}]}
        self.assertEqual(maya.ling_memory(nv, 183.0), ([32.0, 24.0], 183.0))
        # an AMD card: its VRAM less what the engines leave the desktop
        card = {"backend": "hip", "gpus": [{"index": 0, "name": "Radeon RX 7900 XT", "arch": "gfx1100",
                                            "vram_gb": 20.0, "vendor": "amd"}]}
        v, ram = maya.ling_memory(card, 64.0)
        self.assertAlmostEqual(v[0], 20.0 - maya.hip_reserve_mib(card["gpus"]) / 1024.0)
        self.assertEqual(ram, 64.0)
        # an APU (Gorgon Halo): the GPU's share of the one memory, and only the rest of the RAM beside it
        apu = {"backend": "hip", "gpus": [{"index": 0, "name": "AMD Radeon 8065S", "arch": "gfx1151", "vram_gb": 0.5,
                                           "vendor": "amd"}]}
        with mock.patch.object(maya, "hy4_apu_gib", return_value=168.0):
            v, ram = maya.ling_memory(apu, 187.0)
        self.assertEqual(v, [168.0])
        self.assertAlmostEqual(ram, 19.0)
        word, _ = LINGF.verdict(v, ram, 32768, "f16", "Ling-3.2-air-Maya-S")
        self.assertEqual(word, "fits")
        self.assertTrue(LINGF.plan(v, 32768, "f16", "Ling-3.2-air-Maya-S", ram_gb=ram)["resident"])

    def test_drafting(self):
        self.assertEqual(maya.ling_drafting(args(), []), "auto")
        self.assertEqual(maya.ling_drafting(args(drafting="off"), ["--drafting", "auto"]), "off")
        self.assertEqual(maya.ling_drafting(args(drafting="on"), []), "mtp")
        self.assertEqual(maya.ling_drafting(args(drafting="measure"), []), "auto")
        self.assertEqual(maya.ling_drafting(args(), ["--drafting", "off"]), "off")   # (the config before's)
        self.assertEqual(maya.ling_drafting(args(), ["--drafting", "mtp"]), "mtp")

    def test_reset_ling_calibration(self):
        cf = self.root / "ling-calibration.txt"
        cf.write_text("key\tmode=gpu\n", encoding="utf-8")
        cfg = {"installer": {"family": "ling", "cal_file": str(cf)}}
        self.assertTrue(maya.reset_ling_calibration(cfg))
        self.assertFalse(cf.exists())

    def test_one_answer_at_a_time(self):
        with self.assertRaises(SystemExit):
            maya.check_parallel(args(parallel=2), "ling")
        self.assertNotIn("ling", maya.PARALLEL_FAMILIES)

    def test_bench_knows_it(self):
        _, cfg = self.write(args())
        self.assertEqual(Path(maya.bench_exe(cfg)).name, maya.LING_EXE.name)
        self.assertEqual(maya.bench_model_name(cfg), "ling-3.2-air Ling-3.2-air-Maya-S")


class Catalog(unittest.TestCase):
    """The Models page's download list (serve/model_pull.py): Ling's sizes only once they can be downloaded, on
    NVIDIA under Linux or Windows; every other row as before."""

    def catalog(self, man, platform="linux", backend="cuda"):
        from serve import model_pull
        p1, p2 = with_manifest(man)
        with p1, p2, mock.patch.object(model_pull.sys, "platform", platform), tempfile.TemporaryDirectory() as td:
            return model_pull.catalog(Path(td), glm_sizes={}, cfg={"backend": backend}, ccs=[70], other_sizes={})

    def test_rows(self):
        before = self.catalog(manifest(""))
        self.assertEqual([r for r in before if r["family"] == "ling"], [])
        after = self.catalog(manifest("0" * 40))
        ling = [r for r in after if r["family"] == "ling"]
        self.assertEqual([(r["id"], r["name"], r["group"], r["size_gb"]) for r in ling],
                         [("ling-3.2-air-maya-s", "Ling-3.2-air · Maya-S", "maya", 36.9),
                          ("ling-3.2-air-maya-m", "Ling-3.2-air · Maya-M", "maya", 46.0),
                          ("ling-3.2-air-maya-l", "Ling-3.2-air · Maya-L", "maya", 59.2)])
        self.assertEqual([r for r in after if r["family"] != "ling"], before)
        # (Windows + NVIDIA: the same rows, v1.1.83)
        win = [r for r in self.catalog(manifest("0" * 40), "win32", "cuda") if r["family"] == "ling"]
        self.assertEqual([(r["id"], r["size_gb"]) for r in win], [(r["id"], r["size_gb"]) for r in ling])
        self.assertTrue(all("Linux" not in r["note"] for r in win))
        for platform, backend in (("win32", "hip"), ("darwin", "metal")):
            rows = self.catalog(manifest("0" * 40), platform, backend)
            self.assertEqual([r for r in rows if r["family"] == "ling"], [], (platform, backend))
        # AMD under Linux: the same sizes (the setup's own check says which AMD GPUs)
        amd = [r for r in self.catalog(manifest("0" * 40), "linux", "hip") if r["family"] == "ling"]
        self.assertEqual([r["id"] for r in amd], [r["id"] for r in ling])
        self.assertIn("AMD", amd[0]["note"])

    def test_setup_command(self):
        from serve import model_pull
        cmd = model_pull.setup_command("ling-3.2-air-maya-s", {"installer": {"models_dir": "/m"}})
        self.assertEqual(cmd[cmd.index("--model") + 1], "ling-3.2-air-maya-s")
        self.assertIn("--download-model", cmd)


if __name__ == "__main__":
    unittest.main()
