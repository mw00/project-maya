"""--import-model (#666): files downloaded some other way are found by size + sha256, whatever their names, and placed in
the models folder's layout.  Fake small files and a patched catalog; no GPU, no network."""
import argparse
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya
import maya_import as MI


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


BIG = b"A" * 5000
BIG2 = b"B" * 7000
SMALL = b'{"a": 1}'
SHARED = b"GGUF" + b"v" * 2000
SHAPE = b"GGUF" + b"x" * 3000          # a GGUF whose size the table does not list


class ImportTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(MI, "SHARED_SMALL", 1000)     # (the fake files are small)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(MI, "MIN_UNLISTED", 1000)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.models = self.root / "models"
        self.src = self.root / "downloads"
        self.src.mkdir()
        self.cat = [
            {"label": "Model-A", "rel": "Model-A/a-00001.safetensors", "size": len(BIG), "sha": [sha(BIG)]},
            {"label": "Model-A", "rel": "Model-A/a-00002.safetensors", "size": len(BIG2), "sha": [sha(BIG2)]},
            {"label": "Model-A", "rel": "Model-A/config.json", "size": len(SMALL), "sha": [sha(SMALL)]},
            {"label": "Model-B", "rel": "Model-B/shard-00001.gguf", "size": None, "sha": [sha(SHAPE)]},
            {"label": "Model-A", "rel": "Model-A/vision/mm.gguf", "size": None, "sha": [sha(SHARED)]},
            {"label": "Model-B", "rel": "Model-B/vision/mm.gguf", "size": None, "sha": [sha(SHARED)]},
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def put(self, rel, data):
        p = self.src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def run_import(self, copy=True, yes=True):
        a = argparse.Namespace(models_dir=str(self.models), copy=copy, yes=yes)
        with patch.object(maya, "import_catalog", return_value=self.cat), patch.object(maya, "configs", return_value=[]), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn") as warn:
            code = maya.import_model(a, str(self.src))
        return code, warn

    def test_found_by_content_not_name_moved_and_recorded(self):
        a = self.put("xl/new folder/IMG_0001.bin", BIG)
        b = self.put("x2.tmp.zip", BIG2)
        self.put("any/cfg (1).json", SMALL)
        self.put("other/same-size-wrong.bin", b"C" * len(BIG))     # the right size, other bytes
        self.put("other/notes.txt", b"hello")
        code, _ = self.run_import()
        self.assertEqual(code, 0)
        self.assertEqual((self.models / "Model-A/a-00001.safetensors").read_bytes(), BIG)
        self.assertEqual((self.models / "Model-A/a-00002.safetensors").read_bytes(), BIG2)
        self.assertEqual((self.models / "Model-A/config.json").read_bytes(), SMALL)
        self.assertFalse(a.exists() or b.exists())                 # same drive: moved, not copied
        self.assertEqual((self.src / "other/same-size-wrong.bin").read_bytes(), b"C" * len(BIG))
        self.assertTrue((self.src / "other/notes.txt").exists())   # what did not match is untouched
        # recorded as verified: Maya's own check does not read the file again
        with patch.object(maya, "sha256_ok", side_effect=AssertionError("read again")), patch.object(maya, "say"):
            self.assertTrue(maya.sha256_checked(self.models / "Model-A/a-00001.safetensors", sha(BIG)))

    def test_unlisted_size_gguf_matched_by_hash_and_shared_file_follows_its_model(self):
        self.put("dl/1.part1", SHAPE)
        self.put("dl/2.dat", SHARED)
        self.run_import()
        self.assertEqual((self.models / "Model-B/shard-00001.gguf").read_bytes(), SHAPE)
        self.assertTrue((self.models / "Model-B/vision/mm.gguf").exists())    # B's file is there: B gets it
        self.assertFalse((self.models / "Model-A/vision/mm.gguf").exists())

    def test_shared_file_alone_is_not_guessed_into_a_model(self):
        self.put("dl/2.dat", SHARED)
        code, _ = self.run_import()
        self.assertFalse(self.models.exists() and any(self.models.rglob("mm.gguf")))
        self.assertTrue((self.src / "dl/2.dat").exists())

    def test_another_drive_is_copied_only_with_copy_and_the_original_stays(self):
        a = self.put("a.bin", BIG)
        with patch.object(MI, "same_drive", return_value=False):
            code, warn = self.run_import(copy=False)               # --yes, no --copy: nothing is copied
            self.assertFalse((self.models / "Model-A/a-00001.safetensors").exists())
            self.assertTrue(a.exists())
            self.assertTrue(any("--copy" in str(c) for c in warn.call_args_list))
            self.run_import(copy=True)
        self.assertEqual((self.models / "Model-A/a-00001.safetensors").read_bytes(), BIG)
        self.assertTrue(a.exists())                                # the original is never deleted by a copy
        self.assertFalse(list(self.models.rglob("*.maya-import")))

    def test_existing_destination_is_not_replaced(self):
        dest = self.models / "Model-A/a-00001.safetensors"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"partial")
        a = self.put("a.bin", BIG)
        _, warn = self.run_import()
        self.assertEqual(dest.read_bytes(), b"partial")
        self.assertTrue(a.exists())
        self.assertTrue(warn.called)

    def test_a_hugging_face_cache_is_copied_and_left_whole(self):
        blob = self.put("hub/models--x--y/blobs/" + sha(BIG)[:20], BIG)
        snap = self.src / "hub/models--x--y/snapshots/abc/a-00001.safetensors"
        snap.parent.mkdir(parents=True)
        try:
            os.symlink(blob, snap)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks here")
        self.run_import()
        self.assertEqual((self.models / "Model-A/a-00001.safetensors").read_bytes(), BIG)
        self.assertTrue(blob.exists() and snap.is_symlink())       # the cache still works

    def test_report_names_what_is_missing(self):
        self.put("a.bin", BIG)
        got = MI.find(self.src, self.cat)
        placed, _ = MI.choose(got["found"])
        _, missing = MI.report_lines(self.cat, placed, self.models)
        self.assertEqual(missing, [("Model-A", ["a-00002.safetensors", "config.json", "mm.gguf"])])


if __name__ == "__main__":
    unittest.main()
