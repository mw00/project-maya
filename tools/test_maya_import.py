"""--import-model (#666): files downloaded some other way are found by size + sha256, whatever their names, and placed in
the models folder's layout.  Fake small files and a patched catalog; no GPU, no network."""
import argparse
import hashlib
import os
import struct
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

    def test_link_checks_type_only_and_keeps_originals(self):
        first = self.put("model-00001.gguf", b"GGUF short header")
        second = self.put("model-00002.gguf", b"other shard")
        cat = [{"label": "Maya-S-v2-IQ2_XXS", "rel": f"Maya-S-v2-IQ2_XXS/{p.name}",
                "size": 1000000, "sha": ["wrong"]} for p in (first, second)]
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "gguf_head", side_effect=lambda p: ("glm5-next", 1) if p == first else ("", 0)), \
                patch.object(MI, "sha256_of", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "sha256_ok", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "glm"), 0)
            dest = self.models / "Maya-S-v2-IQ2_XXS/model-00001.gguf"
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(), first.resolve())
            self.assertTrue((dest.parent / "model-00002.gguf").is_symlink())
            self.assertTrue(first.exists() and second.exists())
            self.assertFalse((dest.parent / MI.VERIFIED).exists())
            self.assertTrue(MI.is_linked(dest))
            self.assertIsNone(maya.file_incomplete({"files": {dest.name: 1000000}}, dest))
            self.assertTrue(maya.sha256_checked(dest, "wrong"))

    def test_review_detects_two_quants_and_links_only_the_selected_one(self):
        first = self.put("one/first-00001-of-00002.gguf", b"GGUF first")
        second = self.put("one/first-00002-of-00002.gguf", b"GGUF second")
        other = self.put("two/other.gguf", b"GGUF other")
        draft = self.put(maya.DFLASH["file"], b"GGUF draft")
        vision = self.put("vision/mmproj-test.gguf", b"GGUF vision")
        labels = list(maya.MODELS)[:2]
        cat = ([{"label": labels[0], "rel": f"{labels[0]}/{p.name}", "size": 999, "sha": ["wrong"]}
                for p in (first, second)] +
               [{"label": labels[1], "rel": f"{labels[1]}/{other.name}", "size": 999, "sha": ["wrong"]},
                {"label": labels[1], "rel": f"{labels[1]}/vision/{vision.name}", "size": 999,
                 "sha": ["wrong"]}])
        with patch.object(maya, "gguf_head", side_effect=lambda p: ("glm5-next", 1) if p in (first, other)
                          else ("", 0)), patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "choose", side_effect=[0, 1, 3]), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            found = maya.review_candidates(self.src, "glm")
            self.assertEqual([c["path"] for c in found["model"]], [first, other])
            self.assertEqual(found["model"][0]["files"], {first, second})
            self.assertEqual([c["path"] for c in found["drafter"]], [draft])
            self.assertEqual([c["path"] for c in found["vision"]], [vision])
            self.assertEqual(maya.review_downloaded_model(argparse.Namespace(), self.src, self.models, "glm"), 0)
        self.assertTrue((self.models / labels[1] / other.name).is_symlink())
        self.assertFalse((self.models / labels[0] / first.name).exists())
        self.assertTrue((self.models / labels[1] / "vision" / vision.name).is_symlink())
        self.assertTrue((self.models / "GLM-5.3-Flash-DFlash2" / draft.name).is_symlink())

    def test_import_selected_files_ignores_another_quant(self):
        selected = self.put("one/a.bin", BIG)
        other = self.put("two/shard-00001.gguf", SHAPE)
        args = argparse.Namespace(models_dir=str(self.models), copy=True, yes=True)
        with patch.object(maya, "import_catalog", return_value=self.cat), \
                patch.object(maya, "configs", return_value=[]), patch.object(maya, "say"), \
                patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.import_model(args, str(self.src), selected_files={selected}), 0)
        self.assertTrue((self.models / "Model-A/a-00001.safetensors").exists())
        self.assertFalse((self.models / "Model-B/shard-00001.gguf").exists())
        self.assertTrue(other.exists())

    def test_catalog_skips_unpublished_quants_without_failing_the_live_setup(self):
        with patch.object(maya, "fail", side_effect=AssertionError("catalog interrupted the setup")):
            catalog = maya.import_catalog(self.models)
        self.assertTrue(any(e["label"] == maya.QWEN_UD_DIR for e in catalog))

    def test_edited_file_name_keeps_an_invalid_value_for_correction(self):
        chosen = self.put("extra/model.gguf", b"GGUF model")
        with patch.object(maya, "choose", return_value=0), \
                patch.object(maya.S, "ask_text", side_effect=["extra/missing.gguf", "extra/model.gguf"]) as ask, \
                patch.object(maya, "warn"):
            result = maya.review_pick_file(self.src, "model", [], None)
        self.assertEqual(result["path"], chosen)
        self.assertEqual([call.args[1] for call in ask.call_args_list], ["", "extra/missing.gguf"])

    def test_review_can_select_a_file_in_another_directory(self):
        for role in ("model", "drafter", "vision"):
            with self.subTest(role=role):
                folder = self.root / f"outside-{role}"
                folder.mkdir()
                first = folder / "weights-00001-of-00002.gguf"
                second = folder / "weights-00002-of-00002.gguf"
                first.write_bytes(b"GGUF first")
                second.write_bytes(b"GGUF second")
                with patch.object(maya, "choose", return_value=0), \
                        patch.object(maya.S, "ask_text", return_value=str(second)) as ask:
                    result = maya.review_pick_file(self.src, role, [], None)
                self.assertEqual(result["path"], first)
                self.assertEqual(result["files"], {first, second})
                self.assertEqual(maya.review_path(first, self.src), str(first))
                self.assertEqual(ask.call_args.kwargs["file_root"], self.src)

    def test_link_selected_companions_from_other_directories(self):
        main = self.put("main.gguf", b"GGUF model")
        other = self.root / "companion-files"
        other.mkdir()
        draft = other / maya.DFLASH["file"]
        draft.write_bytes(b"GGUF drafter")
        vision = other / "mmproj-test.gguf"
        vision.write_bytes(b"GGUF vision")
        label = next(iter(maya.MODELS))
        cat = [{"label": label, "rel": f"{label}/main.gguf", "size": 999, "sha": ["wrong"]},
               {"label": label, "rel": f"{label}/vision/{vision.name}", "size": 999, "sha": ["wrong"]}]
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "gguf_head", side_effect=lambda p: ("glm5-next", 1) if p == main else ("", 0)), \
                patch.object(MI, "sha256_of", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "glm", selected_model=main,
                                             selected_files={main, draft, vision}), 0)
        for source, dest in ((main, self.models / label / main.name),
                             (vision, self.models / label / "vision" / vision.name),
                             (draft, self.models / "GLM-5.3-Flash-DFlash2" / draft.name)):
            self.assertTrue(source.is_file())
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(), source.resolve())

    def test_import_selected_file_outside_the_source_directory(self):
        external = self.root / "external.bin"
        external.write_bytes(BIG)
        args = argparse.Namespace(models_dir=str(self.models), copy=True, yes=True)
        with patch.object(maya, "import_catalog", return_value=self.cat), \
                patch.object(maya, "configs", return_value=[]), patch.object(MI, "same_drive", return_value=False), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.import_model(args, str(self.src), selected_files={external}), 0)
        self.assertEqual((self.models / "Model-A/a-00001.safetensors").read_bytes(), BIG)
        self.assertEqual(external.read_bytes(), BIG)

    def test_link_rejects_wrong_model_type_and_does_not_replace_existing_file(self):
        first = self.put("model.gguf", b"GGUF header")
        cat = [{"label": "Maya-S-v2-IQ2_XXS", "rel": "Maya-S-v2-IQ2_XXS/model.gguf",
                "size": 1000000, "sha": ["wrong"]}]
        dest = self.models / cat[0]["rel"]
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"existing")
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "gguf_head", return_value=("glm-dsa", 1)), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "glm"), 1)
            self.assertEqual(dest.read_bytes(), b"existing")
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "gguf_head", return_value=("glm5-next", 1)), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "glm"), 1)
            self.assertEqual(dest.read_bytes(), b"existing")
            self.assertTrue(first.exists())

    def test_gguf_type_needs_no_config_json_and_exl3_type_uses_it(self):
        def blob(value):
            raw = value.encode("utf-8")
            return struct.pack("<Q", len(raw)) + raw
        gguf = self.put("glm.gguf", b"GGUF" + struct.pack("<IQQ", 3, 1, 1) +
                        blob("general.architecture") + struct.pack("<I", 8) + blob("glm5-next"))
        self.assertEqual(maya.linked_model_type(gguf), "glm")
        self.assertFalse((self.src / "config.json").exists())
        conf = self.put("config.json", b'{"model_type":"minimax_m3_vl","quantization_config":{"quant_method":"exl3"}}')
        self.assertEqual(maya.linked_model_type(conf), "minimax")

    def test_exl3_links_by_config_type_without_published_size_or_sha(self):
        conf = self.put("config.json", b'{"model_type":"glm5_next","quantization_config":{"quant_method":"exl3"}}')
        shard = self.put("model-00001-of-00015.safetensors", b"short but linked")
        label = "EXL3-3.05bpw"
        cat = [{"label": label, "rel": f"{label}/{p.name}", "size": 1000000, "sha": ["wrong"]}
               for p in (conf, shard)]
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(MI, "sha256_of", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "sha256_ok", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "glm"), 0)
        dest = self.models / label / shard.name
        self.assertTrue(dest.is_symlink())
        self.assertEqual(dest.resolve(), shard.resolve())
        self.assertIsNone(maya.file_incomplete({"files": {shard.name: 1000000}}, dest))

    def test_review_shows_first_exl3_weight_and_links_its_config(self):
        label = "EXL3-3.05bpw"
        conf = self.put(f"{label}/config.json",
                        b'{"model_type":"glm5_next","quantization_config":{"quant_method":"exl3"}}')
        shard = self.put(f"{label}/model-00001-of-00015.safetensors", b"weight")
        cat = [{"label": label, "rel": f"{label}/{p.name}", "size": 999, "sha": ["wrong"]}
               for p in (conf, shard)]
        found = maya.review_candidates(self.src, "glm")
        self.assertEqual([c["path"] for c in found["model"]], [shard])
        self.assertEqual(found["model"][0]["files"], {conf, shard})
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "glm", selected_model=conf,
                                             selected_files={conf, shard}), 0)
        self.assertTrue((self.models / label / conf.name).is_symlink())
        self.assertTrue((self.models / label / shard.name).is_symlink())

    def link_with_companions(self, family, label, arch, extras):
        main = self.put("main.gguf", b"GGUF model header")
        for name, data in extras.items():
            self.put(name, data)
        cat = [{"label": label, "rel": f"{label}/main.gguf", "size": 1000000, "sha": ["wrong"]}]
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "gguf_head", side_effect=lambda path: (arch, 1) if path == main else ("", 0)), \
                patch.object(MI, "sha256_of", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "sha256_ok", side_effect=AssertionError("hashed during link")), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, family), 0)

    def test_glm_links_dflash_and_setup_uses_it_without_hashing(self):
        name = maya.DFLASH["file"]
        self.link_with_companions("glm", "Maya-S-v2-IQ2_XXS", "glm5-next", {name: b"GGUF draft"})
        dest = self.models / "GLM-5.3-Flash-DFlash2" / name
        self.assertTrue(dest.is_symlink())
        with patch.object(maya, "sha256_ok", side_effect=AssertionError("hashed DFlash")), \
                patch.object(maya, "fetch", side_effect=AssertionError("downloaded DFlash")), patch.object(maya, "ok"):
            self.assertEqual(maya.drafter_step(self.models), dest)

    def test_qwen_links_separate_vision_file(self):
        name = next(iter(maya.qwen_vision_model()["files"]))
        self.link_with_companions("qwen", maya.QWEN, "qwen4exp", {name: b"GGUF vision"})
        self.assertTrue((self.models / maya.QWEN / "vision" / name).is_symlink())

    def test_deepseek_links_vision_mmproj(self):
        name = next(iter(maya.ds4_vision_model(True)["files"]))
        label = f"{maya.DS4}-GGUF/UD-Q8_K_XL"
        self.link_with_companions("deepseek", label, "deepseek41", {name: b"GGUF vision"})
        self.assertTrue((self.models / f"{maya.DS4}-GGUF" / "vision" / name).is_symlink())

    def test_deepseek_links_vision_config_from_the_vision_folder(self):
        shard = "model-00001-of-00048.safetensors"
        vision_config = self.put("vision/config.json", b'{"vision": true}')
        self.link_with_companions("deepseek", f"{maya.DS4}-GGUF/UD-Q8_K_XL", "deepseek41",
                                  {f"vision/{shard}": b"vision", "config.json": b"{}"})
        dest = self.models / f"{maya.DS4}-GGUF" / "vision" / "config.json"
        self.assertTrue(dest.is_symlink())
        self.assertEqual(dest.resolve(), vision_config.resolve())

    def test_qwen_links_separate_mtp_head_with_the_model(self):
        mtp = self.put("extra/mtp-Qwen3.8-Flash-Next-Q8_0.gguf", b"GGUF MTP")
        main = self.put("main.gguf", b"GGUF model")
        label = maya.QWEN_UD_DIR
        cat = [{"label": label, "rel": f"{label}/{name}", "size": 1000000, "sha": ["wrong"]}
               for name in (main.name, mtp.name)]
        with patch.object(maya, "import_catalog", return_value=cat), \
                patch.object(maya, "gguf_head", side_effect=lambda path: ("qwen4exp", 1) if path == main else ("", 0)), \
                patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "warn"):
            self.assertEqual(maya.link_model(str(self.src), self.models, "qwen"), 0)
        dest = self.models / label / mtp.name
        self.assertTrue(dest.is_symlink())
        self.assertEqual(dest.resolve(), mtp.resolve())

    def test_minimax_gguf_links_vision_and_dspark(self):
        label = maya.MM3_QUANTS["gguf-xs"]
        self.link_with_companions("minimax", label, "minimax-m3", {
            "vision/model-vision-00001-of-00001.safetensors": b"vision",
            "vision/config.json": b"{}", "vision/preprocessor_config.json": b"{}",
            "dspark/model-dspark-00001.safetensors": b"draft", "dspark/config.json": b"{}",
        })
        base = self.models / maya.MM3_COMPANIONS
        self.assertTrue((base / "model-vision-00001-of-00001.safetensors").is_symlink())
        self.assertTrue((base / "config.json").is_symlink())
        self.assertTrue((base / "dspark/model-dspark-00001.safetensors").is_symlink())
        self.assertTrue((base / "dspark/config.json").is_symlink())

    def test_report_names_what_is_missing(self):
        self.put("a.bin", BIG)
        got = MI.find(self.src, self.cat)
        placed, _ = MI.choose(got["found"])
        _, missing = MI.report_lines(self.cat, placed, self.models)
        self.assertEqual(missing, [("Model-A", ["a-00002.safetensors", "config.json", "mm.gguf"])])


if __name__ == "__main__":
    unittest.main()
