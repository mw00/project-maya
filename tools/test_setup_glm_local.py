"""Local GLM GGUF linking and setup: no published filenames, checksums, downloads or GPU."""
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya
import maya_import as MI


def gguf(path, arch, file_type=None):
    def text(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    fields = [text("general.architecture") + struct.pack("<I", 8) + text(arch)]
    if file_type is not None:
        fields.append(text("general.file_type") + struct.pack("<II", 4, file_type))
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields)) + b"".join(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + b"\0" * (-len(data) % 32))
    return path


class LocalGlm(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "downloads"
        self.models = self.root / "models"
        self.shards = [gguf(self.source / f"GLM-5.3-Flash-UD-Q5_K_XL-{i:05d}-of-00006.gguf", "glm5-next")
                       for i in range(1, 7)]
        self.draft = gguf(self.source / "GLM-5.3-Flash-DFlash2-Q5_K_M.gguf", "dflash")
        self.vision = gguf(self.source / "mmproj-F16.gguf", "clip")
        for target in ("say", "ok", "warn", "step"):
            p = patch.object(maya, target)
            p.start()
            self.addCleanup(p.stop)
        for target, name in ((MI, "sha256_of"), (maya, "sha256_ok"), (maya, "incomplete")):
            p = patch.object(target, name, side_effect=AssertionError(f"unexpected verification: {name}"))
            p.start()
            self.addCleanup(p.stop)

    def chosen(self):
        return {role: options[0] if options else None
                for role, options in maya.review_candidates(self.source, "glm").items()}

    def link(self):
        return maya.link_glm_local(self.models, self.chosen())

    def test_link_keeps_the_selected_quant_and_companions(self):
        linked = self.link()
        self.assertIsInstance(linked, dict)
        self.assertEqual(len(linked["files"]), 6)
        for source, dest in [*zip(self.shards, linked["files"]),
                             (self.draft, linked["drafter"]), (self.vision, linked["vision"])]:
            self.assertTrue(source.is_file())
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(), source)
            self.assertTrue(MI.is_linked(dest))
        self.assertEqual(maya.link_glm_local(self.models, self.chosen()), linked)
        self.assertEqual(maya.model_step(SimpleNamespace(), self.models, ("local", linked["model"])),
                         (linked["folder"], linked["files"], "UD-Q5_K_XL"))

    def test_link_requires_every_model_shard(self):
        self.shards[-1].unlink()
        self.assertIsNone(self.link())
        self.assertFalse(self.models.exists())

    def test_optional_files_can_be_left_out(self):
        chosen = self.chosen()
        chosen.update(drafter=None, vision=None)
        linked = maya.link_glm_local(self.models, chosen)
        self.assertIsNone(linked["drafter"])
        self.assertIsNone(linked["vision"])

    def test_wrong_model_or_companion_types_are_rejected(self):
        for role, arch in (("model", "glm-dsa"), ("drafter", "qwen4exp"), ("vision", "dflash")):
            with self.subTest(role=role):
                chosen = self.chosen()
                wrong = gguf(self.root / f"wrong-{role}.gguf", arch)
                chosen[role] = maya.review_candidate(wrong, role, [wrong])
                self.assertIsNone(maya.link_glm_local(self.models, chosen))
                self.assertFalse(self.models.exists())

    def test_review_routes_unlisted_glm_to_local_linking(self):
        with patch.object(maya, "choose", return_value=3), \
                patch.object(maya, "import_catalog", side_effect=AssertionError("catalog consulted")), \
                patch.object(maya, "link_model", side_effect=AssertionError("catalog link used")):
            linked = maya.review_downloaded_model(SimpleNamespace(), self.source, self.models, "glm")
        self.assertEqual(linked["model"].name, self.shards[0].name)
        self.assertEqual(linked["drafter"].name, self.draft.name)
        self.assertEqual(linked["vision"].name, self.vision.name)

    def test_add_accepts_quant_types_and_architecture_aliases_without_a_catalog(self):
        # These are GGUF header/link tests. They do not claim GPU inference support for every encoding.
        quants = {"Q2_0": 41, "Q2_K": 10, "UD-Q2_K_XL": 10, "Q3_K_S": 11, "UD-Q3_K_XL": 12,
                  "Q4_0": 2, "Q4_1": 3, "Q4_K_M": 15, "UD-Q4_K_XL": 15, "Q5_0": 8, "Q5_1": 9,
                  "Q5_K_M": 17, "UD-Q5_K_XL": 17, "Q6_K": 18, "UD-Q6_K_XL": 18, "Q8_0": 7,
                  "IQ1_S": 24, "IQ1_M": 31, "IQ2_XXS": 19, "IQ2_XS": 20, "IQ2_S": 28, "IQ2_M": 29,
                  "IQ3_XXS": 23, "IQ3_S": 26, "IQ4_NL": 25, "IQ4_XS": 30, "BF16": 32, "F16": 1,
                  "F32": 0, "MXFP4": 38, "NVFP4": 39, "unknown-future-quant": 999}
        for arch in ("glm5-next", "glm5next"):
            for name, file_type in quants.items():
                with self.subTest(arch=arch, quant=name):
                    source = self.root / arch / name
                    main = gguf(source / f"GLM-5.3-Flash-{name}.gguf", arch, file_type)
                    with patch.object(maya, "choose", return_value=3), \
                            patch.object(maya, "import_catalog", side_effect=AssertionError("catalog consulted")):
                        linked = maya.review_downloaded_model(SimpleNamespace(), source, self.models, "glm")
                    self.assertEqual(linked["model"].resolve(), main)
                    self.assertTrue(MI.is_linked(linked["model"]))

    def test_published_model_names_use_the_same_local_flow_and_selected_drafter(self):
        source = self.root / "published-name"
        spec = next(m for m in maya.MODELS.values() if "files" not in m)
        model = [gguf(source / name, "glm5-next") for name in maya.shard_names(spec)]
        draft = gguf(source / "my-custom-dflash.gguf", "dflash")
        with patch.object(maya, "choose", return_value=3), \
                patch.object(maya, "import_catalog", side_effect=AssertionError("catalog consulted")):
            linked = maya.review_downloaded_model(SimpleNamespace(), source, self.models, "glm")
        self.assertEqual(linked["model"].resolve(), model[0])
        self.assertEqual(linked["drafter"].resolve(), draft)

    def test_same_filename_in_different_sources_gets_separate_folders(self):
        selected = []
        for parent, arch, file_type in (("one", "glm5-next", 15), ("two", "glm5-next", 7),
                                        ("three", "glm5next", 18)):
            main = gguf(self.root / parent / "shared" / "model.gguf", arch, file_type)
            chosen = {"model": maya.review_candidate(main, "model", [main]), "drafter": None, "vision": None}
            linked = maya.link_glm_local(self.models, chosen)
            selected.append((main, chosen, linked))
        self.assertEqual(len({linked["folder"] for _, _, linked in selected}), 3)
        for main, chosen, linked in selected:
            self.assertEqual(linked["model"].resolve(), main)
            self.assertEqual(maya.link_glm_local(self.models, chosen), linked)

    def test_setup_writes_the_selected_files_into_the_config(self):
        linked = self.link()
        pack = linked["folder"] / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "index.txt").write_text("output.weight 0 0 0 0 0 0 4096 154880 8 0 32 0 0 0 0 0 0 0\n")
        a = SimpleNamespace(backend="cuda", models_dir=str(self.models), gguf_dir=None, model=None, model_dir=None,
                            parallel=None, yes=True, no_start=True, context=32768, calibrate=False, no_vision=False,
                            env=[], port=None, host=None, api_key=None, rebuild=False, repack=False, draft=None)
        pc = {"backend": "cuda", "gpus": [{"index": 0, "vram_gb": 48}], "archs": [89]}
        other_config = self.root / "maya-ud-q5_k_xl.json"
        other_config.write_text('{"preserve": true}')
        replacements = {
            "ROOT": self.root, "choose_family": lambda *_: "glm", "configs": lambda: [],
            "choose_with_add": lambda *args: linked, "check_pc": lambda *_: pc,
            "choose_context": lambda *args, **kwargs: 32768, "tools_step": lambda *_: self.root / "llama",
            "build_step": lambda *_: {}, "pack_step": lambda *args: pack,
            "vision_encoder_ready": lambda *_: True, "rotational": lambda *_: False,
            "saved_calibration": lambda *args, **kwargs: None, "profile_fits": lambda *_: False,
            "mem_gb": lambda: (256, 200), "setup_gpus": lambda *_: [],
            "ablate_step": lambda *args: None, "drafting_choice": lambda *args: "on",
            "keep_drafting": lambda *args: None, "write_run_script": lambda *args: self.root / "run-test.sh",
        }
        with patch.multiple(maya, **replacements), patch.object(maya.S, "gpus", return_value=[]), \
                patch.object(maya, "choose_draft", side_effect=AssertionError("selected drafter ignored")), \
                patch.object(maya, "drafter_step", side_effect=AssertionError("downloaded another drafter")), \
                patch.object(maya, "vision_step", side_effect=AssertionError("downloaded another mmproj")):
            path = maya.set_up(a, {})
        cfg = json.loads(path.read_text())
        self.assertEqual(path.stem, f"maya-{linked['folder'].name.lower()}")
        self.assertEqual(other_config.read_text(), '{"preserve": true}')
        self.assertEqual(cfg["args"][cfg["args"].index("--dflash") + 1], str(linked["drafter"]))
        self.assertEqual(cfg["vision"]["mmproj"], str(linked["vision"]))
        self.assertEqual(cfg["vision"]["model"], str(linked["model"]))
        self.assertEqual(cfg["installer"]["gguf"], str(linked["model"]))
        self.assertEqual(cfg["installer"]["local_drafter"], str(linked["drafter"]))
        self.assertEqual(cfg["installer"]["local_vision"], str(linked["vision"]))
        i = cfg["args"].index("--dflash")
        del cfg["args"][i:i + 2]
        self.assertEqual(maya.drafter_file(cfg), linked["drafter"])
        # A pre-existing local setup is updated at its old path, retaining the user's server settings.
        old_cfg = json.loads(path.read_text())
        old_cfg["host"] = "127.0.0.2"
        other_config.write_text(json.dumps(old_cfg))
        path.unlink()
        with patch.multiple(maya, **replacements):
            updated = maya.write_config(a, pc, {}, pack, "UD-Q5_K_XL", 32768, self.models, old_cfg["vision"],
                                        ("local", linked["model"]), linked["drafter"], linked_files=linked)
        self.assertEqual(updated, other_config)
        self.assertEqual(json.loads(updated.read_text())["host"], "127.0.0.2")
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
