"""Local DeepSeek/MiniMax GGUF selection through config writing, without downloads or GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya
import maya_import as MI
from test_setup_glm_local import gguf


class LocalOther(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.models = self.root / "models"
        for name in ("say", "ok", "warn", "step"):
            p = patch.object(maya, name)
            p.start(); self.addCleanup(p.stop)
        for obj, name in ((MI, "sha256_of"), (maya, "sha256_ok"), (maya, "incomplete"),
                          (maya, "offer_download"), (maya, "import_catalog")):
            p = patch.object(obj, name, side_effect=AssertionError(f"unexpected {name}"))
            p.start(); self.addCleanup(p.stop)

    def source(self, family, arch=None):
        arch = arch or {"deepseek": "deepseek41", "minimax": "minimax-m3"}[family]
        src = self.root / family
        paths = [gguf(src / f"custom-{family}-Q3_K-{i:05d}-of-00003.gguf", arch) for i in range(1, 4)]
        chosen = {"model": maya.review_candidate(paths[0], "model", paths), "drafter": None, "vision": None}
        return src, paths, chosen

    def test_unlisted_quants_link_by_type_and_keep_sources(self):
        for family in ("deepseek", "minimax"):
            with self.subTest(family=family):
                src, paths, _ = self.source(family)
                with patch.object(maya, "choose", return_value=3):
                    linked = maya.review_downloaded_model(SimpleNamespace(), src, self.models, family)
                for source, dest in zip(paths, linked["files"]):
                    self.assertTrue(dest.is_symlink())
                    self.assertEqual(dest.resolve(), source)
                    self.assertTrue(source.is_file())
                self.assertEqual(maya.linked_model_type(linked["model"]), family)

    def test_foreign_architectures_and_missing_shards_are_rejected(self):
        for family, arch in (("deepseek", "deepseek4"), ("minimax", "glm5-next")):
            _, _, chosen = self.source(family, arch)
            self.assertIsNone(maya.link_other_gguf(self.models, chosen, family))
        for family in ("deepseek", "minimax"):
            _, paths, chosen = self.source(family)
            paths[-1].unlink()
            paths[-1].symlink_to(self.root / "unmounted" / paths[-1].name)
            self.assertIsNone(maya.link_other_gguf(self.models, chosen, family))
        self.assertFalse(self.models.exists())

    def test_unsupported_companions_are_reported_before_links_are_created(self):
        for family in ("deepseek", "minimax"):
            _, _, chosen = self.source(family)
            for role, arch in (("drafter", "dflash"), ("vision", "clip")):
                companion = gguf(self.root / f"{role}.gguf", arch)
                selected = {**chosen, role: maya.review_candidate(companion, role, [companion])}
                self.assertIsNone(maya.link_other_gguf(self.models, selected, family))
        self.assertFalse(self.models.exists())

    def test_minimax_links_the_selected_gguf_drafter_and_vision(self):
        from test_mm3_pack_gguf import gguf_companions
        src, paths, chosen = self.source("minimax")
        drafter, vision = gguf_companions(src)
        for role, path in (("drafter", drafter), ("vision", vision)):
            chosen[role] = maya.review_candidate(path, role, [path])
        linked = maya.link_other_gguf(self.models, chosen, "minimax")
        self.assertEqual(len(linked["files"]), len(paths))
        for role, src in (("drafter", drafter), ("vision", vision)):
            self.assertTrue(linked[role].is_symlink())
            self.assertEqual(linked[role].resolve(), src)

    def test_minimax_pack_rebuilds_when_selected_companions_change(self):
        from test_mm3_pack_gguf import model, gguf_companions
        src = model(self.root, scales=True)
        drafter, vision = gguf_companions(src)
        first = next(src.glob("*00001-of-*.gguf"))
        chosen = {role: maya.review_candidate(path, role, [path]) for role, path in
                  (("model", first), ("drafter", drafter), ("vision", vision))}
        linked = maya.link_other_gguf(self.models, chosen, "minimax")
        a = SimpleNamespace(no_vision=False, repack=False)
        pack = maya.mm3_pack_step(a, linked["folder"], linked["files"], gguf=True, linked_files=linked)
        self.assertTrue(maya.mm3_pack_has_vision(pack))
        self.assertIn("dspark.num_hidden_layers", (pack / "mm3.txt").read_text())
        linked["vision"] = None
        maya.mm3_pack_step(a, linked["folder"], linked["files"], gguf=True, linked_files=linked)
        self.assertFalse(maya.mm3_pack_has_vision(pack))
        self.assertNotIn(vision.name, (pack / "tensors.txt").read_text())
        self.assertIn(drafter.name, (pack / "tensors.txt").read_text())
        self.assertEqual(json.loads((pack / maya.PACK_SOURCES).read_text())["local_companions"],
                         {"drafter": str(linked["drafter"])})

    def test_broken_source_links_have_a_useful_message(self):
        src = self.root / "broken"
        src.mkdir()
        (src / "first.gguf").symlink_to(self.root / "unmounted" / "first.gguf")
        with patch.object(maya, "choose", side_effect=RuntimeError("stop at the file review")):
            with self.assertRaisesRegex(RuntimeError, "stop at the file review"):
                maya.review_downloaded_model(SimpleNamespace(), src, self.models, "minimax")
        self.assertTrue(any("broken symbolic link" in str(call) for call in maya.warn.call_args_list))

    def test_minimax_keeps_its_indexer_side_file(self):
        src, _, chosen = self.source("minimax")
        side = gguf(src / "custom-minimax-Q3_K-maya-indexers.bin", "minimax-m3")
        linked = maya.link_other_gguf(self.models, chosen, "minimax")
        self.assertTrue((linked["folder"] / side.name).is_symlink())
        self.assertEqual((linked["folder"] / side.name).resolve(), side)

    def test_published_files_keep_the_native_catalog_path(self):
        for family, arch in (("deepseek", "deepseek41"), ("minimax", "minimax-m3")):
            name = sorted(n for n in maya.published_other_gguf_names(family) if "-00001-of-" in n)[0]
            src = self.root / family
            gguf(src / name, arch)
            with patch.object(maya, "choose", return_value=3), \
                    patch.object(maya, "link_model", return_value=0) as catalog, \
                    patch.object(maya, "link_other_gguf", side_effect=AssertionError("native packs bypassed")):
                self.assertEqual(maya.review_downloaded_model(SimpleNamespace(), src, self.models, family), 0)
            catalog.assert_called_once()

    def test_local_deepseek_pack_does_not_take_unselected_companions(self):
        from tools.test_ds4_vision_pack import gguf as ds4_gguf
        d = self.root / "weights"
        d.mkdir()
        ds4_gguf(d / "m-00001-of-00001.gguf")
        for kind in ("dspark", "vision"):
            p = self.root / kind
            p.mkdir()
            (p / f"{kind}.json").write_text("not the selected model's JSON")
            (p / f"{kind}.bin").write_bytes(bytes(256))
        result = subprocess.run([sys.executable, str(Path(maya.__file__).parent / "tools/ds4/pack_gguf.py"),
                                 "--model", str(d), "--no-companions"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = (d / "maya-ds4/ds4.txt").read_text()
        self.assertNotIn("vision.", text)
        self.assertNotIn("dspark.num_hidden_layers", text)

    def test_menu_to_configuration_uses_selected_files_and_contexts(self):
        for family, ctx in (("deepseek", 524288), ("minimax", 262144)):
            with self.subTest(family=family):
                _, paths, chosen = self.source(family)
                linked = maya.link_other_gguf(self.models, chosen, family)
                d = linked["folder"]
                pack = d / ("maya-ds4" if family == "deepseek" else "maya-mm3")
                (pack / "tokenizer").mkdir(parents=True)
                (pack / ("ds4.txt" if family == "deepseek" else "mm3.txt")).write_text("hidden_size 32\n")
                legacy = self.root / ("maya-ds4.json" if family == "deepseek" else "maya-mm3.json")
                legacy.write_text('{"preserve": true}')
                a = SimpleNamespace(model=None, model_dir=None, backend="cuda", models_dir=str(self.models),
                                    ds4_quant=None, mm3_quant=None, yes=False, check=False, no_vision=True,
                                    parallel=None, env=[], port=None, host=None, api_key=None, repack=False)
                pc = {"backend": "cuda", "gpus": [{"index": 0, "vram_gb": 48}], "archs": [89]}
                replacements = {
                    "ROOT": self.root, "choose_with_add": lambda *args: linked, "check_pc": lambda *args, **kw: pc,
                    "tools_step": lambda *args: self.root / "llama", "build_step": lambda *args: {},
                    "mem_gb": lambda: (512, 400), "setup_gpus": lambda *_: [],
                    "ds4_pack_step": lambda *args, **kw: pack, "mm3_pack_step": lambda *args, **kw: pack,
                    "saved_calibration": lambda *args, **kw: None, "ds4_seed_profile": lambda *_: False,
                    "ablate_step": lambda *_: None, "write_run_script": lambda *args: self.root / "run-test.sh",
                }
                with patch.multiple(maya, **replacements), patch.object(maya.S, "gpus", return_value=[]), \
                        patch.object(maya, "choose_context", return_value=ctx) as context:
                    path = (maya.set_up_ds4 if family == "deepseek" else maya.set_up_mm3)(a, {})
                cfg = json.loads(path.read_text())
                self.assertEqual(legacy.read_text(), '{"preserve": true}')
                self.assertEqual(cfg["installer"]["family"], family)
                self.assertEqual(cfg["installer"]["gguf"], str(linked["model"]))
                self.assertEqual(cfg["installer"]["model_dir"], str(d))
                self.assertEqual(maya.flag_value(cfg["args"], "--pack"), str(pack))
                self.assertEqual(maya.flag_value(cfg["args"], "--max-context"), str(ctx))
                self.assertIn(262144, context.call_args.args[2])
                if family == "deepseek":
                    self.assertIn(524288, context.call_args.args[2])
                self.assertEqual(cfg["model_name"], f"custom-{family}-q3_k")

    def test_linked_config_model_names_distinguish_quants_and_source_folders(self):
        a = SimpleNamespace(no_vision=True, parallel=None, env=[], port=None, host=None, api_key=None)
        pc = {"backend": "cuda", "gpus": [{"index": 0}]}
        for family, title, model in (("deepseek", "DeepSeek", "DeepSeek-V4.1-Flash-1017"),
                                      ("minimax", "MiniMax", "MiniMax-M3.1-Flash")):
            names = []
            for quant in ("MXFP4", "Q3_K-AntirezLike", "Q3_K-AntirezLike-second-source"):
                d = self.models / f"{title}-local-{model}-{quant}"
                pack = d / ("maya-ds4" if family == "deepseek" else "maya-mm3")
                pack.mkdir(parents=True)
                (pack / "mm3.txt").write_text("hidden_size 32\n")
                linked = {"model": d / f"{model}-{quant}-00001-of-00010.gguf"}
                with patch.object(maya, "ROOT", self.root), \
                        patch.object(maya, "saved_calibration", return_value=None), \
                        patch.object(maya, "ds4_seed_profile", return_value=False), \
                        patch.object(maya, "write_run_script", return_value=self.root / "run-test.sh"):
                    if family == "deepseek":
                        p = maya.write_config_ds4(a, pc, {}, pack, 262144, self.models, d, "local-gguf",
                                                  linked_files=linked)
                    else:
                        p = maya.write_config_mm3(a, pc, {}, pack, 262144, self.models, d, "local-gguf", own=True,
                                                  linked_files=linked)
                cfg = json.loads(p.read_text())
                names.append(cfg["model_name"])
                self.assertEqual(cfg["model_name"], f"{model}-{quant}".lower())
                self.assertEqual(maya.config_family(cfg), family)
            self.assertEqual(len(set(names)), 3)


if __name__ == "__main__":
    unittest.main()
