"""Huihui abliterated GLM download/selection/prepare regression tests, network and GPU free."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402
import prepare_huihui_glm as P  # noqa: E402
from serve.model_pull import catalog, setup_command, Puller  # noqa: E402


class CatalogIntegration(unittest.TestCase):
    def test_ids_cli_gguf_shards_and_hf_urls(self):
        for ident, folder, count, _, digests in maya.HUIHUI_QUANTS:
            self.assertEqual(Q.resolve(ident), ("glm", ident))
            self.assertEqual(Q.resolve(ident.upper()), ("glm", ident))
            arg = SimpleNamespace(model=ident, mm3_quant=None)
            self.assertIsNone(maya.model_option(arg))
            self.assertEqual(arg.model, ident)
            model = maya.MODELS[ident]
            self.assertEqual(model["repo"], maya.HUIHUI_REPO)
            # The vision files are *not* hosted in Huihui's GGUF repository.
            vision = model["vision"]
            self.assertEqual(vision["repo"], maya.MODELS["Maya-M"]["repo"])
            self.assertEqual(vision["revision"], maya.MODELS["Maya-M"]["revision"])
            self.assertEqual(
                maya.hf_url(vision.get("repo", model["repo"]),
                            vision.get("revision", model["revision"]),
                            vision["folder"], vision["mmproj"]),
                "https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/resolve/main/vision/"
                "mmproj-GLM-5.3-Flash-F16.gguf")
            names = maya.shard_names(model)
            self.assertEqual(len(names), count)
            self.assertEqual(set(names), set(model["sha256"]))
            for i, (name, sha) in enumerate(zip(names, digests), 1):
                self.assertEqual(name, f"GLM-5.3-Flash-{folder}-{i:05d}-of-{count:05d}.gguf")
                self.assertEqual(model["sha256"][name], sha)
                self.assertEqual(len(sha), 64)
                self.assertEqual(maya.quant_of(Path(name)), ident)
                self.assertEqual(maya.file_url(model, name),
                                 f"https://huggingface.co/{maya.HUIHUI_REPO}/resolve/main/{folder}/{name}")

    def test_both_quant_options_present_in_interactive_setup_and_server_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows_seen = []
            def pick(_title, _desc, rows, _default, _yes):
                rows_seen.extend(rows)
                return next(i for i, (label, *_rest) in enumerate(rows) if "abliterated UD-IQ4_XS" in label)
            arg = SimpleNamespace(gguf_dir=None, model=None, yes=False, backend="cuda", model_dir=None)
            with patch.object(maya, "choose", side_effect=pick), patch.object(maya.S, "gpus", return_value=[]):
                self.assertEqual(maya.choose_model(arg, Path(tmp), {}),
                                 ("download", "glm5.3-flash-ablit-IQ4_XS"))
            labels = [x[0] for x in rows_seen]
            for label in ("abliterated UD-Q3_K_XL", "abliterated UD-IQ4_XS"):
                self.assertTrue(any(label in x for x in labels))
            for ident, *_ in maya.HUIHUI_QUANTS:
                self.assertTrue(any(f"[{ident}]" in x for x in labels), ident)
            items = catalog(Path(tmp), {})
            for ident, *_ in maya.HUIHUI_QUANTS:
                self.assertIn(ident, [i["id"] for i in items])
                cmd = setup_command(ident, {"installer": {"models_dir": tmp}})
                self.assertEqual(cmd[cmd.index("--model") + 1], ident)
                self.assertIn("--download-model", cmd)
                self.assertEqual(cmd[cmd.index("--models-dir") + 1], tmp)

    def test_pull_api_preserves_case_in_setup_model_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            launched = []
            class DummyPull:
                def __init__(self, ident, cmd):
                    self.id = ident
                    self.command = cmd
                def active(self):
                    return False
            import serve.model_pull as pull
            for ident, *_ in maya.HUIHUI_QUANTS:
                with patch.object(pull, "Pull", side_effect=lambda name, cmd: DummyPull(name, cmd)):
                    slot = Puller(Path(tmp), command=lambda name, cfg: ["maya.py", "--model", name])
                    task = slot.start(ident.upper() + ":LATEST", {})
                self.assertEqual(task.id, ident)
                self.assertEqual(task.command[-1], ident)


def s(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def fixture_bytes(i: int) -> bytes:
    """Metadata-only first GGUF and data-bearing remaining shards, format v3."""
    kvs = (s("general.architecture") + struct.pack("<I", 8) + s("glm5-next") +
           s("general.name") + struct.pack("<I", 8) + s("fixture"))
    n = 0 if i == 1 else 1
    tensor = (s(f"blk.{i}.weight") + struct.pack("<I", 1) + struct.pack("<Q", 32) +
              struct.pack("<IQ", 0, 0)) if n else b""
    header = b"GGUF" + struct.pack("<IQQ", 3, n, 2) + kvs + tensor
    if not n:
        return header  # metadata-only shard has no alignment padding
    return header + bytes((-len(header)) % 32) + bytes([i] * 128)


class Preparation(unittest.TestCase):
    def create(self, root: Path, ident: str, renamed: bool = False):
        spec = dict(maya.MODELS[ident])
        names = maya.shard_names(spec)
        path = root / spec["folder"]
        path.mkdir(parents=True)
        prefix = "from-another-download" if renamed else None
        src_names = ([f"{prefix}-{i:05d}-of-{len(names):05d}.gguf" for i in range(1, len(names) + 1)]
                     if renamed else names)
        files = [path / name for name in src_names]
        for i, f in enumerate(files, 1):
            f.write_bytes(fixture_bytes(i))
        spec["sha256"] = {name: hashlib.sha256(f.read_bytes()).hexdigest() for name, f in zip(names, files)}
        return spec, files, prefix

    def test_hardlink_rename_metadata_and_skip_matching_existing_file(self):
        ident = "glm5.3-flash-ablit-Q3_K_XL"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, files, prefix = self.create(root / "downloads", ident, renamed=True)
            dest = root / "models" / ident
            rows = P.validate_files(files, spec)
            self.assertEqual(rows[0]["tensor_count"], 0)
            self.assertTrue(all(r["sha256_verified"] for r in rows))
            for f, name, rec in zip(files, maya.shard_names(spec), rows):
                self.assertEqual(P.place(f, dest / name, "hardlink", rec["sha256"]), "hardlink")
                self.assertEqual(os.stat(f).st_ino, os.stat(dest / name).st_ino)
                self.assertIn("already present", P.place(f, dest / name, "hardlink", rec["sha256"]))
            out = P.extract_metadata([dest / n for n in maya.shard_names(spec)], spec, ident,
                                     rows, dest / P.METADATA_FILE)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["gguf_architecture"], "glm5-next")
            self.assertEqual(data["metadata"]["general.name"], "fixture")
            self.assertEqual(data["total_tensors"], 3)
            self.assertEqual(P.source_files(root / "downloads", spec, prefix), files)

    def test_both_quants_use_existing_glm_model_step_without_download(self):
        for ident, *_ in maya.HUIHUI_QUANTS:
            with self.subTest(model=ident), tempfile.TemporaryDirectory() as tmp:
                models = Path(tmp) / "models"
                spec, files, _ = self.create(Path(tmp) / "incoming", ident)
                dest = models / ident
                dest.mkdir(parents=True)
                installed = []
                for src, filename in zip(files, maya.shard_names(spec)):
                    dst = dest / filename
                    os.link(src, dst)
                    installed.append(dst)
                with patch.dict(maya.MODELS, {ident: spec}), \
                     patch.object(maya, "step"), patch.object(maya, "ok"), \
                     patch.object(maya, "rotational", return_value=False), \
                     patch.object(maya, "offer_download") as download:
                    got = maya.model_step(SimpleNamespace(download_model=False), models, ("download", ident))
                self.assertEqual(got, (dest, installed, ident))
                download.assert_not_called()

    def test_cli_from_other_prefix_and_metadata_only(self):
        ident = "glm5.3-flash-ablit-IQ4_XS"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, files, prefix = self.create(root / "raw", ident, renamed=True)
            with patch.dict(maya.MODELS, {ident: spec}):
                self.assertEqual(P.main(["--model", ident, "--source", str(root / "raw"),
                                         "--source-prefix", prefix, "--models-dir", str(root / "models")]), 0)
                dest = root / "models" / ident
                self.assertEqual(len(list(dest.glob("*.gguf"))), 5)
                meta = dest / P.METADATA_FILE
                self.assertTrue(meta.is_file())
                self.assertEqual(P.main(["--model", "UD-IQ4_XS", "--source", str(dest),
                                         "--metadata-only", "--metadata-out", str(root / "extracted.json")]), 0)
                self.assertTrue((root / "extracted.json").is_file())

    def test_architecture_geometry_contract_and_rejections(self):
        metadata = {"general.architecture": "glm5-next", "glm5-next.block_count": 46,
                    "glm5-next.nextn_predict_layers": 1, "glm5-next.hyper_connection.count": 4,
                    "glm5-next.attention.key_length_mla": 256,
                    "glm5-next.attention.value_length_mla": 256,
                    "glm5-next.attention.kv_lora_rank": 512,
                    "glm5-next.attention.key_length": 512,
                    "glm5-next.attention.indexer.kpool": 4,
                    "glm5-next.attention.indexer.top_k": 2048,
                    "glm5-next.attention.indexer.kpool_select_tail": True,
                    "glm5-next.attention.head_count_kv": [0] * 34 + [8] * 11,
                    "glm5-next.attention.indexer.types": [1] * 45}
        result = P.inspect_architecture(metadata)
        self.assertEqual(result["status"], "passes_known_geometry_constraints")
        self.assertEqual(result["observed"]["trunk_layers"], 45)
        for key, value, issue in (
            ("hyper_connection.count", 8, "mHC"),
            ("attention.value_length_mla", 128, "value_length_mla"),
            ("attention.key_length", 640, "kv_lora_rank"),
            ("attention.indexer.kpool", 3, "divisible"),
            ("attention.indexer.kpool_select_tail", False, "kpool_select_tail"),
        ):
            bad = dict(metadata, **{"glm5-next." + key: value})
            self.assertIn(issue, "; ".join(P.inspect_architecture(bad)["errors"]))
        bad = dict(metadata)
        bad["glm5-next.attention.indexer.types"] = [1] * 34 + [0] * 11
        self.assertIn("shared (non-full)", "; ".join(P.inspect_architecture(bad)["errors"]))
        self.assertEqual(P.inspect_architecture({"general.architecture": "glm5next"})["status"], "inconclusive")

    def test_q3_k_and_iq4_xs_are_known_to_reader_and_cuda(self):
        from gguf_reader import BLOCK_GEOMETRY
        self.assertEqual(BLOCK_GEOMETRY["Q3_K"], (256, 110))
        self.assertEqual(BLOCK_GEOMETRY["IQ4_XS"], (256, 136))
        dequant = (ROOT / "src/kernels/cuda/dequant_bf16.cu").read_text(encoding="utf-8")
        self.assertIn("TYPE == 11", dequant)   # GGML Q3_K
        self.assertIn("TYPE == 23", dequant)   # GGML IQ4_XS
        mmvq = (ROOT / "src/kernels/cuda/native_mmvq.cu").read_text(encoding="utf-8")
        self.assertIn("Q3_K generic MMVQ", mmvq)
        self.assertIn("IQ4_XS generic MMVQ", mmvq)

    def test_bad_sha_missing_file_collision_and_dry_run(self):
        ident = "glm5.3-flash-ablit-Q3_K_XL"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, files, _ = self.create(root, ident)
            files[1].write_bytes(files[1].read_bytes() + b"X")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                P.validate_files(files, spec)
            files[1].unlink()
            with self.assertRaises(FileNotFoundError):
                P.validate_files(files, spec)
            bad = root / "existing.gguf"
            bad.write_bytes(b"bad")
            with self.assertRaises(FileExistsError):
                P.place(files[0], bad, "copy", spec["sha256"][maya.shard_names(spec)[0]])
            with patch.dict(maya.MODELS, {ident: spec}):
                self.assertEqual(P.main(["--model", ident, "--source", str(root),
                                         "--models-dir", str(root / "models"), "--dry-run"]), 1)


if __name__ == "__main__":
    unittest.main()
