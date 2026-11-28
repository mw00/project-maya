"""Synthetic, CPU-only tests for using a DSpark GGUF as a Maya drafter.

    python -m unittest tools.test_ds4_drafter_from_gguf -v
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ds4"))
import drafter_from_gguf as DG  # noqa: E402
import pack_gguf as PG  # noqa: E402


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def synthetic_gguf(path: Path, tensors: list, *, offsets=None, metadata=None) -> dict:
    """Write just enough of GGUF v3 to exercise Maya's own GGUF metadata reader."""
    kv = {
        "general.architecture": (8, "dflash"),
        "general.alignment": (4, 32),
        "dflash.block_count": (4, 3),
        "dflash.target_layers": (9, (4, [37, 38, 39])),
        "dflash.hyper_connection.single_pass": (7, True),
        "dflash.block_size": (4, 5),
        "tokenizer.ggml.mask_token_id": (4, 128799),
        "dflash.expert_count": (4, 128),
        "dflash.expert_used_count": (4, 3),
    }
    kv.update(metadata or {})

    def value(kind, v):
        if kind == 8:
            return gguf_string(v)
        if kind == 9:
            element_kind, items = v
            return struct.pack("<IQ", element_kind, len(items)) + b"".join(value(element_kind, item) for item in items)
        if kind == 7:
            return struct.pack("<?", v)
        if kind == 4:
            return struct.pack("<I", v)
        raise AssertionError(kind)

    metadata_bytes = b"".join(gguf_string(k) + struct.pack("<I", kind) + value(kind, v)
                              for k, (kind, v) in kv.items())
    payload, relative, blobs = bytearray(), [], []
    for name, typ, dims in tensors:
        _, block, block_bytes = PG.TYPES[typ]
        nbytes = math.prod(dims) // block * block_bytes
        at = (len(payload) + 31) // 32 * 32
        payload.extend(b"\0" * (at - len(payload)))
        data = bytes(((len(relative) * 29 + i) % 251 for i in range(nbytes)))
        payload.extend(data)
        relative.append(at)
        blobs.append(data)
    infos = b"".join(gguf_string(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) +
                     struct.pack("<IQ", typ, relative[i] if offsets is None else offsets[i])
                     for i, (name, typ, dims) in enumerate(tensors))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + metadata_bytes + infos
    aligned = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (aligned - len(header)) + payload)
    return {"data_start": aligned, "relative": relative, "blobs": blobs}


def tensors() -> list:
    specials = [
        ("enc.output_norm.weight", 0, [2]),
        ("fc.weight", 30, [4, 2]),
        ("output_norm.weight", 0, [2]),
        ("conf_proj.weight", 30, [4, 2]),
        ("markov_w1.weight", 30, [8, 2]),
        ("markov_w2.weight", 30, [2, 8]),
    ]
    blocks = []
    for block in range(3):
        for role in sorted(DG.BLOCK_TENSORS):
            if role.endswith("_exps.weight"):
                typ, dims = 39, [32, 2, 128]
            elif any(part in role for part in ("norm", "bias", "sinks", "base", "scale")):
                typ, dims = 0, [2]
            else:
                typ, dims = 30, [4, 2]
            blocks.append((f"blk.{block}.{role}", typ, dims))
    return specials + blocks


class ExistingDrafterGguf(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "dspark.gguf"

    def test_names_expert_slices_and_source_bytes(self):
        info = synthetic_gguf(self.source, tensors())
        original = self.source.read_bytes()
        output = self.root / "pack" / "dspark.json"
        result = DG.convert(self.source, output)
        self.assertEqual((result["source_tensors"], result["mapped_entries"], result["payload_bytes_written"]),
                         (78, 1221, 0))
        self.assertEqual(result["binary_mode"], "symlink")
        self.assertEqual(self.source.read_bytes(), original)
        link = output.with_suffix(".bin")
        self.assertTrue(link.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(link)))
        self.assertEqual(link.resolve(), self.source)

        manifest = json.loads(output.read_text())
        entries = {entry[0]: entry for entry in manifest["tensors"]}
        base = info["data_start"]
        self.assertEqual(entries["mtp.0.main_norm.weight"],
                         ["mtp.0.main_norm.weight", "F32", base + info["relative"][0], 8, [2]])
        self.assertEqual(entries["mtp.1.attn.wo_a.weight"][4], [2, 4])
        expert = tensors().index(("blk.2.ffn_gate_exps.weight", 39, [32, 2, 128]))
        for number in (0, 127):
            name = f"mtp.2.ffn.experts.{number}.w1.weight"
            _, dtype, offset, length, shape = entries[name]
            self.assertEqual((dtype, length, shape), ("MXFP4", 34, [2, 32]))
            self.assertEqual(offset, base + info["relative"][expert] + number * 34)
            self.assertEqual(original[offset:offset + length], info["blobs"][expert][number * 34:(number + 1) * 34])
        self.assertEqual(manifest["config"], {
            "dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_markov_rank": 8,
            "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3,
            "dspark_target_layer_ids": [37, 38, 39],
        })
        geometry = {key: 0 for key in PG.DSPARK_KEYS}
        fill, source = PG.dspark_fill(geometry, manifest, 129280)
        self.assertEqual((fill, source), (manifest["config"], "the drafter's .json"))

    def test_output_mount_alias_and_refuse_other_binary(self):
        synthetic_gguf(self.source, tensors())
        physical = self.root / "physical"
        physical.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(physical, target_is_directory=True)
        output = alias / "dspark.json"
        DG.convert(self.source, output)
        self.assertEqual((physical / "dspark.bin").resolve(), self.source)
        DG.convert(self.source, output)  # regenerating the same manifest is safe
        (physical / "dspark.bin").unlink()
        (physical / "dspark.bin").write_bytes(b"unrelated")
        with self.assertRaisesRegex(SystemExit, "Refusing to replace"):
            DG.convert(self.source, output)
        self.assertEqual((physical / "dspark.bin").read_bytes(), b"unrelated")

    def test_windows_copy_fallback_when_symlink_is_unavailable(self):
        synthetic_gguf(self.source, tensors())
        output = self.root / "draft" / "dspark.json"
        with patch.object(DG, "WINDOWS", True), patch.object(Path, "symlink_to", side_effect=OSError("no privilege")), \
                patch.object(DG.os, "link", side_effect=OSError("different drives")):
            result = DG.convert(self.source, output)
        binary = output.with_suffix(".bin")
        self.assertFalse(binary.is_symlink())
        self.assertEqual(binary.read_bytes(), self.source.read_bytes())
        self.assertEqual(result["binary_mode"], "copy")
        self.assertEqual(result["payload_bytes_written"], self.source.stat().st_size)
        self.assertTrue(output.is_file())

    def test_failed_windows_copy_leaves_no_output(self):
        synthetic_gguf(self.source, tensors())
        output = self.root / "draft" / "dspark.json"
        with patch.object(DG, "WINDOWS", True), patch.object(Path, "symlink_to", side_effect=OSError("no privilege")), \
                patch.object(DG.os, "link", side_effect=OSError("different drives")), \
                patch.object(DG.shutil, "copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                DG.convert(self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".bin").exists())
        self.assertEqual(list(output.parent.glob("*.part")), [])

    def test_windows_hardlink_before_copy_and_repeat_without_payload_writes(self):
        synthetic_gguf(self.source, tensors())
        output = self.root / "draft" / "dspark.json"
        with patch.object(DG, "WINDOWS", True), patch.object(Path, "symlink_to", side_effect=OSError("no privilege")), \
                patch.object(DG.shutil, "copyfileobj") as copy:
            result = DG.convert(self.source, output)
            again = DG.convert(self.source, output)
        self.assertTrue(os.path.samefile(output.with_suffix(".bin"), self.source))
        self.assertFalse(output.with_suffix(".bin").is_symlink())
        self.assertEqual((result["binary_mode"], again["binary_mode"]), ("hardlink", "hardlink"))
        self.assertEqual((result["payload_bytes_written"], again["payload_bytes_written"]), (0, 0))
        copy.assert_not_called()

    def test_copy_warning_precedes_copy_and_repeat_preserves_binary(self):
        synthetic_gguf(self.source, tensors())
        output = self.root / "draft" / "dspark.json"
        warning = io.StringIO()
        copyfileobj = DG.shutil.copyfileobj

        def copy_after_warning(src, dst, length):
            self.assertIn(f"Copying {self.source.stat().st_size} bytes", warning.getvalue())
            return copyfileobj(src, dst, length)

        with patch.object(DG, "WINDOWS", True), patch.object(Path, "symlink_to", side_effect=OSError("no privilege")), \
                patch.object(DG.os, "link", side_effect=OSError("different drives")), \
                patch.object(sys, "stderr", warning), patch.object(DG.shutil, "copyfileobj", side_effect=copy_after_warning):
            DG.convert(self.source, output)
        binary = output.with_suffix(".bin")
        before = binary.stat()
        with patch.object(DG.shutil, "copyfileobj") as copy:
            result = DG.convert(self.source, output)
        self.assertEqual(result["binary_mode"], "copy")
        self.assertEqual(result["payload_bytes_written"], 0)
        self.assertEqual((binary.stat().st_ino, binary.stat().st_mtime_ns), (before.st_ino, before.st_mtime_ns))
        copy.assert_not_called()

    def test_existing_copy_with_changed_header_is_not_replaced(self):
        synthetic_gguf(self.source, tensors())
        output = self.root / "draft" / "dspark.json"
        output.parent.mkdir()
        binary = output.with_suffix(".bin")
        synthetic_gguf(binary, tensors(), metadata={"tokenizer.ggml.mask_token_id": (4, 128800)})
        before = binary.read_bytes()
        self.assertEqual(len(before), self.source.stat().st_size)
        with self.assertRaisesRegex(SystemExit, "Refusing to replace.*header hash changed"):
            DG.convert(self.source, output)
        self.assertEqual(binary.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_existing_dangling_binary_symlink_is_refused(self):
        synthetic_gguf(self.source, tensors())
        output = self.root / "draft" / "dspark.json"
        output.parent.mkdir()
        binary = output.with_suffix(".bin")
        binary.symlink_to(self.root / "missing.gguf")
        with self.assertRaisesRegex(SystemExit, "Refusing to replace"):
            DG.convert(self.source, output)
        self.assertTrue(binary.is_symlink())

    def test_pack_rejects_changed_source_before_writing_pack(self):
        synthetic_gguf(self.source, tensors())
        draft_json = self.root / "draft" / "dspark.json"
        DG.convert(self.source, draft_json)
        manifest = json.loads(draft_json.read_text())
        model_dir = self.root / "model"
        model_dir.mkdir()
        synthetic_gguf(model_dir / "model.gguf", [("token_embd.weight", 30, [4, 8])], metadata={
            "general.architecture": (8, "deepseek41"),
            "deepseek41.attention.output_group_count": (4, 2),
            "tokenizer.ggml.tokens": (9, (8, [f"token_{i}" for i in range(8)])),
        })
        original = self.source.read_bytes()
        for change, message in [("size", "source size changed"), ("header", "header hash changed"),
                                ("manifest", "missing or invalid recorded")]:
            with self.subTest(change=change):
                self.source.write_bytes(original)
                draft_json.write_text(json.dumps(manifest))
                if change == "size":
                    with self.source.open("ab") as f:
                        f.write(b"x")
                elif change == "header":
                    synthetic_gguf(self.source, tensors(), metadata={"tokenizer.ggml.mask_token_id": (4, 128800)})
                else:
                    broken = dict(manifest)
                    del broken["source_header_sha256"]
                    draft_json.write_text(json.dumps(broken))
                out = self.root / f"rejected-pack-{change}"
                with patch.object(sys, "argv", ["pack_gguf.py", "--model", str(model_dir), "--out", str(out),
                                                "--drafter", str(draft_json)]):
                    with self.assertRaisesRegex(SystemExit, message):
                        PG.main()
                self.assertFalse(out.exists())

    def test_verifier_keeps_legacy_binary_format_and_does_not_read_weight_payload(self):
        synthetic_gguf(self.source, tensors())
        info = DG.manifest(self.source)
        header_bytes = PG.Gguf(self.source).data_start
        original_open = Path.open

        class HeaderReader:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()
            def tell(self):
                return self.handle.tell()
            def read(self, size=-1):
                if size < 0 or self.tell() + size > header_bytes:
                    raise AssertionError("verification read weight payload")
                return self.handle.read(size)

        def header_only_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            return HeaderReader(handle) if path == self.source else handle

        with patch.object(Path, "open", header_only_open):
            PG.verify_existing_gguf(self.source, info)
        PG.verify_existing_gguf(self.root / "non-gguf.bin", {"config": {}, "tensors": []})

    def test_rejects_whitespace_output_path_before_writing(self):
        synthetic_gguf(self.source, tensors())
        spaced = self.root / "draft with space"
        with self.assertRaisesRegex(SystemExit, "cannot contain whitespace"):
            DG.convert(self.source, spaced / "dspark.json")
        self.assertFalse(spaced.exists())

        spaced.mkdir()
        alias = self.root / "draft_alias"
        alias.symlink_to(spaced, target_is_directory=True)
        with self.assertRaisesRegex(SystemExit, "cannot contain whitespace"):
            DG.convert(self.source, alias / "dspark.json")
        self.assertFalse((spaced / "dspark.json").exists())
        self.assertFalse((spaced / "dspark.bin").exists())

    def test_existing_part_symlink_is_not_followed_or_replaced(self):
        synthetic_gguf(self.source, tensors())
        draft = self.root / "draft"
        draft.mkdir()
        victim = self.root / "victim.txt"
        victim.write_text("keep this file")
        old_part = draft / "dspark.json.part"
        old_part.symlink_to(victim)

        DG.convert(self.source, draft / "dspark.json")
        self.assertEqual(victim.read_text(), "keep this file")
        self.assertTrue(old_part.is_symlink())
        self.assertEqual(old_part.resolve(), victim)
        self.assertTrue((draft / "dspark.json").is_file())
        self.assertEqual((draft / "dspark.bin").resolve(), self.source)

    def test_rejects_out_of_bounds_and_overlapping_source_ranges(self):
        specs = tensors()
        synthetic_gguf(self.source, specs, offsets=[0] * (len(specs) - 1) + [10**9])
        with self.assertRaisesRegex(SystemExit, "Invalid source byte range"):
            DG.manifest(self.source)
        synthetic_gguf(self.source, specs, offsets=[0, 0] + list(range(2, len(specs))))
        with self.assertRaisesRegex(SystemExit, "Overlapping GGUF tensor byte ranges"):
            DG.manifest(self.source)

    def test_rejects_unmapped_or_incomplete_drafter(self):
        synthetic_gguf(self.source, tensors()[:-1] + [("blk.3.attn_norm.weight", 0, [2])])
        with self.assertRaisesRegex(SystemExit, "Unmapped drafter tensor"):
            DG.manifest(self.source)
        synthetic_gguf(self.source, tensors()[1:])
        with self.assertRaisesRegex(SystemExit, "Missing special tensors"):
            DG.manifest(self.source)
        synthetic_gguf(self.source, tensors()[:-1])
        with self.assertRaisesRegex(SystemExit, "Missing DSpark block tensors"):
            DG.manifest(self.source)

    def test_pack_gguf_references_link_and_exact_byte_ranges(self):
        synthetic_gguf(self.source, tensors())
        draft_json = self.root / "draft" / "dspark.json"
        DG.convert(self.source, draft_json)
        manifest = json.loads(draft_json.read_text())

        model_dir = self.root / "model"
        model_dir.mkdir()
        synthetic_gguf(model_dir / "model.gguf", [("token_embd.weight", 30, [4, 8])], metadata={
            "general.architecture": (8, "deepseek41"),
            "deepseek41.attention.output_group_count": (4, 2),
            "tokenizer.ggml.tokens": (9, (8, [f"token_{i}" for i in range(8)])),
        })
        out = self.root / "maya-pack"
        with patch.object(sys, "argv", ["pack_gguf.py", "--model", str(model_dir), "--out", str(out),
                                        "--drafter", str(draft_json)]):
            PG.main()
        lines = [line.split() for line in (out / "tensors.txt").read_text().splitlines()]
        files = {int(parts[1]): parts[2] for parts in lines if parts[0] == "file"}
        draft_file = next(index for index, name in files.items() if (model_dir / name).resolve() == self.source)
        entries = {parts[1]: parts[2:] for parts in lines if parts[0] == "t"}
        expert = next(item for item in manifest["tensors"] if item[0] == "mtp.2.ffn.experts.127.w1.weight")
        self.assertEqual(entries[expert[0]], [expert[1], str(draft_file), str(expert[2]), str(expert[3]),
                                              *map(str, expert[4])])
        wo_a = next(item for item in manifest["tensors"] if item[0] == "mtp.1.attn.wo_a.weight")
        self.assertEqual(entries["mtp.1.attn.wo_a.slice.1.weight"],
                         [wo_a[1], str(draft_file), str(wo_a[2] + wo_a[3] // 2), str(wo_a[3] // 2), "1", "4"])

        spaced = self.root / "draft with space"
        draft_json.parent.rename(spaced)
        with patch.object(sys, "argv", ["pack_gguf.py", "--model", str(model_dir), "--out",
                                        str(self.root / "bad-pack"), "--drafter", str(spaced / "dspark.json")]):
            with self.assertRaisesRegex(SystemExit, "relative to the model cannot contain whitespace"):
                PG.main()
        self.assertFalse((self.root / "bad-pack").exists())

    def test_refuses_to_overwrite_source(self):
        source = self.root / "dspark.bin"
        synthetic_gguf(source, tensors())
        before = source.read_bytes()
        with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
            DG.convert(source, self.root / "dspark.json")
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
