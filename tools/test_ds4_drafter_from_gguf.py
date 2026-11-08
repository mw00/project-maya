"""Synthetic, CPU-only tests for using a DSpark GGUF as a Maya drafter.

    python -m unittest tools.test_ds4_drafter_from_gguf -v
"""
from __future__ import annotations

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

    def test_refuses_to_overwrite_source(self):
        source = self.root / "dspark.json.part"
        synthetic_gguf(source, tensors())
        before = source.read_bytes()
        with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
            DG.convert(source, self.root / "dspark.json")
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
