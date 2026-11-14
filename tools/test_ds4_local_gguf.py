"""Selected DeepSeek GGUF companions: source offsets, tap conventions, cache changes and config round trip."""
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/ds4")]
import maya
from tools.ds4 import pack_gguf as PG
from tools.test_mm3_pack_gguf import gguf


def files(root, vision_type=30, checkpoint_taps=True):
    root.mkdir(exist_ok=True)
    d = 64
    common = {"embedding_length": d, "attention.head_count": 4, "attention.key_length": 16,
              "attention.q_lora_rank": 32, "attention.output_group_count": 2, "attention.output_lora_rank": 16,
              "expert_feed_forward_length": 64, "hyper_connection.count": 4}
    kv = {"general.architecture": (8, "deepseek41"), "tokenizer.ggml.tokens": (9, (8, ["a", "b", "c", "d"]))}
    native = {**common, "block_count": 3, "attention.compress_ratios": [0, 0, 0], "engram.layer_ids": [1],
              "engram.head_count": 1, "engram.key_length": 32, "engram.max_ngram_size": 2, "engram.pad_id": 1,
              "engram.token_map": [0, 1, 2, 3], "engram.multipliers": [3, 7], "engram.primes": [7], "engram.offsets": [0]}
    kv.update({"deepseek41." + k: (9, (4, v)) if isinstance(v, list) else (4, v) for k, v in native.items()})
    model = root / "custom-MXFP4.gguf"
    gguf(model, kv, [("token_embd.weight", 30, [d, 4]), ("blk.1.engram_embd.weight", 30, [32, 8]),
                     ("blk.1.engram_kv.weight", 30, [32, d]), ("blk.1.engram_gate.weight", 0, [d])])
    kv = {"general.architecture": (8, "dflash"), "dflash.dsv41_semantics": (7, checkpoint_taps),
          "tokenizer.ggml.mask_token_id": (4, 1), "dflash.block_size": (4, 5), "dflash.block_count": (4, 2),
          "dflash.target_layers": (9, (4, [0, 2] if checkpoint_taps else [1, 3])),
          "dflash.expert_count": (4, 2), "dflash.expert_used_count": (4, 1)}
    kv.update({"dflash." + k: (4, v) for k, v in common.items()})
    draft = root / "dspark-BF16.gguf"
    gguf(draft, kv, [("blk.0.attn_q_a.weight", 30, [d, 32]), ("blk.0.attn_output_a.weight", 30, [32, 32]),
                     ("blk.0.ffn_gate_exps.weight", 30, [d, 64, 2]), ("fc.weight", 30, [2 * d, d]),
                     ("enc.output_norm.weight", 0, [d]), ("output_norm.weight", 0, [d]),
                     ("conf_proj.weight", 0, [d + 32, 1]), ("markov_w1.weight", 30, [32, 4]),
                     ("markov_w2.weight", 30, [32, 4])])
    from vision_pack import mmproj_names, MMPROJ_MARKS
    kv = {"general.architecture": (8, "clip"), "clip.projector_type": (8, "deepseek41v"),
          "clip.vision.projection_dim": (4, d), "clip.vision.block_count": (4, 1),
          "clip.vision.embedding_length": (4, 32), "clip.vision.feed_forward_length": (4, 64),
          "clip.vision.patch_size": (4, 2)}
    ts = []
    for name, sources in mmproj_names(1).items():
        for source in sources:
            shape = [32, 32] if name.endswith("weight") and ("attn." in name or "mlp." in name) else [32]
            if name == "vision.patch_embed.proj.weight": shape = [2, 2, 3, 32]
            typ = 30 if name == "vision.patch_embed.proj.weight" else vision_type if len(shape) > 1 else 0
            ts.append((source, typ, shape))
    ts.extend((source, 0, [d]) for source in MMPROJ_MARKS.values())
    vision = root / "mmproj.gguf"
    gguf(vision, kv, ts)
    return model, draft, vision


class Local(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ("say", "ok", "warn", "step"):
            p = patch.object(maya, name); p.start(); self.addCleanup(p.stop)

    def test_missing_attention_sources_come_from_the_actual_matrices(self):
        g = SimpleNamespace(tensors=[(name, 30, [64, 64], 0) for name in
            ("blk.2.attn_compressor_kv.weight", "blk.8.attn_compressor_kv.weight",
             "blk.2.indexer.attn_q_b.weight", "blk.8.indexer.attn_q_b.weight", "blk.12.indexer.attn_q_b.weight")])
        kv = {"general.architecture": "deepseek41"}
        PG.newer_keys([g], kv)
        self.assertEqual(kv["deepseek41.attention.kv_source_layers"], [2, 8])
        self.assertEqual(kv["deepseek41.attention.indexer.source_layers"], [2, 8, 12])
        kv["deepseek41.attention.kv_source_layers"] = [2]
        PG.newer_keys([g], kv)
        self.assertEqual(kv["deepseek41.attention.kv_source_layers"], [2])

    def test_link_pack_and_change_selected_vision(self):
        model, draft, vision = files(self.root / "download")
        chosen = {role: maya.review_candidate(p, role, [p]) for role, p in
                  (("model", model), ("drafter", draft), ("vision", vision))}
        linked = maya.link_other_gguf(self.root / "models", chosen, "deepseek")
        self.assertTrue(linked)
        for role, source in (("model", model), ("drafter", draft), ("vision", vision)):
            self.assertTrue(linked[role].is_symlink())
            self.assertEqual(linked[role].resolve(), source)
        a = SimpleNamespace(no_vision=False, repack=False)
        pack = maya.ds4_pack_step(a, linked["folder"], gguf=True, files=linked["files"],
                                  local_only=True, linked_files=linked)
        kv = dict(line.split(" ", 1) for line in (pack / "ds4.txt").read_text().splitlines())
        self.assertEqual(kv["dspark_target_layer_ids"], "0,2")
        self.assertEqual(kv["compress_ratios"], "0,0,0,0,0")
        self.assertEqual(kv["dspark_block_size"], "5")
        entries = (pack / "tensors.txt").read_text()
        self.assertIn("mtp.0.attn.wo_a.slice.1.weight", entries)
        self.assertIn("mtp.0.ffn.experts.1.w1.weight", entries)
        self.assertIn("vision.blocks.0.mlp.w1.weight.part.1", entries)
        self.assertIn("image_start F32", entries)
        self.assertEqual(struct.unpack("<4s6i", (pack / "engram.bin").read_bytes()[:28]),
                         (b"DSE1", 4, 1, 2, 1, 32, 1))
        opts = SimpleNamespace(no_vision=False, env=[], port=None, host=None, api_key=None)
        with patch.object(maya, "ROOT", self.root), patch.object(maya, "saved_calibration", return_value=None), \
                patch.object(maya, "ds4_seed_profile", return_value=False), \
                patch.object(maya, "write_run_script", return_value=self.root / "run-test.sh"):
            path = maya.write_config_ds4(opts, {"gpus": [{"index": 0}, {"index": 1}]}, {}, pack, 262144,
                                         self.root / "models", linked["folder"], "local-gguf", linked_files=linked)
        cfg = json.loads(path.read_text())
        self.assertEqual(cfg["installer"]["local_drafter"], str(linked["drafter"]))
        self.assertEqual(cfg["installer"]["local_vision"], str(linked["vision"]))
        self.assertNotEqual(maya.flag_value(cfg["args"], "--vision"), "off")
        self.assertEqual(maya.flag_value(cfg["args"], "--dspark"), "none")
        linked["vision"] = None
        maya.ds4_pack_step(a, linked["folder"], gguf=True, files=linked["files"], local_only=True, linked_files=linked)
        self.assertFalse(maya.ds4_pack_has_vision(pack))
        self.assertNotIn(vision.name, (pack / "tensors.txt").read_text())
        self.assertIn(draft.name, (pack / "tensors.txt").read_text())

    def test_standard_tap_indices_and_q8_vision(self):
        model, draft, vision = files(self.root / "download", 8, checkpoint_taps=False)
        import subprocess
        out = self.root / "pack"
        cmd = [sys.executable, str(ROOT / "tools/ds4/pack_gguf.py"), "--model", str(model.parent), "--out", str(out),
               "--shards", model.name, "--drafter", str(draft), "--vision", str(vision), "--no-companions"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("dspark_target_layer_ids 0,2", (out / "ds4.txt").read_text())
        self.assertIn("vision.blocks.0.attn.wqkv.weight Q8_0", (out / "tensors.txt").read_text())


if __name__ == "__main__":
    unittest.main()
