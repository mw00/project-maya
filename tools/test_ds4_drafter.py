"""tools/test_ds4_drafter.py - DeepSeek's DSpark drafter for the GGUF path without a GPU or a download: its conversion
(tools/ds4/drafter_pack.py: every value kept, the experts in ggml's MXFP4 block order, each role's experts back to back)
and the speed calibration's choice (tools/calibrate_ds4.py: stand-in engines).

    python -m unittest tools.test_ds4_drafter -v
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "ds4"), str(ROOT)]
import calibrate_draft as CD  # noqa: E402
import calibrate_ds4 as CDS  # noqa: E402
import drafter_pack as DP  # noqa: E402
import pack_gguf as PG  # noqa: E402

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def safetensors(path: Path, tensors: dict):
    """name -> (dtype, shape, uint8 bytes): a shard as safetensors lays it out."""
    head, off, blobs = {}, 0, []
    for name, (dt, shape, b) in tensors.items():
        head[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class Conversion(unittest.TestCase):
    def test_values_kept_and_experts_laid_out(self):
        rng = np.random.default_rng(1)
        d = Path(tempfile.mkdtemp())
        w8 = rng.integers(0, 126, (64, 64), dtype=np.uint8)            # E4M3 codes (no NaN)
        s8 = np.array([[120, 127], [130, 125]], np.uint8)             # E8M0, a 32 x 32 block each
        tensors = {"mtp.0.attn.wq_a.weight": ("F8_E4M3", [64, 64], w8.tobytes()),
                   "mtp.0.attn.wq_a.scale": ("F8_E8M0", [2, 2], s8.tobytes()),
                   "mtp.0.attn_norm.weight": ("BF16", [4], np.arange(4, dtype=np.uint16).tobytes())}
        codes = {}
        for e in range(3):                                              # experts out of name order (10 before 2)
            for r in ("w1", "w2", "w3"):
                c = rng.integers(0, 16, (2, 64), dtype=np.uint8)        # 2 rows of 64 values
                codes[(e, r)] = c
                packed = (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)
                tensors[f"mtp.0.ffn.experts.{e}.{r}.weight"] = ("I8", [2, 32], packed.tobytes())
                tensors[f"mtp.0.ffn.experts.{e}.{r}.scale"] = ("F8_E8M0", [2, 2], np.full((2, 2), 127 + e, np.uint8).tobytes())
        safetensors(d / DP.SHARDS[0], tensors)
        for name in DP.SHARDS[1:]:
            safetensors(d / name, {})
        with patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(d)]):
            DP.main()
        raw = (d / "dspark.bin").read_bytes()
        ent = {t[0]: t for t in json.loads((d / "dspark.json").read_text())["tensors"]}
        name, dt, off, nb, shape = ent["mtp.0.attn.wq_a.weight"]
        self.assertEqual((dt, shape), ("BF16", [64, 64]))
        got = (np.frombuffer(raw[off:off + nb], np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(64, 64)
        want = DP.E4M3[w8] * np.exp2(np.repeat(np.repeat(s8.astype(np.float32) - 127, 32, 0), 32, 1))
        np.testing.assert_array_equal(got, want)                        # exact
        for r in ("w1", "w3", "w2"):
            first = ent[f"mtp.0.ffn.experts.0.{r}.weight"]
            for e in range(3):
                _, dt, off, nb, shape = ent[f"mtp.0.ffn.experts.{e}.{r}.weight"]
                self.assertEqual((dt, shape, off), ("MXFP4", [2, 64], first[2] + e * first[3]))   # back to back
                blocks = np.frombuffer(raw[off:off + nb], np.uint8).reshape(2, 2, 17)
                self.assertTrue((blocks[..., 0] == 127 + e).all())
                lo, hi = blocks[..., 1:] & 15, blocks[..., 1:] >> 4      # value j, value j + 16
                np.testing.assert_array_equal(np.concatenate([lo, hi], -1).reshape(2, 64), codes[(e, r)])


class Geometry(unittest.TestCase):
    """#500: a GGUF without the DSpark keys (ISTA's GSQ-RCO: only the target layers) packed a drafter with block size 0
    that never drafted - pack_gguf.py fills them from the drafter's .json, else DeepSeek-V4.1-Flash's own for that
    model's trunk, else leaves the drafter out with a warning; drafter_pack.py records the checkpoint's."""
    FLASH = {"hidden_size": 5120, "num_hidden_layers": 40, "n_routed_experts": 384}

    @staticmethod
    def info(n_exp=128, config=None) -> dict:
        t = [["mtp.0.ffn.experts.%d.w1.weight" % e, "MXFP4", 0, 17, [1, 32]] for e in range(n_exp)]
        return dict({"tensors": [["mtp.0.attn_norm.weight", "BF16", 0, 8, [4]]] + t}, **({"config": config} if config else {}))

    def ista(self, **kw) -> dict:
        geo = dict(self.FLASH, **{k: 0 for k in PG.DSPARK_KEYS[:-1]}, dspark_target_layer_ids=[37, 38, 39])
        geo.update(kw)
        return geo

    def test_flash_values_filled(self):
        fill, src = PG.dspark_fill(self.ista(), self.info(), 129280)
        self.assertEqual(fill, {k: v for k, v in PG.DSPARK_V41_FLASH.items() if k != "dspark_target_layer_ids"})
        self.assertIn("DeepSeek-V4.1-Flash", src)
        fill, _ = PG.dspark_fill(self.ista(dspark_target_layer_ids=[]), self.info(), 129280)
        self.assertEqual(fill["dspark_target_layer_ids"], [37, 38, 39])

    def test_the_ggufs_own_kept(self):
        geo = dict(self.FLASH, **PG.DSPARK_V41_FLASH)
        geo["dspark_block_size"] = 4
        self.assertEqual(PG.dspark_fill(geo, self.info(), 129280), ({}, None))

    def test_the_drafters_own_first(self):
        own = {"dspark_block_size": 7, "dspark_noise_token_id": 3, "dspark_markov_rank": 16,
               "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 2}
        fill, src = PG.dspark_fill(self.ista(hidden_size=4096), self.info(config=own), 129280)   # (any trunk)
        self.assertEqual((fill, src), (own, "the drafter's .json"))

    def test_left_out_without_them(self):
        fill, rest = PG.dspark_fill(self.ista(hidden_size=4096), self.info(), 129280)   # (another model's trunk)
        self.assertIsNone(fill)
        self.assertIn("dspark_block_size", rest)
        self.assertIsNone(PG.dspark_fill(self.ista(), self.info(n_exp=64), 129280)[0])   # (another drafter)
        self.assertIsNone(PG.dspark_fill(self.ista(), self.info(), 4096)[0])             # (a vocabulary too small)

    def test_drafter_pack_records_the_checkpoints(self):
        d = Path(tempfile.mkdtemp())
        safetensors(d / DP.SHARDS[0], {"mtp.0.attn_norm.weight": ("BF16", [4], np.arange(4, dtype=np.uint16).tobytes())})
        for name in DP.SHARDS[1:]:
            safetensors(d / name, {})
        with patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(d)]):
            DP.main()
        self.assertNotIn("config", json.loads((d / "dspark.json").read_text()))   # (no config.json beside them)
        (d / "config.json").write_text(json.dumps({"text_config": {"dspark_block_size": 5, "hidden_size": 5120,
                                                                   "dspark_noise_token_id": 128799}}))
        with patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(d)]):
            DP.main()
        self.assertEqual(json.loads((d / "dspark.json").read_text())["config"],
                         {"dspark_block_size": 5, "dspark_noise_token_id": 128799})


class Calibration(unittest.TestCase):
    def run_cal(self, none_tok_s, dspark_tok_s):
        class Fake:
            def __init__(self, tok_s):
                self.tok_s, self.last = tok_s, {}

            def generate(self, ids, max_new, sampling, cancel):
                self.last = {"decode_ms": max_new / self.tok_s * 1000.0, "drafts_accepted": 3, "drafts_offered": 5}
                yield from range(max_new)
        seen = []

        def start(cfg):
            mode = cfg["args"][cfg["args"].index("--dspark") + 1]
            seen.append((mode, cfg["args"][cfg["args"].index("--expert-profile") + 1] if "--expert-profile" in
                         cfg["args"] else None))
            return Fake(dspark_tok_s if mode == "on" else none_tok_s)
        cfg = {"args": ["--pack", "/nowhere", "--max-context", "8192", "--dspark", "none"],
               "sampling": {"temperature": 1.0, "top_p": 1.0}}
        with patch.object(CD, "requests", lambda c: ([1], [[2]] * 8, 2048)):
            return CDS.run(cfg, say=lambda *a: None, start_engine=start), seen

    def test_keeps_the_faster(self):
        res, seen = self.run_cal(3.0, 4.0)
        self.assertEqual((res["keep"], [m for m, _ in seen]), ("dspark", ["none", "on"]))
        self.assertEqual(self.run_cal(3.0, 3.1)[0]["keep"], "none")       # (a tie: no drafting)
        self.assertEqual(self.run_cal(3.0, 2.5)[0]["keep"], "none")

    def test_options(self):
        cfg = {"args": ["--pack", "p", "--dspark", "on", "--max-context", "1"]}
        self.assertEqual(CDS.with_dspark(cfg, "none")["args"], ["--pack", "p", "--max-context", "1", "--dspark", "none"])
        self.assertEqual(cfg["args"][3], "on")                            # (a copy)
        p = Path(tempfile.mkdtemp())
        (p / "tensors.txt").write_text("file 0 a.gguf\nt layers.0.x BF16 0 0 2 1\n")
        self.assertFalse(CDS.has_drafter({"args": ["--pack", str(p)]}))
        (p / "tensors.txt").write_text("file 0 a.gguf\nt mtp.0.attn.wq_a.weight BF16 0 0 2 1\n")
        self.assertTrue(CDS.has_drafter({"args": ["--pack", str(p)]}))


if __name__ == "__main__":
    unittest.main()
