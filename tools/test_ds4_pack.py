"""tools/ds4/pack.py's chat template and exl3_k_map.json check: the model repo's own template goes into the pack (this
directory's only for a repo without one), and the K map is checked against every EXL3 matrix's trellis.

    python -m unittest tools.test_ds4_pack
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "ds4"))
import pack  # noqa: E402


def model_dir(root: pathlib.Path, template: str | None) -> pathlib.Path:
    """A model directory with just enough of a tokenizer.json for write_tokenizer (and maybe a template)."""
    m = root / "model"
    m.mkdir()
    (m / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
        "added_tokens": [{"id": 3, "content": "<｜end▁of▁sentence｜>", "special": True}]}))
    if template is not None:
        (m / "chat_template.jinja").write_text(template, encoding="utf-8")
    return m


def fake_st(ks: dict[str, int]) -> types.SimpleNamespace:
    """SafeTensors' t: name -> (file, dtype, shape, offset, bytes); a trellis's last dim is 16 K."""
    return types.SimpleNamespace(t={n: ("f", "I16", [320, 144, 16 * k], 0, 0) for n, k in ks.items()})


K_MAP = {"codebook": "mul1", "mul1_multiplier": pack.MUL1, "routed": {"0": 4, "1": 3}, "shared": {"0": 6, "1": 6},
         "shared_overrides": {}, "engram": {"1": 6}, "mtp_bits": 6, "head_bits": 6, "attn_default": 6,
         "indexer_wk_bits": 8, "num_hidden_layers": 2}
TENSORS = {"layers.0.ffn.experts.7.w1.trellis": 4, "layers.1.ffn.experts.0.w2.trellis": 3,
           "layers.0.ffn.shared_experts.w3.trellis": 6, "layers.1.engram.wkv.trellis": 6,
           "layers.1.attn.wq_a.trellis": 6, "layers.0.attn.indexer.wk.trellis": 8, "mtp.0.attn.wq_b.trellis": 6,
           "head.trellis": 6}


class Template(unittest.TestCase):
    def test_the_repos_own(self):
        with tempfile.TemporaryDirectory() as d:
            m = model_dir(pathlib.Path(d), "{{ 'the repo template' }}")
            out = pathlib.Path(d) / "tok"
            self.assertEqual(pack.write_tokenizer(out, m), m / "chat_template.jinja")
            self.assertEqual((out / "chat_template.jinja").read_text(encoding="utf-8"), "{{ 'the repo template' }}")

    def test_ours_without_one(self):
        with tempfile.TemporaryDirectory() as d:
            m = model_dir(pathlib.Path(d), None)
            out = pathlib.Path(d) / "tok"
            self.assertEqual(pack.write_tokenizer(out, m), pack.TEMPLATE)
            self.assertEqual((out / "chat_template.jinja").read_text(encoding="utf-8"),
                             pack.TEMPLATE.read_text(encoding="utf-8"))


class KMap(unittest.TestCase):
    def check(self, km: dict | None, ks: dict[str, int]):
        with tempfile.TemporaryDirectory() as d:
            m = pathlib.Path(d)
            if km is not None:
                (m / "exl3_k_map.json").write_text(json.dumps(km))
            return pack.check_k_map(m, fake_st(ks))

    def test_agrees(self):
        self.assertEqual(self.check(K_MAP, {**TENSORS, "layers.0.attn_norm.weight": 0}), len(TENSORS))

    def test_no_map(self):
        self.assertIsNone(self.check(None, TENSORS))

    def test_libertaidai_mixed_precision(self):
        km = {**K_MAP, "attn_default": 5, "mtp_bits": 4,
              "shared": {"29": 4},
              "shared_overrides": {"29": {"note": "layer 29 shared is mixed 4/5; infer from trellis"}}}
        ks = {"layers.0.attn.wq_a.trellis": 6, "layers.0.attn.wkv.trellis": 6,
              "layers.0.attn.wq_b.trellis": 5, "layers.1.attn.wq_a.trellis": 5,
              "layers.29.ffn.shared_experts.w1.trellis": 4,
              "layers.29.ffn.shared_experts.w2.trellis": 5,
              "layers.29.ffn.shared_experts.w3.trellis": 4}
        for layer in (0, 1):
            for matrix in ("ffn.experts.0.w1", "ffn.experts.0.w2", "ffn.experts.0.w3"):
                ks[f"mtp.{layer}.{matrix}.trellis"] = 4
            for matrix in ("ffn.shared_experts.w1", "ffn.shared_experts.w2", "attn.wq_a",
                           "attn.wo_a.slice.0", "main_proj"):
                ks[f"mtp.{layer}.{matrix}.trellis"] = 6
        self.assertEqual(self.check(km, ks), len(ks) - 3)
        for name in ("mtp.0.ffn.experts.0.w1.trellis", "mtp.1.attn.wq_a.trellis",
                     "layers.0.attn.wkv.trellis", "layers.1.attn.wq_a.trellis"):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                self.check(km, {**ks, name: ks[name] - 1})

    def test_numeric_shared_override(self):
        km = {**K_MAP, "shared_overrides": {"0": 5}}
        ks = {"layers.0.ffn.shared_experts.w3.trellis": 5}
        self.assertEqual(self.check(km, ks), 1)
        with self.assertRaises(SystemExit):
            self.check(km, {next(iter(ks)): 6})

    def test_a_wrong_k_stops(self):
        with self.assertRaises(SystemExit) as e:
            self.check(K_MAP, {**TENSORS, "layers.1.ffn.experts.0.w2.trellis": 4})
        self.assertIn("layers.1.ffn.experts.0.w2.trellis: K 4, the map 3", str(e.exception))

    def test_another_codebook_stops(self):
        with self.assertRaises(SystemExit):
            self.check({**K_MAP, "codebook": "3inst"}, TENSORS)
        with self.assertRaises(SystemExit):
            self.check({**K_MAP, "mul1_multiplier": 1}, TENSORS)


if __name__ == "__main__":
    unittest.main()
