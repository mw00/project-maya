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


class FakeSt:
    """SafeTensors' t and has() alone: name -> (file, dtype, shape, offset, bytes)."""
    def __init__(self, t: dict):
        self.t = t

    def has(self, name: str) -> bool:
        return name in self.t


def need_st(**extra) -> FakeSt:
    """Every tensor need() asks for in a one-layer model with an Engram layer, the head as EXL3."""
    t = {}
    for n in ("attn.wq_a.trellis", "attn.wq_b.trellis", "attn.wkv.trellis", "attn.wo_b.trellis", "attn.attn_sink",
              "attn.q_norm.weight", "attn.kv_norm.weight", "attn_norm.weight", "ffn_norm.weight", "ffn.gate.weight",
              "ffn.gate.bias", "hc_attn_fn", "hc_ffn_fn", "ffn.shared_experts.w1.trellis"):
        t["layers.0." + n] = ("a", "I16", (1,), 0, 2)
    t["layers.0.engram.embed.weight"] = ("e", "F8_E4M3", (1, 1), 0, 1)
    for n in ("embed.weight", "norm.weight", "head.trellis"):
        t[n] = ("a", "BF16", (1,), 0, 2)
    t.update(extra)
    return FakeSt({k: v for k, v in t.items() if v is not None})


class Need(unittest.TestCase):
    T = {"num_hidden_layers": 1, "engram_layer_ids": [0]}

    def test_complete(self):
        self.assertEqual(pack.need(need_st(), self.T), [])

    def test_a_bf16_head(self):   # (#806: a quant that kept the head as BF16 head.weight)
        self.assertEqual(pack.need(need_st(**{"head.trellis": None, "head.weight": ("a", "BF16", (1,), 0, 2)}),
                                   self.T), [])
        self.assertEqual(pack.need(need_st(**{"head.trellis": None}), self.T), ["head.trellis"])

    def test_int4_engram_tables(self):   # (#806: a quant's own U8 + embed.scale tables, not DeepSeek's FP8)
        miss = pack.need(need_st(**{"layers.0.engram.embed.weight": ("e", "U8", (1, 1), 0, 1)}), self.T)
        self.assertEqual(len(miss), 1)
        self.assertIn("INT4 (U8)", miss[0])
        self.assertIn("47 / 48", miss[0])


def experts(order: str, n: int = 3, sizes=(100, 10, 10)) -> FakeSt:
    """n experts of layer 0 in one file: 'expert' = each expert's 9 pieces back to back, 'type' = every trellis first,
    then every suh, then every svh (#806's shards)."""
    pcs = [(r, q, sizes[i]) for r in ("w1", "w3", "w2") for i, q in enumerate(("trellis", "suh", "svh"))]
    order_ = ([(e, p) for e in range(n) for p in pcs] if order == "expert"
              else [(e, p) for q in ("trellis", "suh", "svh") for e in range(n) for p in pcs if p[1] == q])
    t, off = {}, 0
    for e, (r, q, nb) in order_:
        nb = nb * 4096   # (pieces of a real size: the engine's slack is 1 MiB)
        t["layers.0.ffn.experts.%d.%s.%s" % (e, r, q)] = ("s", "I16", (320, 144, 48), off, nb)
        off += nb
    return FakeSt(t)


class TypeOrdered(unittest.TestCase):
    def test_expert_major(self):
        self.assertIsNone(pack.type_ordered(experts("expert", 40)))

    def test_by_type_stops(self):
        msg = pack.type_ordered(experts("type", 40))
        self.assertIsNotNone(msg)
        self.assertIn("layers.0.ffn.experts.0: its pieces are not together", msg)
        self.assertIn("ordered by type", msg)

    def test_mixed_layouts_are_read_by_pieces(self):   # (several layouts in a layer: the engine reads piece by piece)
        st = experts("type", 40)
        k = "layers.0.ffn.experts.1.w1.trellis"
        f, dt, sh, off, nb = st.t[k]
        st.t[k] = (f, dt, (320, 144, 32), off, nb)
        self.assertIsNone(pack.type_ordered(st))


def write_st(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]):
    """A .safetensors file: name -> (dtype, shape, bytes)."""
    h, off = {}, 0
    for name, (dt, shape, b) in tensors.items():
        h[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(h).encode()
    path.write_bytes(len(hb).to_bytes(8, "little") + hb + b"".join(b for _, _, b in tensors.values()))


class MarkovNames(unittest.TestCase):
    """exllamav3 renames the drafter's markov head (turboderp's EXL3 quants): SafeTensors lists it under DeepSeek's names."""

    def test_exllamav3_names(self):
        import st
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:   # (Windows: the maps stay open)
            m = pathlib.Path(td)
            write_st(m / "model-00001-of-00001.safetensors", {
                "mtp.2.markov_head.markov_w1.weight": ("BF16", [4, 2], bytes(range(16))),
                "mtp.2.markov_head.markov_w2.weight": ("F16", [4, 2], bytes(range(16, 32))),
                "mtp.2.confidence_head.proj.weight": ("F16", [1, 4], bytes(8))})
            (m / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                k: "model-00001-of-00001.safetensors" for k in ("mtp.2.markov_head.markov_w1.weight",
                                                                 "mtp.2.markov_head.markov_w2.weight",
                                                                 "mtp.2.confidence_head.proj.weight")}}))
            s = st.SafeTensors(m)
            self.assertEqual(sorted(s.t), ["mtp.2.confidence_head.proj.weight", "mtp.2.markov_head.embed.weight",
                                           "mtp.2.markov_head.head.weight"])
            self.assertEqual(s.t["mtp.2.markov_head.embed.weight"][1:3], ("BF16", (4, 2)))
            self.assertEqual(bytes(s.raw("mtp.2.markov_head.head.weight")), bytes(range(16, 32)))

    def test_deepseeks_own_names_unchanged(self):
        import st
        self.assertEqual(st.deepseek_name("mtp.2.markov_head.embed.weight"), "mtp.2.markov_head.embed.weight")
        self.assertEqual(st.deepseek_name("layers.0.attn.wq_a.trellis"), "layers.0.attn.wq_a.trellis")


class VisionBf16(unittest.TestCase):
    """#886: a quant's own vision tower in F16 (turboderp's EXL3) is replaced by a BF16 copy - DeepSeek's shard 1 in
    the folder, else vision_pack.py's folder - and a warning says what to do when neither is there."""

    QUANT = "model-00001-of-00002.safetensors"

    def model(self, root: pathlib.Path, dt: str = "F16", shard1: bool = False) -> pathlib.Path:
        m = root / "quant"
        m.mkdir()
        own = {"vision.patch_embed.proj.weight": (dt, [4, 12], bytes(96)), "aligner.w1.bias": (dt, [4], bytes(8)),
               "layers.0.attn.wq_a.trellis": ("I16", [2, 2], bytes(8))}
        write_st(m / self.QUANT, own)
        (m / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: self.QUANT for k in own}}))
        if shard1:   # (the checkpoint's patch embedding as a 4-D [hidden][3][p][p], and a name the quant lacks)
            write_st(m / pack.SHARD1, {"vision.patch_embed.proj.weight": ("BF16", [4, 3, 2, 2], bytes(range(96))),
                                       "aligner.w1.bias": ("BF16", [4], bytes(8)),
                                       "vision.norm.weight": ("BF16", [4], bytes(8))})
        return m

    def test_shard1_wins_over_the_quants_f16(self):
        import st
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            m = self.model(pathlib.Path(td), shard1=True)
            s = st.SafeTensors(m)
            self.assertEqual(s.t["vision.patch_embed.proj.weight"][:3], (self.QUANT, "F16", (4, 12)))
            line = pack.vision_bf16(s, m, m.parent / "vision")
            self.assertIn("2 tensors in BF16 from %s" % pack.SHARD1, line)
            self.assertIn("F16", line)
            fn, dt, shape, off, nb = s.t["vision.patch_embed.proj.weight"]
            self.assertEqual((fn, dt, shape, nb), (pack.SHARD1, "BF16", (4, 12), 96))
            self.assertEqual(bytes(s.raw("vision.patch_embed.proj.weight")), bytes(range(96)))
            self.assertEqual(s.t["aligner.w1.bias"][:3], (pack.SHARD1, "BF16", (4,)))
            self.assertEqual(s.t["vision.norm.weight"][:2], (pack.SHARD1, "BF16"))   # (added as before)
            self.assertEqual(s.t["layers.0.attn.wq_a.trellis"][0], self.QUANT)          # (nothing else moves)

    def test_the_vision_folder_when_shard1_is_not_there(self):
        import st
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            m = self.model(pathlib.Path(td))
            v = pathlib.Path(td) / "vision"
            v.mkdir()
            (v / "vision.bin").write_bytes(bytes(512))
            (v / "vision.json").write_text(json.dumps({"tensors": [
                ["aligner.w1.bias", "BF16", 0, 8, [4]], ["vision.patch_embed.proj.weight", "BF16", 256, 96, [4, 12]]]}))
            s = st.SafeTensors(m)
            line = pack.vision_bf16(s, m, v)
            self.assertIn("../vision/vision.bin", line)
            self.assertIn("../vision/vision.bin", s.files)
            self.assertEqual(s.t["vision.patch_embed.proj.weight"], ("../vision/vision.bin", "BF16", (4, 12), 256, 96))
            self.assertEqual(s.t["aligner.w1.bias"], ("../vision/vision.bin", "BF16", (4,), 0, 8))

    def test_a_warning_without_a_bf16_copy(self):
        import st
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            m = self.model(pathlib.Path(td))
            s = st.SafeTensors(m)
            line = pack.vision_bf16(s, m, m.parent / "vision")
            self.assertIn("WARNING", line)
            self.assertIn(pack.SHARD1, line)
            self.assertEqual(s.t["vision.patch_embed.proj.weight"][1], "F16")

    def test_a_bf16_tower_is_left_alone(self):
        import st
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            m = self.model(pathlib.Path(td), dt="BF16", shard1=True)
            s = st.SafeTensors(m)
            self.assertIsNone(pack.vision_bf16(s, m, None))
            self.assertEqual(s.t["vision.patch_embed.proj.weight"][:2], (self.QUANT, "BF16"))

    def test_flat_patch(self):
        self.assertEqual(pack.flat_patch("vision.patch_embed.proj.weight", (1024, 3, 14, 14)), (1024, 588))
        self.assertEqual(pack.flat_patch("vision.patch_embed.proj.weight", (1024, 588)), (1024, 588))
        self.assertEqual(pack.flat_patch("aligner.w1.weight", (2, 3, 4)), (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
