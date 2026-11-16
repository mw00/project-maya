"""tools/iq_fixture.py (B13.1): the expert-type fixtures are valid, deterministic and checked.  No GPU, no model.

    python3 -m unittest tools.test_iq_fixture -v
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import iq_fixture as F  # noqa: E402
from iq_fixture import gguf  # noqa: E402

SMALL = ["--n-embd", "256", "--n-ff", "256", "--n-expert", "8"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class IqFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, name: str, *extra: str) -> Path:
        out = self.d / name
        self.assertEqual(F.main(["--out", str(out), *SMALL, *extra]), 0)
        return out

    def test_default_types_check_and_manifest(self):
        out = self.make("a.gguf", "--check", "--ref-out", str(self.d / "a.ref"))
        man = json.loads((self.d / "a.gguf.json").read_text())
        self.assertEqual([e["type"] for e in man["layers"]], list(F.DEFAULT_TYPES))
        self.assertEqual(man["file_sha256"], sha(out))
        r = gguf.GGUFReader(str(out))
        names = {t.name for t in r.tensors}
        self.assertEqual(len(names), 3 * len(F.DEFAULT_TYPES))
        # the reference rows: per layer REF_ROWS gate rows (n_embd) and REF_ROWS down rows (n_ff)
        ref = np.fromfile(self.d / "a.ref", dtype=np.float32)
        self.assertEqual(ref.size, len(F.DEFAULT_TYPES) * F.REF_ROWS * (256 + 256))
        self.assertTrue(np.all(np.isfinite(ref)))

    def test_shapes_follow_llama_cpp_expert_layout(self):
        out = self.make("s.gguf", "--types", "Q4_K,IQ3_S", "--n-ff", "512")
        r = gguf.GGUFReader(str(out))
        by = {t.name: t for t in r.tensors}
        g = by["blk.0.ffn_gate_exps.weight"]
        d = by["blk.1.ffn_down_exps.weight"]
        self.assertEqual(tuple(int(x) for x in g.shape), (256, 512, 8))
        self.assertEqual(tuple(int(x) for x in d.shape), (512, 256, 8))
        self.assertEqual(g.tensor_type, gguf.GGMLQuantizationType.Q4_K)
        self.assertEqual(d.tensor_type, gguf.GGMLQuantizationType.IQ3_S)

    def test_deterministic(self):
        a = self.make("x.gguf")
        b = self.make("y.gguf")
        self.assertEqual(sha(a), sha(b))
        c = self.make("z.gguf", "--seed", "2")
        self.assertNotEqual(sha(a), sha(c))

    def test_appending_a_type_keeps_earlier_layers(self):
        a = self.make("p.gguf", "--types", "Q2_K,IQ2_XXS")
        b = self.make("q.gguf", "--types", "Q2_K,IQ2_XXS,IQ4_XS")
        ma = json.loads((self.d / "p.gguf.json").read_text())["layers"]
        mb = json.loads((self.d / "q.gguf.json").read_text())["layers"]
        self.assertEqual(ma[0]["sha256"], mb[0]["sha256"])
        self.assertEqual(ma[1]["sha256"], mb[1]["sha256"])
        self.assertNotEqual(sha(a), sha(b))

    def test_quantized_types_via_gguf_py(self):
        out = self.make("g.gguf", "--types", "Q4_0,Q5_1,Q8_0,BF16,F16,F32", "--check")
        self.assertTrue(out.is_file())

    def test_scales_in_range(self):
        # every constructed type decodes to O(1e-2..1) weights: no denormal or huge super-scale
        rng = np.random.default_rng(0)
        for ty in F.CONSTRUCTED:
            blk, bsz = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[ty]]
            b = F.constructed_blocks(rng, ty, 64)
            self.assertEqual(b.shape, (64, bsz))
            v = gguf.quants.dequantize(b.reshape(-1), gguf.GGMLQuantizationType[ty])
            m = float(np.abs(v).max())
            self.assertTrue(np.all(np.isfinite(v)), ty)
            self.assertGreater(m, 1e-3, ty)
            self.assertLess(m, 10.0, ty)

    def test_check_catches_a_broken_block(self):
        out = self.make("b.gguf", "--types", "IQ2_XXS")
        self.assertEqual(F.check(out, ["IQ2_XXS"], 256, 256, 8), [])
        # set one block's d to +inf (fp16 0x7c00) inside the first tensor's data
        r = gguf.GGUFReader(str(out))
        off = next(t for t in r.tensors if t.name == "blk.0.ffn_gate_exps.weight").data_offset
        del r   # (its memory map: Windows refuses to rewrite a mapped file)
        raw = bytearray(out.read_bytes())
        raw[off:off + 2] = (0x7C00).to_bytes(2, "little")
        out.write_bytes(bytes(raw))
        bad = F.check(out, ["IQ2_XXS"], 256, 256, 8)
        self.assertTrue(any("non-finite" in b for b in bad), bad)
        # and a wrong expectation of the type is reported
        self.assertTrue(F.check(out, ["Q2_K"], 256, 256, 8))

    def test_refuses_bad_geometry_and_types(self):
        with self.assertRaises(SystemExit):
            F.main(["--out", str(self.d / "e.gguf"), "--types", "Q2_K", "--n-embd", "300"])
        with self.assertRaises(SystemExit):
            F.main(["--out", str(self.d / "e.gguf"), "--types", "NOPE"])


if __name__ == "__main__":
    unittest.main()
