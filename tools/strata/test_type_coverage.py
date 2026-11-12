"""tools/strata/test_type_coverage.py - the B5.3 coverage table (tools/strata/type_coverage.py) on synthetic GGUFs.

  * the engine tables parsed from the source agree with the pack's own lists (glm_dsa_pack.MOE_TYPES, MV_TYPES,
    UPCAST_TYPES);
  * PACK9's header (community non-routed types, 9 expert size classes; header only, no tensor data): everything covered, Q8_0 attn_k_b upcast to BF16,
    9 classes counted;
  * the B13.1 expert fixture (tools/iq_fixture.py: one layer per GLM-5.3 expert type) is covered;
  * a header dump with a routed Q4_0 down projection and a TQ1_0 attn_output is UNCOVERED (exit 1); an F16
    attn_q_b is an upcast by default and native with --native32;
  * the audit-TSV input gives the same rows as the GGUF.
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import gguf_audit as ga  # noqa: E402
import glm_dsa_pack as gp  # noqa: E402
import type_coverage as tc  # noqa: E402

try:
    import glm_dsa_synth_gguf as synth  # noqa: E402
    _ERR = None
except ImportError as e:
    _ERR = e


def header_only(path: pathlib.Path, preset: str) -> None:
    """The preset's GGUF header (key/values and the tensor table) without tensor data: all the coverage table reads,
    in a few KB instead of the full file."""
    import numpy as np
    g = synth._gguf()
    b = synth.Builder(preset, synth.DEFAULT_SEED)
    w = g.GGUFWriter(path, synth.ARCH)
    VT = g.GGUFValueType
    for key, kind, val in b.metadata():
        if kind == "u32":
            w.add_uint32(key, int(val))
        elif kind == "f32":
            w.add_float32(key, float(val))
        elif kind == "bool":
            w.add_bool(key, bool(val))
        elif kind == "str":
            w.add_string(key, str(val))
        elif kind == "arr_bool":
            w.add_key_value(key, [bool(x) for x in val], VT.ARRAY, sub_type=VT.BOOL)
        elif kind == "arr_i32":
            w.add_key_value(key, [int(x) for x in val], VT.ARRAY, sub_type=VT.INT32)
        elif kind == "arr_str":
            w.add_key_value(key, list(val), VT.ARRAY, sub_type=VT.STRING)
    for s in b.tensors():
        if s.type_name == "F32":
            w.add_tensor_info(s.name, tuple(reversed(s.ne)), np.dtype(np.float32), s.nbytes())
        elif s.type_name == "F16":
            w.add_tensor_info(s.name, tuple(reversed(s.ne)), np.dtype(np.float16), s.nbytes())
        else:
            blk, bsz = synth.quant_geometry(s.type_name)
            w.add_tensor_info(s.name, tuple(reversed(s.ne[1:])) + (s.ne[0] // blk * bsz,), np.dtype(np.uint8),
                              s.nbytes(), raw_dtype=g.GGMLQuantizationType[s.type_name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()


def quiet(fn, *a):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = fn(*a)
    return rc, out.getvalue()


class Tables(unittest.TestCase):
    def test_parsed_tables_match_the_pack(self):
        t = tc.engine_tables(ROOT, None)
        ids = lambda names: {ga.TYPE_ID[n] for n in names}  # noqa: E731
        self.assertEqual(t["moe"], ids(gp.MOE_TYPES))
        self.assertEqual(t["mv"], ids(gp.MV_TYPES))
        self.assertEqual(t["mv_ext"], ids(gp.UPCAST_TYPES))
        self.assertTrue({ga.TYPE_ID[n] for n in ("Q2_K", "IQ2_XXS", "IQ3_XXS")} <= t["mmq"])
        self.assertTrue({ga.TYPE_ID[n] for n in ("Q2_K", "IQ2_XXS", "IQ1_S", "IQ4_XS")} <= t["vec_dot"])
        self.assertIn(ga.TYPE_ID["Q8_0"], t["f16"])


class Files(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _ERR is not None:
            raise unittest.SkipTest(f"gguf-py not importable: {_ERR}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = pathlib.Path(cls.tmp.name)
        cls.pack9 = cls.d / "pack9.gguf"
        header_only(cls.pack9, "PACK9")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pack9(self):
        out = self.d / "tc.md"
        rc, _ = quiet(tc.main, [str(self.pack9), "--out", str(out), "--native32", "--root", str(ROOT)])
        self.assertEqual(rc, 0)
        md = out.read_text()
        self.assertIn("every type covered", md)
        self.assertIn("**9**", md)
        self.assertIn("native_expert_parity", md)
        tensors, _ = tc.load_tensors([str(self.pack9)])
        cov = tc.coverage(tensors, tc.engine_tables(ROOT, None), tc.oracle_types(), native32=False)
        kb = [r for r in cov["dense"] if r["tensor"] == "attn_k_b.weight"][0]
        self.assertEqual((kb["type"], kb["verdict"]), ("Q8_0", "upcast to BF16"))
        self.assertEqual(len(cov["classes"]), 9)
        self.assertEqual(cov["uncovered"], [])

    def test_iq_fixture(self):
        f = self.d / "iq.gguf"
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "iq_fixture.py"), "--out", str(f), "--n-embd", "256",
                            "--n-ff", "256"], capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest(f"iq_fixture.py: {r.stderr[-300:]}")
        tensors, _ = tc.load_tensors([str(f)])
        cov = tc.coverage(tensors, tc.engine_tables(ROOT, None), tc.oracle_types(), native32=False)
        types = {r["type"] for r in cov["routed"]}
        self.assertTrue({"Q2_K", "Q6_K", "IQ2_XXS", "IQ1_S", "IQ4_XS"} <= types, types)
        self.assertEqual([r for r in cov["routed"] if r["verdict"] != "covered"], [])

    def edited_dump(self, edits: dict[str, str]) -> pathlib.Path:
        shards = ga.load_local([str(self.pack9)])
        p = self.d / f"hdr{len(edits)}{abs(hash(tuple(edits.items())))}.json"
        ga.dump_header(shards, p, "test")
        doc = json.loads(p.read_text())
        hit = set()
        for s in doc["shards"]:
            for t in s["tensors"]:
                if t[0] in edits:
                    t[1] = ga.TYPE_ID[edits[t[0]]]
                    hit.add(t[0])
        self.assertEqual(hit, set(edits))
        p.write_text(json.dumps(doc))
        return p

    def test_uncovered_types(self):
        p = self.edited_dump({"blk.4.ffn_down_exps.weight": "Q4_0", "blk.3.attn_output.weight": "TQ1_0"})
        out = self.d / "bad.md"
        rc, _ = quiet(tc.main, [str(p), "--out", str(out)])
        self.assertEqual(rc, 1)
        md = out.read_text()
        self.assertIn("UNCOVERED: ", md)
        self.assertIn("routed down Q4_0", md)
        self.assertIn("attn_output.weight TQ1_0", md)

    def test_native32_and_tsv(self):
        p = self.edited_dump({"blk.2.attn_q_b.weight": "F16"})
        tab = tc.engine_tables(ROOT, None)
        t, _ = tc.load_tensors([str(p)])
        row = lambda cov: [r for r in cov["dense"] if r["tensor"] == "attn_q_b.weight" and r["type"] == "F16"][0]  # noqa: E731
        self.assertEqual(row(tc.coverage(t, tab, None, False))["verdict"], "upcast to BF16")
        # native F16 is served by mv (B10.7) and by the batched prompt's FP16 dequant (glm_prefill_types.hpp, W7)
        nat = tc.coverage(t, tab, None, True)
        self.assertEqual(row(nat)["verdict"], "covered")
        self.assertEqual(nat["prefill_gaps"], [])
        # a prompt without the extra types would report the gap (the token path)
        less = dict(tab, f16=tab["f16"] - {1})
        gap = tc.coverage(t, less, None, True)
        self.assertEqual(row(gap)["verdict"], "covered; batched prefill refuses it (token path)")
        self.assertIn("attn_q_b.weight F16", gap["prefill_gaps"])
        # the same tensors through an audit-style TSV
        tsv = self.d / "x_audit.tsv"
        lines = ["name\ttype\tdims\tshard\tbytes\tpacked_bytes\trole\tgpu"]
        lines += [f"{x.name}\t{x.tname}\t{','.join(map(str, x.dims))}\t1\t0\t0\tdense\t0" for x in t]
        tsv.write_text("\n".join(lines) + "\n")
        t2, src = tc.load_tensors([str(tsv)])
        self.assertIn("audit TSV", src)
        a = tc.coverage(t, tab, None, False)
        b = tc.coverage(t2, tab, None, False)
        self.assertEqual(a["routed"], b["routed"])
        self.assertEqual(a["dense"], b["dense"])

    def test_usage(self):
        rc, _ = quiet(tc.main, [str(self.d / "missing.gguf")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
