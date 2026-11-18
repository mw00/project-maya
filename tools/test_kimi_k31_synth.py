"""tools/test_kimi_k31_synth.py - the synthetic Kimi K3.1 fixture (tools/kimi_k31_synth_gguf.py).

Builds the fixture into a temporary directory and checks:

  * the selftest (every file re-read: tiny.gguf against the converter transforms of hf-f32,
    hf-mxfp4 against hf-f32 bit for bit, the release's dtypes, the A_log zero tail, the configs);
  * determinism: the same seed writes the same bytes, another seed different weights;
  * the number formats against hand-worked values (bf16 rounding, the mxfp4-pack nibble order and
    the E8M0 scale), so the packer and the unpacker cannot share one mistake;
  * the oracle's own loader (ref/load.py) reads the 4-D conv and the stacked experts;
  * with KIMI_K31_MXFP4_GGUF (a GGUF converted from hf-mxfp4 by a converter that keeps the experts MXFP4, such
    as llama.cpp's convert_hf_to_gguf.py), ref/load.py decodes its MXFP4 experts bit-equal to the F32 copy
    (KIMI_K31_GGUF, default the tiny.gguf built here): values made outside ggml, not by another decoder;
  * the fixture requirements a K3.1 port depends on (layer layout, block boundaries, top-k > 8,
    the 576-wide MLA latent, an index_topk that a short sequence exercises);
  * the streamed GGUF writer writes tools/glm_synth_gguf.py write_gguf's bytes, and the tiny test
    sequences clear the tie bounds on a whole-sequence prefill;
  * the gate's pick / verify / gap commands refuse a GGUF of another geometry; the unit importance
    matrix (--unit-imatrix) has the layout llama-quantize reads;
  * the big geometry without writing it (2.9 GB): its requirements (each refused when broken), its
    sequences, the pick record against the gate's bounds and the current geometry, and the recorded
    float32-vs-float64 gap. With KIMI_K31_BIG_FIXTURE=<dir of a --geometry big build> the big
    fixture's selftest (which pins its files to the pick) and a whole-prefill check of its picked
    sequences under both eps presets run too (~2 min).

CPU only, numpy only (no torch, gguf or safetensors package), a few seconds:
`python -m unittest tools.test_kimi_k31_synth` (or pytest, or unittest discover).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))

import numpy as np  # noqa: E402

import kimi_k31_synth_gguf as ks  # noqa: E402

P = ks.P


class KimiK31SynthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-k31-synth-")
        cls.out = pathlib.Path(cls.tmp.name) / "a"
        cls.manifest = ks.build(cls.out)
        cls.vals, cls.dtypes, cls.mx = ks.make_weights(ks.TINY, ks.SEED)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_selftest(self):
        ks.selftest(self.out)

    def test_deterministic(self):
        again = ks.build(pathlib.Path(self.tmp.name) / "b")
        self.assertEqual({f: i["sha256"] for f, i in again["files"].items()},
                         {f: i["sha256"] for f, i in self.manifest["files"].items()})
        other, _, _ = ks.make_weights(ks.TINY, ks.SEED + 1)
        name = P + "layers.7.self_attn.indexer.wk.weight"
        self.assertFalse(np.array_equal(other[name], self.vals[name]))

    def test_bf16_round(self):
        x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, 1.0 + 2 ** -9], dtype=np.float32)
        # ties go to even: 1+2^-8 is half way between 1 and 1+2^-7 (mantissa 0 is even)
        want = np.array([1.0, 1.0, 1.0 + 2 ** -6, -2.5, 1.0], dtype=np.float32)
        self.assertTrue(np.array_equal(ks.bf16_round(x), want))
        bits = ks.f32_to_bf16_bits(want)
        self.assertEqual(int(bits[0]), 0x3F80)
        self.assertTrue(np.array_equal(ks.bf16_bits_to_f32(bits), want))
        with self.assertRaises(ValueError):
            ks.f32_to_bf16_bits(np.array([1.0 + 2 ** -10], dtype=np.float32))

    def test_mxfp4_by_hand(self):
        # one row, one group of 32: element 0 = code 1 (+0.5), element 1 = code 15 (-6),
        # element 31 = code 7 (+6); scale byte 125 = 2^-2
        codes = np.zeros((1, 32), dtype=np.uint8)
        codes[0, 0], codes[0, 1], codes[0, 31] = 1, 15, 7
        packed, scale = ks.mxfp4_pack(codes, np.array([[125]], dtype=np.uint8))
        self.assertEqual(packed.shape, (1, 16))
        self.assertEqual(int(packed[0, 0]), 0xF1)          # element 0 low nibble, element 1 high
        self.assertEqual(int(packed[0, 15]), 0x70)
        v = ks.mxfp4_dequant(packed, scale)
        self.assertEqual(float(v[0, 0]), 0.125)
        self.assertEqual(float(v[0, 1]), -1.5)
        self.assertEqual(float(v[0, 31]), 1.5)
        self.assertEqual(int(np.count_nonzero(v)), 3)

    def test_experts_exact_and_no_negative_zero(self):
        for name, (codes, scales) in list(self.mx.items())[:60]:
            self.assertFalse(np.any(codes == 8), name)    # -0 would not survive every dequantizer
            self.assertTrue(np.all((scales >= 118) & (scales <= 126)), name)
            v = self.vals[name]
            self.assertFalse(np.any(np.signbit(v) & (v == 0)), name)
            self.assertTrue(np.array_equal(ks.bf16_round(v), v), name)   # FP4 x 2^k is bf16 exact

    def test_hf_mxfp4_names_and_shapes(self):
        st = ks.read_safetensors(self.out / "hf-mxfp4" / "model.safetensors")
        g = ks.TINY
        w1 = st[P + "layers.1.block_sparse_moe.experts.0.w1.weight_packed"]
        w2s = st[P + "layers.8.block_sparse_moe.experts.23.w2.weight_scale"]
        self.assertEqual((w1[0], w1[1].shape), ("U8", (g["n_ff_exp"], g["latent"] // 2)))
        self.assertEqual((w2s[0], w2s[1].shape), ("U8", (g["latent"], g["n_ff_exp"] // 32)))
        self.assertNotIn(P + "layers.1.block_sparse_moe.experts.0.w1.weight", st)
        self.assertEqual(st[P + "layers.3.self_attn.indexer.k_norm.bias"][0], "BF16")
        self.assertEqual(st[P + "layers.0.self_attn.A_log"][1].shape, (g["kda_head_dim"],))
        cfg = json.loads((self.out / "hf-mxfp4" / "config.json").read_text())
        qc = cfg["text_config"]["quantization_config"]
        self.assertEqual((qc["format"], qc["config_groups"]["group_0"]["weights"]["group_size"]),
                         ("mxfp4-pack-quantized", 32))
        self.assertNotIn("quantization_config",
                         json.loads((self.out / "hf-f32" / "config.json").read_text())["text_config"])

    def test_oracle_loader_reads_it(self):
        from load import Gguf                               # ref/load.py
        g = Gguf(self.out / "tiny.gguf")
        try:
            conv = g.tensor("blk.4.ssm_conv1d_k.weight")
            self.assertEqual(conv.shape, (1, ks.TINY["n_head"] * ks.TINY["kda_head_dim"], 1, ks.TINY["d_conv"]))
            k_conv = self.vals[P + "layers.4.self_attn.k_conv1d.weight"]
            self.assertTrue(np.array_equal(conv[0, :, 0, :], k_conv[:, 0, :]))
            up = g.tensor("blk.8.ffn_up_exps.weight")
            self.assertTrue(np.array_equal(up[11], self.vals[P + "layers.8.block_sparse_moe.experts.11.w3.weight"]))
            md = g.g.metadata
            self.assertEqual(md["kimi-k31.expert_used_count"], 10)
            self.assertEqual(md["kimi-k31.attention.layer_norm_epsilon"], np.float32(1e-5))
            self.assertEqual(len(md["tokenizer.ggml.tokens"]), ks.TINY["vocab"])
        finally:
            g.close()

    def test_mxfp4_twin_decodes_to_the_f32_copy(self):
        twin = os.environ.get("KIMI_K31_MXFP4_GGUF")
        if not twin:
            self.skipTest("set KIMI_K31_MXFP4_GGUF to a GGUF converted from the fixture's hf-mxfp4")
        from load import Gguf                               # ref/load.py
        f32 = pathlib.Path(os.environ.get("KIMI_K31_GGUF") or self.out / "tiny.gguf").expanduser()
        mx, ref = Gguf(pathlib.Path(twin).expanduser()), Gguf(f32)
        bits = lambda x: np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)   # noqa: E731
        try:
            names = sorted(t.name for t in mx.g.tensors if t.type_name == "MXFP4")
            experts = sorted(t.name for t in ref.g.tensors if t.name.endswith("_exps.weight"))
            self.assertTrue(experts)
            self.assertEqual(names, experts, "the twin's MXFP4 tensors are not the routed experts")
            for name in names:
                self.assertEqual(ref.info(name).type_name, "F32", name)
                a, b = mx.tensor(name), ref.tensor(name)
                self.assertEqual(a.shape, b.shape, name)
                self.assertTrue(np.array_equal(bits(a), bits(b)),
                                f"{name}: MXFP4 decode != {f32.name} ({int(np.sum(bits(a) != bits(b)))} values; "
                                "a twin of another fixture version needs that version's tiny.gguf in KIMI_K31_GGUF)")
                for e in (0, a.shape[0] - 1):
                    self.assertTrue(np.array_equal(bits(mx.expert(name, e)), bits(b[e])), f"{name}[{e}]")
        finally:
            mx.close()
            ref.close()

    def test_layout(self):
        lay = self.manifest["layers"]
        self.assertEqual(lay["mla_dsa_0idx"], [3, 7, 8])
        self.assertEqual(lay["kda_0idx"], [0, 1, 2, 4, 5, 6])
        self.assertEqual(lay["attn_res_block_pushes_0idx"], [0, 3, 6])
        tc = self.manifest["config"]["text_config"]
        self.assertEqual(tc["linear_attn_config"]["full_attn_layers"], [4, 8, 9])
        self.assertEqual(tc["dsa_config"]["dsa_layers"], [4, 8, 9])
        self.assertEqual(tc["num_experts_per_token"], 10)
        self.assertEqual(tc["top_k"], 50)                   # the generation default, not the MoE top-k
        self.assertEqual(tc["kv_lora_rank"] + tc["qk_rope_head_dim"], 576)
        self.assertLess(tc["linear_attn_config"]["num_heads"], tc["linear_attn_config"]["head_dim"])
        self.assertEqual(len(self.manifest["test_tokens"]), 32)
        self.assertTrue(all(0 <= t < tc["vocab_size"] for t in self.manifest["test_tokens"]))

    def test_stream_writer_is_write_gguf(self):
        import glm_synth_gguf
        md, ts = ks.to_gguf(self.vals, ks.TINY)
        a, b = pathlib.Path(self.tmp.name) / "w1.gguf", pathlib.Path(self.tmp.name) / "w2.gguf"
        glm_synth_gguf.write_gguf(a, md, ts)
        ks.write_gguf_stream(b, md, ts)
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_tiny_sequences(self):
        k = ks.TINY["idx_topk"]
        want = [
            {"name": "seq0", "tokens": ks.TEST_TOKENS["seq0"], "decode_from": 7, "tie_free": True,
             "why": f"32 tokens > index_topk {k}: the indexer selects for positions >= {k}"},
            {"name": "seq1", "tokens": ks.TEST_TOKENS["seq0"][:5], "decode_from": 1, "tie_free": True,
             "why": f"5 tokens <= index_topk {k}: every row is dense causal MLA"},
            {"name": "seq2", "tokens": ks.TEST_TOKENS["seq2"], "decode_from": 66, "tie_free": True,
             "why": "80 tokens: crosses fla's 64-token KDA chunk; decode continues past the chunk boundary"},
        ]
        self.assertEqual(self.manifest["test_sequences"], want)
        self.assertNotIn("test_sequences_pick", self.manifest)
        self.assertEqual(ks.pick_record(ks.TINY, ks.SEED), None)

    def test_tiny_sequences_tie_free(self):
        import kimi_k31
        import kimi_k31_gate as G
        self.assertEqual(G.pick_bounds("tiny"), {"router": G.PICK_ROOM * G.MOE_TIE,
                                                 "indexer": G.PICK_ROOM * G.IDX_SCORE_ATOL})
        res = G.verify_picks(self.out / "tiny.gguf", "tiny", ks.SEED)    # whole prefills, every eps preset
        self.assertTrue(res["ok"], res)
        self.assertEqual(sorted(res["presets"]), sorted(kimi_k31.EPS_PRESETS))
        for eps, per in res["presets"].items():
            for s in self.manifest["test_sequences"]:
                r = per[s["name"]]
                self.assertGreaterEqual(r["min_router_margin"], G.PICK_ROOM * G.MOE_TIE, (eps, s["name"]))
                self.assertGreaterEqual(r["min_indexer_margin"], G.PICK_ROOM * G.IDX_SCORE_ATOL, (eps, s["name"]))
                if len(s["tokens"]) <= ks.TINY["idx_topk"]:
                    self.assertEqual(r["min_indexer_margin"], float("inf"))

    def test_tiny_gap_inside_the_band(self):
        """TIE_BANDS["tiny"] covers the cut-margin shift the float32 reference and the other eps preset
        make on tiny's sequences, and float32 changes no selection there."""
        import kimi_k31_gate as G
        res = G.gap(self.out / "tiny.gguf", "tiny", ks.SEED)
        for k in ("router", "indexer"):
            self.assertLess(res["margin_shift_vs_f64_moonshot"][k], G.TIE_BANDS["tiny"][k], k)
        for eps, r in res["presets"].items():
            self.assertEqual((r["all"]["expert_rows_differ"], r["all"]["indexer_rows_differ"]), (0, 0), eps)
            self.assertLess(r["all"]["logits_max"], G.TOL["float32"]["logits_atol"], eps)

    def test_pick_refuses_another_geometry(self):
        import contextlib
        import io
        import kimi_k31_gate as G
        for cmd in (["pick"], ["pick", "--verify"], ["gap"]):
            err = io.StringIO()
            with self.subTest(cmd=cmd), contextlib.redirect_stderr(err):
                rc = G.main(cmd + ["--geometry", "big", "--gguf", str(self.out / "tiny.gguf")])
                self.assertEqual(rc, 2)
                self.assertIn("not the big fixture geometry (--geometry big)", err.getvalue())
                self.assertIn("expert_count 24 != 64", err.getvalue())

    def test_unit_imatrix(self):
        import struct
        from gguf_reader import GGUFFile
        path = pathlib.Path(self.tmp.name) / "unit.dat"
        n = ks.write_unit_imatrix(self.out / "tiny.gguf", path)
        mats = {t.name: t.shape for t in GGUFFile(self.out / "tiny.gguf").tensors if len(t.shape) >= 2}
        self.assertEqual(n, len(mats))
        raw, off, seen = path.read_bytes(), 4, {}
        self.assertEqual(struct.unpack_from("<i", raw)[0], n)
        for _ in range(n):
            (ln,) = struct.unpack_from("<i", raw, off)
            name = raw[off + 4:off + 4 + ln].decode()
            ncall, nval = struct.unpack_from("<ii", raw, off + 4 + ln)
            vals = np.frombuffer(raw, "<f4", nval, off + 12 + ln)
            seen[name] = nval
            self.assertEqual(ncall, 1)
            self.assertTrue((vals == 1).all(), name)
            off += 12 + ln + 4 * nval
        self.assertEqual(seen, {k: int(s[0]) * int(np.prod(s[2:], dtype=np.int64)) for k, s in mats.items()})
        self.assertEqual(seen["blk.1.ffn_up_exps.weight"], ks.TINY["latent"] * ks.TINY["n_expert"])
        self.assertEqual(struct.unpack_from("<i", raw, off)[0], 1)    # chunk count, then the dataset name


class KimiK31SynthBig(unittest.TestCase):
    """The big geometry (--geometry big), checked without writing it."""

    def setUp(self):
        self.g = ks.BIG
        self.seqs = ks.gate_sequences(self.g, ks.SEED)

    def test_requirements(self):
        ks.check_requirements(self.g, self.seqs)
        ks.check_requirements(ks.TINY, ks.gate_sequences(ks.TINY, ks.SEED))
        self.assertEqual(ks.geometry_name(self.g), "big")
        self.assertEqual(ks.gguf_name(self.g), "big.gguf")
        self.assertEqual(ks.gguf_name(ks.TINY), "tiny.gguf")
        self.assertEqual(list(self.g), list(ks.TINY))        # the same keys, so every helper reads both

    def test_requirements_refused(self):
        cases = [
            (dict(n_ff_dense=1800), "multiples of 256"),
            (dict(q_lora=640), "multiples of 256"),
            (dict(n_ff_exp=192), "multiples of 256"),
            (dict(n_expert=48), "16 of 64"),
            (dict(n_expert_used=8), "top-k > 8"),
            (dict(attn_res_block=4), "pushes at 3, 6 and 9"),
            (dict(attn_res_block=6), "two attn-res block boundaries"),
            (dict(idx_heads=16), "released shape"),
            (dict(idx_topk=96), "seq0 longer than index_topk"),
            (dict(idx_topk=31), "seq1 no longer than index_topk"),
            (dict(full_attn_layers=[4, 8, 10]), "last two layers MLA"),
            (dict(qk_rope=128), "576"),
        ]
        for change, msg in cases:
            with self.subTest(change=change):
                g = dict(self.g, **change)
                with self.assertRaises(AssertionError) as e:
                    ks.check_requirements(g, self.seqs, kind="big")
                self.assertIn(msg, str(e.exception))
        short = [dict(s, tokens=s["tokens"][:200], decode_from=150) if s["name"] == "seq2" else s for s in self.seqs]
        with self.assertRaises(AssertionError) as e:
            ks.check_requirements(self.g, short, kind="big")
        self.assertIn("several of fla's chunks", str(e.exception))

    def test_sequences(self):
        s = {q["name"]: q for q in self.seqs}
        self.assertEqual([(q["name"], len(q["tokens"]), q["decode_from"]) for q in self.seqs],
                         [("seq0", 96, 33), ("seq1", 32, 1), ("seq2", 300, 260)])
        self.assertTrue(all(q["tie_free"] for q in self.seqs))
        self.assertEqual(s["seq0"]["tokens"], ks.BIG_TEST_TOKENS["seq0"])
        self.assertEqual(s["seq2"]["tokens"], ks.BIG_TEST_TOKENS["seq2"])
        self.assertEqual(s["seq1"]["tokens"], ks.BIG_TEST_TOKENS["seq0"][:32])
        self.assertEqual(len(s["seq1"]["tokens"]), self.g["idx_topk"])     # dense, the last row at the cut
        self.assertGreater(s["seq0"]["decode_from"], self.g["idx_topk"])   # every decode step selects
        self.assertIn("crosses 4 of fla's 64-token KDA chunk boundaries", s["seq2"]["why"])
        self.assertEqual(ks.sample_tokens(self.g, ks.SEED), ks.BIG_TEST_TOKENS["seq0"])
        other = ks.gate_sequences(self.g, ks.SEED + 1)                     # another seed: random, not picked
        self.assertFalse(any(q["tie_free"] for q in other))
        self.assertEqual([len(q["tokens"]) for q in other], [96, 32, 300])
        self.assertEqual(ks.pick_record(self.g, ks.SEED + 1), None)

    def test_pick_record(self):
        """BIG_PICK against the gate's own numbers and the current geometry. The picked sequences against
        the big weights are checked by test_built_fixture (KIMI_K31_BIG_FIXTURE), and by selftest, which
        also requires the files to be the ones BIG_PICK pins."""
        import kimi_k31
        import kimi_k31_gate as G
        rec = ks.pick_record(self.g, ks.SEED)
        self.assertEqual(rec["bounds"], G.pick_bounds("big"))
        self.assertEqual(rec["presets"], list(G.PICK_EPS["big"]))
        self.assertEqual(sorted(G.PICK_EPS["big"]), sorted(kimi_k31.EPS_PRESETS))
        self.assertEqual(rec["picked_on"]["geometry_sha256"], ks.geometry_sha256(ks.BIG),
                         "BIG changed: re-pick (kimi_k31_gate.py pick --geometry big) and rebuild the fixture")
        self.assertEqual(sorted(rec["picked_on"]["sha256"]),
                         ["big.gguf", "hf-f32/model.safetensors", "hf-mxfp4/model.safetensors"])
        for name in ("seq0", "seq2"):
            self.assertEqual(rec[name]["tokens"], len(ks.BIG_TEST_TOKENS[name]))
            self.assertGreaterEqual(rec[name]["min_router_margin"], rec["bounds"]["router"])
            self.assertGreaterEqual(rec[name]["min_indexer_margin"], rec["bounds"]["indexer"])
        ver = rec["verified"]
        self.assertEqual(sorted(ver["min_router_margin"]), sorted(kimi_k31.EPS_PRESETS))
        for eps in kimi_k31.EPS_PRESETS:
            for name in ("seq0", "seq1", "seq2"):
                self.assertGreaterEqual(ver["min_router_margin"][eps][name], rec["bounds"]["router"], (eps, name))
                idx = float(ver["min_indexer_margin"][eps][name])
                self.assertGreaterEqual(idx, rec["bounds"]["indexer"], (eps, name))
            self.assertEqual(ver["min_indexer_margin"][eps]["seq1"], "inf")     # dense: no indexer cut

    def test_gap_record(self):
        """BIG_GAP (the float32-vs-float64 reference gap on the picked sequences) sits inside the big tie
        band and changed no selection; the M1a gate-2 bounds are 2x these numbers."""
        import kimi_k31
        import kimi_k31_gate as G
        gap = ks.gap_record(self.g, ks.SEED)
        self.assertEqual(sorted(gap["presets"]), sorted(kimi_k31.EPS_PRESETS))
        for k in ("router", "indexer"):
            self.assertLessEqual(gap["margin_shift_vs_f64_moonshot"][k], G.TIE_BANDS["big"][k], k)
        for eps, r in gap["presets"].items():
            self.assertEqual(sorted(r), ["all", "seq0", "seq1", "seq2"])
            a = r["all"]
            self.assertEqual((a["expert_rows_differ"], a["indexer_rows_differ"]), (0, 0), eps)
            for key in ("logits_max", "logits_rms", "mla_seam_max", "mla_seam_rms", "kda_seam_max", "kda_seam_rms",
                        "router_score_max", "indexer_score_max"):
                self.assertGreater(a[key], 0, (eps, key))
                if key.endswith("_max"):
                    self.assertEqual(a[key], max(r[s][key] for s in ("seq0", "seq1", "seq2")), (eps, key))
            if eps == "moonshot":                          # one of the runs the overall shift is taken over
                for k in ("router", "indexer"):
                    self.assertLessEqual(a[k + "_margin_shift_max"], gap["margin_shift_vs_f64_moonshot"][k], k)
            self.assertEqual(r["seq1"]["indexer_margin_shift_max"], 0.0)          # dense: no indexer cut
        self.assertIsNone(ks.gap_record(self.g, ks.SEED + 1))
        self.assertIsNone(ks.gap_record(ks.TINY, ks.SEED))

    @unittest.skipUnless(os.environ.get("KIMI_K31_BIG_FIXTURE"),
                         "KIMI_K31_BIG_FIXTURE not set (the directory of a --geometry big build)")
    def test_built_fixture(self):
        import kimi_k31_gate as G
        d = pathlib.Path(os.environ["KIMI_K31_BIG_FIXTURE"]).expanduser()
        ks.selftest(d, self.g)                             # includes the BIG_PICK file pins
        man = json.loads((d / "manifest.json").read_text())
        self.assertEqual(man["test_sequences_pick"], ks.pick_record(self.g, ks.SEED))
        self.assertEqual(man["reference_gap"], ks.gap_record(self.g, ks.SEED))
        res = G.verify_picks(d / "big.gguf", "big", ks.SEED)      # every sequence, every eps preset
        self.assertTrue(res["ok"], res)
        rec = ks.pick_record(self.g, ks.SEED)["verified"]
        for eps, per in res["presets"].items():
            for name, r in per.items():
                self.assertAlmostEqual(r["min_router_margin"], rec["min_router_margin"][eps][name], delta=1e-12)
                self.assertAlmostEqual(r["min_indexer_margin"], float(rec["min_indexer_margin"][eps][name]),
                                       delta=1e-12)


if __name__ == "__main__":
    unittest.main()
