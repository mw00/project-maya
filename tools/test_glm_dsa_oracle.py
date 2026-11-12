"""tools/test_glm_dsa_oracle.py - the glm-dsa oracle (ref/glm_dsa.py) and its fixtures, CPU only.

glm-dsa BUILD-PLAN B3.3.  Runs in the Python CI job (tests.yml discovers tools/test_*.py) in about
20 s on a 4-core container.  Every fixture is generated here by tools/glm_dsa_synth_gguf.py, so
nothing large is committed; data/glm-dsa/ holds only hashes and reference logits.

  * invariants that hold by construction (causality, dense = sparse below top-k, shared layers
    reuse the previous full selection, absorbed MLA = naive MLA with scale 1/sqrt(256), MoE weights
    sum to the scale, Hadamard is a no-op, HF and llama readings differ only where documented);
  * the oracle against committed llama.cpp aa94f208 CPU logits (tools/strata/glm_dsa_llama_logits)
    on SMALL (48 tokens, top-8 selection active), LONG (2080 tokens, top-2048 active) and the
    quantized PACK/TIERS files;
  * every named misreading (MUTATIONS) moves the logits by more than 10x the parity gate;
  * the loader against gguf-py's own reader for every ggml type the fixtures carry;
  * the generator is byte-identical to data/glm-dsa/sha256.txt.

Gate values (BUILD-PLAN B3.5/B3.6): logits max abs error 1e-4; KL p99 1e-4 with KL taken over the
reference's top-64 tokens plus a rest bucket.

Opt-in (not part of CI): MAYA_SYNTH_LARGE=1 also generates BIG (~2.3 GB) and WIDE (~1 GB) and runs
the oracle on BIG; tools/strata/glm_dsa_hf_xcheck.py compares against transformers when torch is
installed.

    python -m unittest tools.test_glm_dsa_oracle -v
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ref"))

import glm_dsa_synth_gguf as synth  # noqa: E402
from glm_dsa import (MUTATIONS, COVER, GlmDsa, GlmDsaConfig, Options, apply_rope, hadamard,  # noqa: E402
                     kl_top_rest, rope_angles)
from glm_dsa_load import GgufAny  # noqa: E402

DATA = ROOT / "data" / "glm-dsa"
LOGIT_GATE = 1e-4          # B3.5: logits max abs error vs the oracle
KL_GATE = 1e-4             # B3.6: KL p99 vs the oracle
_T0 = time.time()


def toks(n: int, n_vocab: int) -> list[int]:
    return [(i * 37 + 11) % n_vocab for i in range(n)]


def ref(name: str, n_vocab: int) -> np.ndarray:
    """A committed (T, vocab) float32 reference, returned as (vocab, T) float64."""
    return np.fromfile(DATA / name, dtype=np.float32).reshape(-1, n_vocab).T.astype(np.float64)


_FILES: dict[str, pathlib.Path] = {}
_TMP: tempfile.TemporaryDirectory | None = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory(prefix="strata-glm-dsa-")
    for stem, preset, kb in synth.default_set(include_large=False):
        p = pathlib.Path(_TMP.name) / f"{stem}.gguf"
        synth.write(p, preset, synth.DEFAULT_SEED, kb)
        _FILES[stem] = p


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()
    print(f"\ntest_glm_dsa_oracle: {time.time() - _T0:.1f} s", file=sys.stderr)


# ============================================================================ SMALL
class SmallOracle(unittest.TestCase):
    T = 48

    @classmethod
    def setUpClass(cls):
        cls.m = GlmDsa(_FILES["glm-dsa-small"], Options(mode="hf"), cache=True)
        cls.c = cls.m.cfg
        cls.tokens = toks(cls.T, cls.c.n_vocab)
        COVER.clear()
        cls.m.collect = True
        cls.logits = cls.m.forward(cls.tokens)
        cls.dumps = cls.m.dumps
        cls.h_nextn = cls.m.h_nextn
        cls.cover = dict(COVER)
        cls.m.collect = False

    @classmethod
    def tearDownClass(cls):
        cls.m.close()

    def view(self, **kw) -> GlmDsa:
        return self.m.with_options(Options(**kw))

    # ---- config and fixture coverage
    def test_config(self):
        c = self.c
        self.assertEqual((c.n_layer, c.n_nextn, c.n_embd, c.n_head), (8, 1, 128, 4))
        self.assertEqual((c.qk_nope, c.n_rot, c.k_mla, c.v_mla, c.kv_lora, c.q_lora), (48, 16, 64, 32, 32, 64))
        self.assertEqual((c.idx_heads, c.idx_dim, c.top_k), (16, 64, 8))
        self.assertEqual([i for i in range(8) if c.idx_full[i]], [0, 1, 2, 6])
        self.assertEqual(c.idx_types_source, "gguf")
        self.assertEqual((c.n_expert, c.n_used, c.dense_lead, c.w_scale, c.w_norm), (16, 4, 3, 2.5, True))
        self.assertEqual(c.rms_eps, np.float32(1e-5))
        self.assertEqual(c.rope_base, 8e6)

    def test_fixture_exercises_guarded_paths(self):
        self.assertGreater(self.cover["swiglu_over10"], 0, "no SwiGLU input beyond 10: clamp untestable")
        self.assertGreater(self.cover["idx_rows_selecting"], 0, "top-k never selects")
        self.assertEqual(self.cover["idx_near_ties"], 0, "f32 references would flip near-tied selections")
        for il in range(self.c.n_layer):
            if not self.c.idx_full[il]:
                self.assertTrue(self.m.has(il, "indexer.attn_q_b"), "SMALL carries indexers on shared layers")

    def test_logits_finite(self):
        self.assertEqual(self.logits.shape, (self.c.n_vocab, self.T))
        self.assertTrue(np.isfinite(self.logits).all())

    # ---- committed references
    def test_matches_committed_oracle(self):
        want = ref("oracle-hf-small-48.f32", self.c.n_vocab)
        self.assertLess(np.abs(self.logits - want).max(), 1e-5)

    def test_llama_mode_matches_llama_cpp(self):
        """B3.1 acceptance: llama mode vs llama.cpp aa94f208 CPU (F32 KV, FA off) inside the gates."""
        got = self.view(mode="llama").forward(self.tokens)
        want = ref("llama-small-48.f32", self.c.n_vocab)
        self.assertLess(np.abs(got - want).max(), LOGIT_GATE)
        self.assertLess(np.percentile(kl_top_rest(got, want), 99), KL_GATE)
        self.assertTrue((got.argmax(0) == want.argmax(0)).all())

    def test_mtp_matches_committed_oracle(self):
        d = self.m.mtp(self.h_nextn[:, :-1], self.tokens[1:])
        want = ref("oracle-hf-small-mtp-47.f32", self.c.n_vocab)
        self.assertLess(np.abs(d - want).max(), 1e-5)

    # ---- invariants
    def test_causal_prefix(self):
        pre = self.m.forward(self.tokens[:13])
        self.assertLess(np.abs(pre - self.logits[:, :13]).max(), 1e-10)

    def test_dense_equals_sparse_below_top_k(self):
        dense = self.view(top_k=10 ** 9).forward(self.tokens)
        k = self.c.top_k
        self.assertLess(np.abs(dense[:, :k] - self.logits[:, :k]).max(), 1e-12)
        self.assertGreater(np.abs(dense[:, k:] - self.logits[:, k:]).max(), 10 * LOGIT_GATE)

    def test_layer_slices_compose(self):
        """B7.5's slice mode: layers [0, 6) then [6, 8) (6 is a full layer) equal one pass; a slice
        may not start on a shared layer (its selection lives in an earlier layer)."""
        c = self.c
        x = self.m.embed(self.tokens)
        pos = self.m.positions(self.T, 0)
        mid = self.m.run_layers(x, 0, 6, pos)
        out = self.m.run_layers(mid, 6, c.n_layer, pos)
        np.testing.assert_allclose(out, self.dumps[f"l_out-{c.n_layer - 1}"], rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            self.m.run_layers(mid, 4, c.n_layer, pos)

    def test_selection_sizes_and_causality(self):
        for il in range(self.c.n_layer):
            sel = self.dumps[f"top_k-{il}"]
            T = sel.shape[0]
            self.assertFalse(np.triu(sel, 1).any(), f"layer {il}: future cell selected")
            np.testing.assert_array_equal(sel.sum(axis=1), np.minimum(np.arange(1, T + 1), self.c.top_k))

    def test_shared_layers_reuse_previous_full(self):
        last = None
        for il in range(self.c.n_layer):
            sel = self.dumps[f"top_k-{il}"]
            if self.c.idx_full[il]:
                last = sel
            else:
                np.testing.assert_array_equal(sel, last, f"layer {il}")
        self.assertFalse(np.array_equal(self.dumps["top_k-2"], self.dumps["top_k-6"]))

    def test_moe_weights(self):
        for il in range(self.c.dense_lead, self.c.n_layer):
            w = self.dumps[f"ffn_moe_weights-{il}"]
            np.testing.assert_allclose(w.sum(axis=0), self.c.w_scale, atol=1e-12)
            top = self.dumps[f"ffn_moe_topk-{il}"]
            probs = self.dumps[f"ffn_moe_probs-{il}"]
            cols = np.arange(top.shape[1])[None, :]
            p = probs[top, cols]
            np.testing.assert_allclose(w, p / p.sum(axis=0) * self.c.w_scale, atol=1e-12)
            # selection uses the biased score: every chosen biased score >= every rejected one
            bias = self.m.L(il, "exp_probs_b", "bias")
            biased = probs + bias[:, None]
            for t in range(top.shape[1]):
                chosen = biased[top[:, t], t]
                rest = np.delete(biased[:, t], top[:, t])
                self.assertGreaterEqual(chosen.min(), rest.max())

    def test_indexer_weight_scale(self):
        c = self.c
        h = self.dumps["attn_norm-0"]
        w = self.dumps["indexer_weights-0"]
        np.testing.assert_allclose(w, (self.m.L(0, "indexer.proj") @ h) / math.sqrt(c.idx_heads * c.idx_dim),
                                   rtol=0, atol=1e-13)
        self.assertEqual(math.sqrt(c.idx_heads * c.idx_dim), 32.0)

    def test_absorbed_mla_equals_naive_mla(self):
        """Decompress K and V per head (HF form: k_h = [W_kb,h^T c ; k_pe], scale 1/sqrt(nope+rope))
        and compare with the absorbed kqv_out the oracle computed, on a full and a shared layer."""
        c = self.c
        for il in (0, 4):
            qr = self.dumps[f"qr-{il}"]
            c_kv = self.dumps[f"kv_cmpr-{il}"]
            k_pe = self.dumps[f"k_pe-{il}"]
            q_pe = self.dumps[f"q_pe-{il}"]
            sel = self.dumps[f"top_k-{il}"]
            q = self.m.L(il, "attn_q_b") @ qr
            q = q.reshape(c.n_head, c.k_mla, -1)                              # (H, 256, T)
            wk_b, wv_b = self.m.L(il, "attn_k_b"), self.m.L(il, "attn_v_b")   # (H, kv, nope), (H, v, kv)
            out = np.zeros((c.n_head, c.v_mla, q.shape[2]))
            for h in range(c.n_head):
                k_nope = wk_b[h].T @ c_kv                                     # (nope, S)
                v = wv_b[h] @ c_kv                                            # (v, S)
                qh = np.concatenate([q[h, : c.qk_nope], q_pe[:, h, :]], axis=0)
                kh = np.concatenate([k_nope, k_pe], axis=0)
                s = (qh.T @ kh) / math.sqrt(c.qk_nope + c.n_rot)
                s[~sel] = -np.inf
                p = np.exp(s - s.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                out[h] = v @ p.T
            np.testing.assert_allclose(out.reshape(c.n_head * c.v_mla, -1), self.dumps[f"kqv_out-{il}"],
                                       rtol=0, atol=1e-11)

    def test_hadamard_is_a_noop(self):
        had = self.view(mode="hf", hadamard=True).forward(self.tokens)
        self.assertLess(np.abs(had - self.logits).max(), 1e-9)
        H = hadamard(64)
        np.testing.assert_allclose(H @ H.T, np.eye(64), atol=1e-14)

    def test_llama_reading_differs_only_where_documented(self):
        """llama mode with HF's eps and angles equals HF mode (the Hadamard is a no-op, the 6.1e-5
        clamp never binds); the MTP block differs only by dense attention."""
        m = self.view(mode="llama", idx_ln_eps=1e-6, rope_angles="hf", mtp_dense=False)
        self.assertLess(np.abs(m.forward(self.tokens) - self.logits).max(), 1e-9)
        d1 = m.mtp(self.h_nextn[:, :-1], self.tokens[1:])
        d0 = self.m.mtp(self.h_nextn[:, :-1], self.tokens[1:])
        self.assertLess(np.abs(d1 - d0).max(), 1e-9)

    def test_mtp_dense_equals_indexer_below_top_k(self):
        k = self.c.top_k
        a = self.m.mtp(self.h_nextn[:, :k], self.tokens[1:k + 1])
        b = self.view(mode="hf", mtp_dense=True).mtp(self.h_nextn[:, :k], self.tokens[1:k + 1])
        self.assertLess(np.abs(a - b).max(), 1e-12)
        a = self.m.mtp(self.h_nextn[:, :-1], self.tokens[1:])
        b = self.view(mode="hf", mtp_dense=True).mtp(self.h_nextn[:, :-1], self.tokens[1:])
        self.assertGreater(np.abs(a - b).max(), 10 * LOGIT_GATE)

    def test_eps_reading_matters_on_this_fixture(self):
        """SMALL's indexer keys have variance near 1e-6 on purpose: llama.cpp's eps 0 and HF's 1e-6
        select different cells here (on the real model the difference is what M2 (g) records)."""
        llama = self.view(mode="llama").forward(self.tokens)
        self.assertGreater(np.abs(llama - self.logits).max(), 10 * LOGIT_GATE)

    # ---- the named misreadings
    def test_each_mutation_breaks_the_gate(self):
        for name in sorted(MUTATIONS):
            with self.subTest(mutation=name):
                bad = self.view(mutations={name}).forward(self.tokens)
                dmax = float(np.abs(bad - self.logits).max())
                kl99 = float(np.percentile(kl_top_rest(self.logits, bad), 99))
                self.assertGreater(dmax, 10 * LOGIT_GATE, f"{name}: max |dlogit| {dmax:.3g}")
                self.assertGreater(kl99, 10 * KL_GATE, f"{name}: KL p99 {kl99:.3g}")

    def test_unknown_mutation_refused(self):
        with self.assertRaises(ValueError):
            Options(mutations={"no_such_thing"})


# ============================================================================ RoPE
class Rope(unittest.TestCase):
    def test_norm_pairs_are_adjacent(self):
        x = np.zeros((8, 1, 1))
        x[0, 0, 0] = 1.0
        ang = np.full((4, 1), math.pi / 2)
        y = apply_rope(x, ang, neox=False)
        np.testing.assert_allclose(y[:, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], atol=1e-15)
        y = apply_rope(x, ang, neox=True)
        np.testing.assert_allclose(y[:, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], atol=1e-15)

    def test_relative_position_property(self):
        rng = np.random.default_rng(0)
        q, k = rng.standard_normal((16, 1, 1)), rng.standard_normal((16, 1, 1))
        inv = 8e6 ** (-np.arange(8) * 2 / 16)
        def at(x, p):
            return apply_rope(x, (inv * p)[:, None], neox=False)[:, 0, 0]
        self.assertAlmostEqual(float(at(q, 7) @ at(k, 3)), float(at(q, 104) @ at(k, 100)), places=12)

    def test_angle_models(self):
        pos = np.array([0, 1, 2047, 2048, 131071, 1000000])
        hf = rope_angles(pos, 64, 8e6, "hf")
        inv = np.array([np.float32(8e6 ** (-2.0 * i / 64)) for i in range(32)], dtype=np.float32)
        np.testing.assert_array_equal(hf, (pos.astype(np.float32)[None, :] * inv[:, None]).astype(np.float64))
        self.assertTrue((hf[:, 0] == 0).all())
        gg = rope_angles(pos, 64, 8e6, "ggml")
        np.testing.assert_array_equal(gg[0], pos.astype(np.float32))
        # the two models agree to f32 rounding at small positions
        self.assertLess(np.abs(gg[:, :2] - hf[:, :2]).max(), 1e-6)


# ============================================================================ LONG
class LongOracle(unittest.TestCase):
    """top-2048 selection active: 2080 tokens, compared with llama.cpp on rows 2048..2079."""

    @classmethod
    def setUpClass(cls):
        cls.m = GlmDsa(_FILES["glm-dsa-long"], Options(mode="hf"), cache=True)
        cls.tokens = toks(2080, cls.m.cfg.n_vocab)
        COVER.clear()
        cls.logits = cls.m.forward(cls.tokens)
        cls.cover = dict(COVER)

    @classmethod
    def tearDownClass(cls):
        cls.m.close()

    def test_back_compat_indexer_schedule(self):
        c = self.m.cfg
        self.assertEqual(c.idx_types_source, "bc-glm52")
        self.assertEqual([i for i in range(c.n_layer) if c.idx_full[i]], [0, 1, 2, 6])
        self.assertEqual(c.top_k, 2048)

    def test_selection_active(self):
        self.assertEqual(self.cover["idx_rows_selecting"], 4 * 32)    # 4 full layers x rows 2048..2079

    def test_matches_llama_cpp_tail(self):
        want = ref("llama-long-2080-tail32.f32", self.m.cfg.n_vocab)
        got = self.logits[:, 2048:]
        self.assertLess(np.abs(got - want).max(), LOGIT_GATE)
        self.assertLess(np.percentile(kl_top_rest(got, want), 99), KL_GATE)


# ============================================================================ quantized files
class QuantizedFiles(unittest.TestCase):
    def test_loader_matches_gguf_py_reader(self):
        """Every tensor of PACK and PACK9 (F32, F16, Q8_0, Q2_K..Q6_K, IQ1_S, IQ2_*, IQ3_*, IQ4_XS)
        decodes exactly as gguf-py's own GGUFReader + dequantize."""
        from glm_dsa_load import _gguf_py
        Q, quants = _gguf_py()
        from gguf import GGUFReader
        seen = set()
        for stem in ("glm-dsa-pack", "glm-dsa-pack9", "glm-dsa-pack-iq4_nl", "glm-dsa-pack-q4_0"):
            r = GGUFReader(_FILES[stem])
            with GgufAny(_FILES[stem]) as g:
                for t in r.tensors:
                    want = np.asarray(quants.dequantize(t.data, t.tensor_type), dtype=np.float32)
                    got = g.tensor(t.name)
                    np.testing.assert_array_equal(got.reshape(-1), want.reshape(-1), t.name)
                    seen.add(t.tensor_type.name)
        self.assertTrue({"F32", "Q8_0", "Q4_0", "IQ4_NL", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ1_S",
                         "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS"} <= seen, sorted(seen))

    def test_lazy_reads(self):
        with GgufAny(_FILES["glm-dsa-tiers"]) as g:
            name = "blk.3.ffn_down_exps.weight"
            full = g.tensor(name)
            for e in (0, 17, 255):
                np.testing.assert_array_equal(g.expert(name, e), full[e])
            emb = g.tensor("token_embd.weight")
            np.testing.assert_array_equal(g.rows("token_embd.weight", [5, 1000, 0]), emb[[5, 1000, 0]])

    def test_kb_type_variants_encode_the_same_weights(self):
        """PACK's attn_k_b in Q8_0, Q4_0, IQ4_NL and F16 come from the same float weights."""
        name = "blk.1.attn_k_b.weight"
        with GgufAny(_FILES["glm-dsa-pack-f16"]) as g:
            ref16 = g.tensor(name).astype(np.float64)
            self.assertEqual(g.info(name).type_name, "F16")
            self.assertEqual(g.info(name).shape, [192, 512, 4])
        rms = math.sqrt(float((ref16 ** 2).mean()))
        for stem, kind, tol in (("glm-dsa-pack", "Q8_0", 0.01), ("glm-dsa-pack-q4_0", "Q4_0", 0.15),
                                ("glm-dsa-pack-iq4_nl", "IQ4_NL", 0.12)):
            with GgufAny(_FILES[stem]) as g:
                self.assertEqual(g.info(name).type_name, kind)
                err = math.sqrt(float(((g.tensor(name) - ref16) ** 2).mean())) / rms
                self.assertLess(err, tol, f"{kind}: relative RMS error {err:.3f}")

    def test_pack9_has_nine_expert_classes(self):
        with GgufAny(_FILES["glm-dsa-pack9"]) as g:
            cfg = GlmDsaConfig(g.metadata)
            classes = set()
            for il in range(cfg.dense_lead, cfg.n_layer_all):
                gu = g.info(f"blk.{il}.ffn_gate_exps.weight")
                up = g.info(f"blk.{il}.ffn_up_exps.weight")
                dn = g.info(f"blk.{il}.ffn_down_exps.weight")
                self.assertEqual(gu.type_name, up.type_name)
                classes.add((g.row_bytes(gu), g.row_bytes(dn)))
            self.assertEqual(len(classes), 9)

    def test_oracle_vs_llama_cpp_on_quantized_files(self):
        """llama.cpp quantizes activations (q8_K/q8_0) for every quantized matmul, so the logits move
        by up to about 1 here; the check is that the weight decoding and the graph agree."""
        for stem, refname in (("glm-dsa-pack", "llama-pack-12.f32"), ("glm-dsa-tiers", "llama-tiers-12.f32")):
            with self.subTest(file=stem):
                m = GlmDsa(_FILES[stem], Options(mode="llama"))
                got = m.forward(toks(12, m.cfg.n_vocab))
                want = ref(refname, m.cfg.n_vocab)
                kl = kl_top_rest(got, want)
                self.assertTrue((got.argmax(0) == want.argmax(0)).all())
                self.assertLess(kl.mean(), 1e-2)
                self.assertLess(kl.max(), 0.1)
                m.close()

    def test_iq4_nl_quantizer_roundtrip(self):
        from glm_dsa_load import _gguf_py
        Q, quants = _gguf_py()
        x = synth.Streams(1).normal("iq4nl", 32 * 64).astype(np.float32).reshape(8, 256)
        q = synth.quantize_iq4_nl(x)
        y = np.asarray(quants.dequantize(q, Q.IQ4_NL), dtype=np.float64).reshape(8, 256)
        self.assertLess(math.sqrt(float(((y - x) ** 2).mean())), 0.12)


# ============================================================================ generator
class Generator(unittest.TestCase):
    def test_byte_identical_to_committed_hashes(self):
        want = {}
        for line in (DATA / "sha256.txt").read_text().split("\n"):
            if line.strip():
                h, n = line.split()
                want[n] = h
        self.assertEqual(set(want), {f"{s}.gguf" for s, _, _ in synth.default_set(False)})
        for stem, path in _FILES.items():
            self.assertEqual(synth.sha256(path), want[f"{stem}.gguf"], stem)

    def test_regenerate_identical(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "again.gguf"
            synth.write(p, "SMALL", synth.DEFAULT_SEED)
            self.assertEqual(p.read_bytes(), _FILES["glm-dsa-small"].read_bytes())
            synth.write(p, "SMALL", synth.DEFAULT_SEED + 1)
            self.assertNotEqual(synth.sha256(p), synth.sha256(_FILES["glm-dsa-small"]))

    def test_split_shards_load_as_one(self):
        with tempfile.TemporaryDirectory() as d:
            paths = synth.write(pathlib.Path(d) / "pack.gguf", "PACK", synth.DEFAULT_SEED, split_max_size=6 << 20)
            self.assertGreater(len(paths), 1)
            with GgufAny(paths[0]) as g, GgufAny(_FILES["glm-dsa-pack"]) as one:
                self.assertEqual(sorted(g.names()), sorted(one.names()))
                for n in ("output.weight", "blk.6.ffn_down_exps.weight"):
                    np.testing.assert_array_equal(g.tensor(n), one.tensor(n))

    def test_split_larger_than_the_file_writes_one(self):
        # gguf-py does not split a model smaller than split_max_size: write() returns that one file (was [], and
        # --check / --sha256 then failed with IndexError)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "one.gguf"
            paths = synth.write(p, "SMALL", synth.DEFAULT_SEED, split_max_size=1 << 40)
            self.assertEqual(paths, [p])
            self.assertEqual(p.read_bytes(), _FILES["glm-dsa-small"].read_bytes())

    def test_metadata_matches_llama_cpp_keys(self):
        with GgufAny(_FILES["glm-dsa-tiers"]) as g:
            md = g.metadata
            self.assertEqual(md["general.architecture"], "glm-dsa")
            for k, v in {"block_count": 7, "attention.head_count_kv": 1, "attention.key_length": 576,
                         "attention.value_length": 512, "attention.key_length_mla": 256,
                         "attention.value_length_mla": 256, "rope.dimension_count": 64,
                         "expert_count": 256, "expert_used_count": 8, "expert_gating_func": 2,
                         "nextn_predict_layers": 1, "attention.indexer.top_k": 2048,
                         "context_length": 1048576}.items():
                self.assertEqual(md[f"glm-dsa.{k}"], v, k)
            self.assertEqual(md["glm-dsa.attention.indexer.types"], [True, True, True, False, False, False])
            self.assertTrue(g.has("blk.6.nextn.eh_proj.weight"))
            self.assertTrue(g.has("blk.6.indexer.attn_k.weight"))          # MTP layer is full
            self.assertFalse(g.has("blk.3.indexer.attn_k.weight"))         # shared trunk layer

    def test_streams_are_standardized(self):
        x = synth.Streams(7).normal("probe", 200000)
        self.assertLess(abs(x.mean()), 0.01)
        self.assertLess(abs(x.std() - 1.0), 0.01)


# ============================================================================ transformers (opt-in)
def _have_hf() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers import AutoConfig
        AutoConfig.for_model("glm_moe_dsa")
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_hf(), "needs torch and transformers with glm_moe_dsa (not installed in CI)")
class HFTransformers(unittest.TestCase):
    def test_oracle_matches_transformers_f64(self):
        """B3.1 / A-46: transformers float64 on SMALL agrees within 1e-9 (unverified: never run here)."""
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "strata" / "glm_dsa_hf_xcheck.py"),
                            str(_FILES["glm-dsa-small"]), "--tokens", "24", "--tol", "1e-9"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


# ============================================================================ large presets (opt-in)
@unittest.skipUnless(os.environ.get("MAYA_SYNTH_LARGE") == "1", "set MAYA_SYNTH_LARGE=1 (writes ~3.3 GB)")
class LargePresets(unittest.TestCase):
    def test_big_and_wide(self):
        with tempfile.TemporaryDirectory() as d:
            big = synth.write(pathlib.Path(d) / "big.gguf", "BIG")[0]
            with GgufAny(big) as g:
                self.assertEqual(g.info("blk.40.attn_k_b.weight").shape, [192, 512, 64])
                self.assertEqual(g.info("blk.40.attn_v_b.weight").shape, [512, 256, 64])
                self.assertEqual(len(g.metadata["glm-dsa.expert_feed_forward_length"]), 79)
                self.assertEqual(len(g.metadata["glm-dsa.attention.head_count_kv"]), 79)
                self.assertEqual(len(g.metadata["glm-dsa.attention.indexer.types"]), 78)
            m = GlmDsa(big, Options(mode="hf"))
            self.assertEqual(sum(m.cfg.idx_full), 21)
            lg = m.forward(toks(4, m.cfg.n_vocab))
            self.assertTrue(np.isfinite(lg).all())
            m.close()
            big.unlink()
            wide = synth.write(pathlib.Path(d) / "wide.gguf", "WIDE")[0]
            with GgufAny(wide) as g:
                self.assertEqual(g.info("blk.1.ffn_gate_exps.weight").shape, [6144, 2048, 12])
                self.assertEqual(g.info("blk.1.ffn_gate_exps.weight").type_name, "IQ2_XXS")
            m = GlmDsa(wide, Options(mode="hf"))
            self.assertTrue(np.isfinite(m.forward(toks(3, m.cfg.n_vocab))).all())
            m.close()


if __name__ == "__main__":
    unittest.main()
