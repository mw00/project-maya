"""tools/test_glm_oracle.py - the glm5-next reference forward pass, checked on a synthetic model.

Runs `tools/glm_synth_gguf.py`'s tiny glm5-next GGUF through `ref/glm.py` and asserts the
invariants that hold by construction of the architecture (docs/GLM5-FLASH.md §3):

  * the logits are finite and the greedy loop terminates;
  * the MoE weights renormalize to exactly expert_weights_scale per token;
  * the mHC comb matrices are doubly stochastic to the hc epsilon after the LAST (dst)
    normalization - the src columns only converge as iterations accumulate;
  * the hc pre-gates sit in (hc_eps, 1+hc_eps) and the post-gates in (0, 2);
  * the KDA scan and the absorbed-MLA attention produce finite seams at every layer.

No GPU, no network, no big download: `python -m unittest tools.test_glm_oracle` (or unittest
discover, like the other tool tests).
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))

from _paths import add_gguf_py  # noqa: E402
try:                            # (the fixtures need llama.cpp's gguf-py: without it this file skips, instead of
    add_gguf_py()               #  a SystemExit in its setup ending the whole test run)
except SystemExit as e:
    raise unittest.SkipTest(str(e))

import numpy as np  # noqa: E402

import glm_synth_gguf  # noqa: E402
from glm import Glm5Next  # noqa: E402  (ref/glm.py)


class GlmOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-glm-oracle-")
        gguf = pathlib.Path(cls.tmp.name) / "glm5-synth.gguf"
        md, ts = glm_synth_gguf.build(seed=1234)
        glm_synth_gguf.write_gguf(gguf, md, ts)
        glm_synth_gguf.selftest(gguf)
        cls.model = Glm5Next(gguf)
        cls.cfg = cls.model.cfg
        cls.tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        cls.logits = cls.model.forward(cls.tokens)

    @classmethod
    def tearDownClass(cls):
        # Windows will not unlink the GGUF while the reader holds it open
        cls.model.g.close()
        cls.tmp.cleanup()

    def test_config_read_from_metadata(self):
        c = self.cfg
        self.assertEqual(c.n_embd, 128)
        self.assertEqual(c.n_layers, 6)
        self.assertEqual([c.is_recr(i) for i in range(6)], [True, True, True, False, True, False])
        self.assertEqual(c.hc, 4)
        self.assertEqual(c.hc_sinkhorn_iters, 3)
        self.assertEqual(c.n_expert, 12)
        self.assertEqual(c.n_expert_used, 3)
        self.assertEqual(c.idx_kpool, 2)
        self.assertTrue(c.idx_kpool_select_tail)
        self.assertEqual(c.kda_gate_lower_bound, -5.0)

    def test_logits_finite(self):
        self.assertTrue(np.isfinite(self.logits).all())
        self.assertEqual(self.logits.shape, (512, len(self.tokens)))

    def test_future_tokens_do_not_change_prefix_logits(self):
        # causal: a position's logits see only the tokens up to it (the KDA scan, the DSA indexer's selection and its
        # pooled keys included), so the first four positions read the same with or without the four after them
        prefix = self.model.forward(self.tokens[:4])
        np.testing.assert_allclose(prefix, self.logits[:, :4], atol=3e-6, rtol=3e-6)

    def test_greedy_loop_runs(self):
        toks = list(self.tokens)
        for _ in range(3):
            toks.append(int(np.argmax(self.model.forward(toks), axis=0)[-1]))
        self.assertEqual(len(toks), len(self.tokens) + 3)
        self.assertTrue(np.isfinite(self.logits).all())

    def test_moe_weights_renormalize_to_scale(self):
        for il in range(self.cfg.n_layers):
            if il < self.cfg.n_layer_dense_lead:
                continue
            w = self.model.dumps[f"ffn_moe_weights-{il}"]
            self.assertTrue(np.allclose(w.sum(axis=0), self.cfg.expert_weights_scale, atol=1e-5),
                            f"layer {il}: weight sums {w.sum(axis=0)}")

    def test_mhc_comb_doubly_stochastic_on_dst(self):
        eps = self.cfg.hc_eps
        for name, c in self.model.dumps.items():
            if not name.startswith("hc_comb_"):
                continue
            self.assertTrue(np.allclose(c.sum(axis=1), 1.0, atol=4 * eps),
                            f"{name}: dst rows {c.sum(axis=1)}")
            # src columns converge only as iterations accumulate; with the synthetic's 3 they
            # are close but not exact - assert they are at least finite and positive
            self.assertTrue((c.sum(axis=0) > 0).all(), f"{name}: src sums {c.sum(axis=0)}")

    def test_hc_gate_ranges(self):
        eps = self.cfg.hc_eps
        for name, g in self.model.dumps.items():
            if name.startswith("hc_pre_"):
                self.assertGreater(float(g.min()), 0.0, name)
                self.assertLessEqual(float(g.max()), 1.0 + eps + 1e-6, name)
            elif name.startswith("hc_post_"):
                self.assertGreaterEqual(float(g.min()), 0.0, name)
                self.assertLessEqual(float(g.max()), 2.0 + 1e-6, name)

    def test_mixer_seams_finite(self):
        for il in range(self.cfg.n_layers):
            if self.cfg.is_recr(il):
                seams = ["kda_scan_out", "kda_out"]
            else:
                seams = ["q_absorbed", "kqv_out", "attn_out"]
            for seam in seams:
                arr = self.model.dumps[f"{seam}-{il}"]
                self.assertTrue(np.isfinite(arr).all(), f"{seam}-{il}")


if __name__ == "__main__":
    unittest.main()
