"""tools/strata/test_glm_dsa_pack_native32.py - glm-dsa B10.7: the glm-dsa pack's STRATA_PACK_NATIVE32 planning rule.

Default (flag off): every non-routed tensor whose type mv() lacked is upcast to BF16, attn_k_b / attn_v_b always
become BF16 (unchanged B1.6 behaviour).  With the flag: the 32-value block types (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL)
and F16 of the mv tensors stay native, Q8_0 attn_k_b / attn_v_b stay native (glmd::headwise), other k_b / v_b types
and the indexer projections are still converted.

    PYTHONPATH=<gguf-py> python3 -m unittest discover -s tools/strata -p 'test_glm_dsa_pack_native32.py' -v
"""
from __future__ import annotations

import os
import pathlib
import sys
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import glm_dsa_pack as P  # noqa: E402


class Native32(unittest.TestCase):
    def test_default_upcasts(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_PACK_NATIVE32", None)
            for t in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "F16"):
                self.assertEqual(P.plan_tensor("blk.5.attn_q_b.weight", t), ("convert", "BF16"), t)
            self.assertEqual(P.plan_tensor("blk.5.attn_k_b.weight", "Q8_0"), ("convert", "BF16"))
            self.assertEqual(P.plan_tensor("blk.5.attn_v_b.weight", "Q8_0"), ("convert", "BF16"))
            self.assertEqual(P.plan_tensor("blk.5.attn_q_b.weight", "Q8_0"), ("native", "Q8_0"))

    def test_native32(self):
        with mock.patch.dict(os.environ, {"STRATA_PACK_NATIVE32": "1"}):
            self.assertTrue(P.native32_enabled())
            for t in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "F16"):
                self.assertEqual(P.plan_tensor("blk.5.attn_q_b.weight", t), ("native", t), t)
                self.assertEqual(P.plan_tensor("blk.1.ffn_down.weight", t), ("native", t), t)
            self.assertEqual(P.plan_tensor("blk.5.attn_k_b.weight", "Q8_0"), ("native", "Q8_0"))
            self.assertEqual(P.plan_tensor("blk.78.attn_v_b.weight", "Q8_0"), ("native", "Q8_0"))
            # the headwise GEMV reads Q8_0 or BF16 only; the indexer stays float
            self.assertEqual(P.plan_tensor("blk.5.attn_k_b.weight", "Q4_0"), ("convert", "BF16"))
            self.assertEqual(P.plan_tensor("blk.0.indexer.attn_k.weight", "Q8_0"), ("convert", "BF16"))
            self.assertEqual(P.plan_tensor("blk.5.attn_k_b.weight", "BF16"), ("copy", "BF16"))
            self.assertEqual(P.plan_tensor("blk.5.attn_q_b.weight", "BF16"), ("copy", "BF16"))

    def test_explicit_argument_wins(self):
        with mock.patch.dict(os.environ, {"STRATA_PACK_NATIVE32": "1"}):
            self.assertEqual(P.plan_tensor("blk.5.attn_q_b.weight", "Q4_0", native32=False), ("convert", "BF16"))
        os.environ.pop("STRATA_PACK_NATIVE32", None)
        self.assertEqual(P.plan_tensor("blk.5.attn_q_b.weight", "Q4_0", native32=True), ("native", "Q4_0"))


if __name__ == "__main__":
    unittest.main()
