"""tools/test_iq_pack_glm.py - the packer's glm5-next support, as pure-function tests.

The packer (tools/iq_pack.py) is mostly architecture-agnostic - it works off tensor names and
derives the expert geometry from the tensors - but three things are per-architecture and are what
these tests pin:

  * the architecture check (qwen4exp and glm5-next accepted, everything else refused);
  * the BF16-preserve list.  The two architectures read DIFFERENT projections as high-precision:
    qwen4exp's hyper-connections are hc_*_down/up/inject, glm5-next's mHC is hc_*_fn/base/scale,
    and their indexers share no tensor name.  A list copied from one to the other would preserve
    the wrong tensors - plausible weights, the wrong model;
  * the 3-D flattening.  glm5-next's absorbed-MLA projections (attn_k_b, attn_v_b) and conv
    weights are 3-D; the flat index records two dimensions.  The flatten must be the byte-order
    preserving one (ne0*ne1, ne2) - the engine reads packed tensors by byte offset.

CPU only: no model file, no GPU.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import iq_pack  # noqa: E402


class Glm5NextBf16List(unittest.TestCase):
    def test_glm5_mhc_names_preserved(self):
        for name in ("blk.0.hc_attn_fn.weight", "blk.3.hc_attn_base.weight", "blk.3.hc_attn_scale.weight",
                     "blk.3.hc_ffn_fn.weight", "blk.3.hc_ffn_base.weight", "blk.3.hc_ffn_scale.weight"):
            self.assertTrue(iq_pack.needs_bf16(name, "Q3_K", "glm5-next"), name)
            # the qwen4exp bottleneck names are NOT glm5-next tensors
            self.assertFalse(iq_pack.needs_bf16(name, "Q3_K", "qwen4exp"), name)

    def test_glm5_indexer_and_router_preserved(self):
        for name in ("blk.3.indexer.attn_q_b.weight", "blk.3.indexer.attn_k.weight",
                     "blk.3.indexer_compressor_gate.weight", "blk.3.indexer_compressor_ape.weight",
                     "blk.3.indexer.proj.weight", "blk.3.indexer.k_norm.bias",
                     "blk.3.ffn_gate_inp.weight"):
            self.assertTrue(iq_pack.needs_bf16(name, "Q3_K", "glm5-next"), name)

    def test_glm5_selection_bias_stays_f32(self):
        # the noaux_tc selection bias has a large per-layer offset and small within-layer differences: BF16 rounding
        # collapsed 288 values to 6-11 (#861); the pack keeps the stored F32 (kind 2, which the loader reads as is)
        self.assertFalse(iq_pack.needs_bf16("blk.3.exp_probs_b.bias", "F32", "glm5-next"))
        self.assertNotIn("exp_probs_b.bias", iq_pack.GLM5NEXT_BF16)

    def test_glm5_kda_gates_and_conv_preserved(self):
        for name in ("blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight",
                     "blk.0.ssm_beta.weight", "blk.0.ssm_f_a.weight", "blk.0.ssm_f_b.weight",
                     "blk.0.ssm_g_a.weight", "blk.0.ssm_g_b.weight",
                     # the conv weights seed the whole KDA recurrence - noise there accumulates
                     # per token, so they are preserved explicitly (second-opinion item 4)
                     "blk.0.ssm_conv1d_q.weight", "blk.0.ssm_conv1d_k.weight",
                     "blk.0.ssm_conv1d_v.weight"):
            self.assertTrue(iq_pack.needs_bf16(name, "Q3_K", "glm5-next"), name)

    def test_glm5_absorbed_mla_preserved(self):
        # the 3-D absorbed projections form the attention scores and values themselves
        for name in ("blk.3.attn_k_b.weight", "blk.3.attn_v_b.weight"):
            self.assertTrue(iq_pack.needs_bf16(name, "Q3_K", "glm5-next"), name)

    def test_glm5_expert_names_not_preserved(self):
        # the experts ride the native path; the plain qwen4exp-style attention name is not a
        # glm5-next tensor at all
        for name in ("blk.3.ffn_gate_exps.weight", "blk.3.ffn_down_exps.weight",
                     "blk.3.attn_q.weight"):
            self.assertFalse(iq_pack.needs_bf16(name, "IQ3_XXS", "glm5-next"), name)

    def test_glm5_fail_closed_partition(self):
        # EVERY non-expert blk tensor name must be decidable: either BF16-preserved, or allowed
        # native (quantized), or FLOAT-by-type.  A quantized tensor in NEITHER list fails the
        # pack closed (index_standalone errors) - that is the point of the inversion.
        listed = set(iq_pack.GLM5NEXT_BF16)
        native = set(iq_pack.GLM5NEXT_NATIVE_OK)
        self.assertFalse(listed & native, "a tensor on both lists is ambiguous")
        for name in ("attn_norm.weight", "ffn_norm.weight", "output_norm.weight",
                     "hc_attn_fn.weight", "indexer.k_norm.weight"):
            self.assertNotIn(name, native, name)
        # the native set covers exactly what dynamic recipes quantize and the engine may read
        # quantized: the attention projections and the dense/shared FFN
        for name in ("attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight",
                     "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                     "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
            self.assertIn(name, native, name)

    def test_qwen4exp_list_unchanged(self):
        # the qwen4exp route must be exactly what it was (the ORCA compatibility depends on it)
        self.assertTrue(iq_pack.needs_bf16("blk.0.hc_attn_down.weight", "IQ3_XXS"))
        self.assertTrue(iq_pack.needs_bf16("blk.0.ssm_alpha.weight", "IQ3_XXS"))
        self.assertTrue(iq_pack.needs_bf16("blk.0.indexer.q_proj.weight", "IQ3_XXS"))
        self.assertTrue(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q4_K"))
        self.assertFalse(iq_pack.needs_bf16("blk.0.hc_attn_fn.weight", "IQ3_XXS"))
        self.assertFalse(iq_pack.needs_bf16("blk.0.ssm_dt.bias", "IQ3_XXS"))

    def test_non_block_tensors(self):
        self.assertTrue(iq_pack.needs_bf16("output_hc_down.weight", "Q3_K"))
        self.assertFalse(iq_pack.needs_bf16("token_embd.weight", "Q3_K", "glm5-next"))


class IndexShape(unittest.TestCase):
    def test_2d_passthrough(self):
        self.assertEqual(iq_pack.index_shape((128, 256)), (128, 256))

    def test_1d(self):
        self.assertEqual(iq_pack.index_shape((64,)), (64, 0))

    def test_3d_mla_projection(self):
        # attn_k_b {qk_nope, kv_lora, n_head}: rows = qk_nope*kv_lora, one row per head
        self.assertEqual(iq_pack.index_shape((48, 64, 8)), (48 * 64, 8))

    def test_3d_conv_weight(self):
        # ssm_conv1d_q {d_conv, 1, d_inner}: rows = tap, cols = channel - the runner reads
        # conv_w[tap + d_conv*channel], i.e. element (r, c) at r + rows*c
        self.assertEqual(iq_pack.index_shape((4, 1, 256)), (4, 256))

    def test_4d_conv_weight_trailing_one(self):
        # some converters write the conv weight as {d_conv, 1, d_inner, 1}: the trailing 1 adds no
        # bytes, so it packs exactly like the 3-D form
        self.assertEqual(iq_pack.index_shape((4, 1, 256, 1)), iq_pack.index_shape((4, 1, 256)))

    def test_flatten_is_byte_preserving(self):
        # the pack's 2-D convention is GGML: element (i0, i1) at i0 + ne0*i1 (ne0 fastest).  A 3-D
        # tensor stored as the (ne0*ne1, ne2) view preserves every byte offset exactly when
        # (i0, i1, i2) at i0 + ne0*i1 + ne0*ne1*i2 == r + R*c with r = i0 + ne0*i1, c = i2.
        import numpy as np
        ne0, ne1, ne2 = 4, 3, 2
        rows, cols = iq_pack.index_shape((ne0, ne1, ne2))
        self.assertEqual((rows, cols), (ne0 * ne1, ne2))
        raw = np.arange(ne0 * ne1 * ne2, dtype=np.int32).reshape(ne0, ne1, ne2)
        ggml = raw.flatten(order="F")                    # the GGUF's byte order, ne0 fastest
        for i0 in range(ne0):
            for i1 in range(ne1):
                for i2 in range(ne2):
                    self.assertEqual(ggml[i0 + ne0 * i1 + ne0 * ne1 * i2], raw[i0, i1, i2])
                    self.assertEqual(ggml[(i0 + ne0 * i1) + rows * i2], raw[i0, i1, i2])

class ArchitectureCheck(unittest.TestCase):
    def test_known_architectures(self):
        self.assertIn("qwen4exp", iq_pack.ARCHITECTURES)
        self.assertIn("glm5-next", iq_pack.ARCHITECTURES)
        self.assertIn("glm5next", iq_pack.ARCHITECTURES)   # unsloth's spelling, same KV family
        self.assertEqual(len(iq_pack.ARCHITECTURES), 4)   # + glm-dsa (glm-dsa B1.6)


if __name__ == "__main__":
    unittest.main()
