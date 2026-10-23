"""tools/test_exl3_pack.py - the packer's plan for the experts-only community quants (no GPU, no network, no files):

unquantized matrices (BF16/F16/F32 sources) become BF16 float rows, the unfused q/k/v projections emit as three
rows, and an unquantized eh_proj packs the NextN (MTP) block - its floats marked for the pack GGUF, its experts
in exl3.txt.  The plan is everything write_all consumes: the float entries (name, gguf shape, type, producer),
the EXL3 mats, the per-layer expert tables and the notes.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_pack  # noqa: E402

L = "model.language_model."


class FakeST:
    """Duck-typed SafeTensors: name -> (dtype, shape); f32() returns zeros shaped as stored."""

    def __init__(self, tensors):
        self.t = {n: ("model-00001.safetensors", dt, list(shp), 0, 0) for n, (dt, shp) in tensors.items()}
        self.files = ["model-00001.safetensors"]

    def has(self, name):
        return name in self.t

    def f32(self, name):
        _, dt, shp, _, _ = self.t[name]
        if dt not in ("F32", "F16", "BF16"):
            raise ValueError("%s is %s, not a float tensor" % (name, dt))
        return np.zeros(shp, dtype=np.float32)


def trellis(k, n, K=1):
    """An EXL3 trellis shape for a k x n matrix at K bits ((k/16, n/16, 16K), plus F16 suh/svh)."""
    return ("F32", (k // 16, n // 16, 16 * K)), ("F16", (k,)), ("F16", (n,))


def source(unfused_qkv=False, nextn_eh="BF16"):
    """A tiny glm5_next EXL3 directory: layer 0 dense (MLA), 1 MoE (fused-KDA), 2 MoE (MLA, experts-only
    layout: BF16 q_a / q_b / o_proj, F32 kv_a), 3 MoE (KDA, unfused q/k/v when unfused_qkv), + NextN."""
    H, h, ql, kl = 64, 64, 32, 32
    nope, rope, vhd, nh = 16, 8, 16, 4
    di, kda_hd = 64, 16
    moe, n_exp = 32, 4
    t = {}

    def f32(n, shp):
        t[n] = ("F32", shp)

    def bf16(n, shp):
        t[n] = ("BF16", shp)

    def exl3(n, k, n_, K=1):
        tr, suh, svh = trellis(k, n_, K)
        t[n + ".trellis"], t[n + ".suh"], t[n + ".svh"] = tr, suh, svh

    # the head and the embedding-side norms
    bf16("lm_head.weight", (256, 64))   # experts-only quants keep the head unquantized (head_bits 16)
    f32(L + "norm.weight", (h,))
    for il in range(4):
        P, A, M = "blk.%d." % il, L + "layers.%d.self_attn." % il, L + "layers.%d.mlp." % il
        f32(A[:-10] + "input_layernorm.weight", (h,))
        f32(A[:-10] + "post_attention_layernorm.weight", (h,))
        if il < 4:
            bf16(A[:-10] + "hc_attn_fn", (256, 8))
            f32(A[:-10] + "hc_attn_base", (8,))
            f32(A[:-10] + "hc_attn_scale", (8,))
            bf16(A[:-10] + "hc_ffn_fn", (256, 8))
            f32(A[:-10] + "hc_ffn_base", (8,))
            f32(A[:-10] + "hc_ffn_scale", (8,))
        if il in (0, 2):   # the MLA layers
            if il == 2:    # the experts-only layout: unquantized projections
                bf16(A + "q_a_proj.weight", (ql, h))
                bf16(A + "q_b_proj.weight", (nh * (nope + rope + vhd), ql))
                f32(A + "kv_a_proj_with_mqa.weight", (kl + rope, h))
                bf16(A + "o_proj.weight", (h, nh * vhd))
            else:          # layer 0 keeps its projections EXL3
                exl3(A + "q_a_proj", ql, h)
                exl3(A + "q_b_proj", nh * (nope + rope + vhd), ql)
                exl3(A + "kv_a_proj_with_mqa", kl + rope, h)
                exl3(A + "o_proj", h, nh * vhd)
            bf16(A + "indexer.wq_b.weight", (128, ql))
            f32(A + "q_a_layernorm.weight", (ql,))
            f32(A + "kv_a_layernorm.weight", (kl,))
            bf16(A + "kv_b_proj.weight", (nh * (nope + vhd), kl))
            bf16(A + "indexer.wk.weight", (128, h))
            bf16(A + "indexer.weights_proj.weight", (32, 128))
            f32(A + "indexer.k_norm.weight", (128,))
            f32(A + "indexer.k_norm.bias", (128,))
            bf16(A + "indexer.index_kpool_compress_gate", (8,))
            f32(A + "indexer.index_kpool_compress_ape", (8,))
        else:              # the KDA layers: fused qkv_proj (layer 1) or the unfused trio (layer 3)
            if il == 1 or not unfused_qkv:
                exl3(A + "qkv_proj", h, 3 * di)
            else:
                bf16(A + "q_proj.weight", (di, h))
                bf16(A + "k_proj.weight", (di, h))
                bf16(A + "v_proj.weight", (di, h))
            bf16(A + "o_proj.weight", (h, di))
            f32(A + "A_log", (8,))
            f32(A + "dt_bias", (8,))
            bf16(A + "b_proj.weight", (8, di))
            for nm in ("f_a", "f_b", "g_a", "g_b"):
                bf16(A + "%s_proj.weight" % nm, (di, di))
            f32(A + "o_norm.weight", (kda_hd,))
            f32(A + "conv1d.weight", (3, di, 4))
        if il < 1:         # the dense lead layer
            for r in ("gate", "up", "down"):
                exl3(M + "%s_proj" % r, h if r != "down" else 128, 128 if r != "down" else h)
        else:              # the MoE layers (and the NextN block's)
            bf16(M + "gate.weight", (n_exp, h))
            f32(M + "gate.e_score_correction_bias", (n_exp,))
            for r in ("gate", "up", "down"):
                if il == 1:   # layer 1 keeps its shared experts EXL3 (Maya's own quants quantize them)
                    exl3(M + "shared_experts.%s_proj" % r, h if r != "down" else moe, moe if r != "down" else h)
                else:
                    bf16(M + "shared_experts.%s_proj.weight" % r, (moe, h)) if r != "down" else \
                        bf16(M + "shared_experts.down_proj.weight", (h, moe))
            for e in range(n_exp):
                for r, k, n_ in (("gate", h, moe), ("up", h, moe), ("down", moe, h)):
                    exl3(M + "experts.%d.%s_proj" % (e, r), k, n_)
    f32(L + "layers.4.enorm.weight", (h,))
    f32(L + "layers.4.hnorm.weight", (h,))
    f32(L + "layers.4.shared_head.norm.weight", (h,))
    A4 = L + "layers.4.self_attn."   # the NextN block runs the MLA branch: experts-only layout there too
    bf16(A4 + "q_a_proj.weight", (ql, h))
    bf16(A4 + "q_b_proj.weight", (nh * (nope + rope + vhd), ql))
    bf16(A4 + "kv_a_proj_with_mqa.weight", (kl + rope, h))
    bf16(A4 + "o_proj.weight", (h, nh * vhd))
    bf16(A4 + "indexer.wq_b.weight", (128, ql))
    f32(A4 + "q_a_layernorm.weight", (ql,))
    f32(A4 + "kv_a_layernorm.weight", (kl,))
    bf16(A4 + "kv_b_proj.weight", (nh * (nope + vhd), kl))
    bf16(A4 + "indexer.wk.weight", (128, h))
    bf16(A4 + "indexer.weights_proj.weight", (32, 128))
    f32(A4 + "indexer.k_norm.weight", (128,))
    f32(A4 + "indexer.k_norm.bias", (128,))
    bf16(A4 + "indexer.index_kpool_compress_gate", (8,))
    f32(A4 + "indexer.index_kpool_compress_ape", (8,))
    M4 = L + "layers.4.mlp."         # ... and the MoE branch
    bf16(M4 + "gate.weight", (n_exp, h))
    f32(M4 + "gate.e_score_correction_bias", (n_exp,))
    for r in ("gate", "up", "down"):
        bf16(M4 + "shared_experts.%s_proj.weight" % r, (moe, h)) if r != "down" else \
            bf16(M4 + "shared_experts.down_proj.weight", (h, moe))
    for e in range(n_exp):
        for r, k, n_ in (("gate", h, moe), ("up", h, moe), ("down", moe, h)):
            exl3(M4 + "experts.%d.%s_proj" % (e, r), k, n_)
    t[L + "layers.4.eh_proj.weight"] = (nextn_eh, (h, 2 * h))
    return FakeST(t)


CFG = {
    "num_hidden_layers": 4, "num_attention_heads": nh_ if (nh_ := 4) else 4, "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8, "v_head_dim": 16, "q_lora_rank": 32, "kv_lora_rank": 32,
    "linear_attn_config": {"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4},
    "first_k_dense_replace": 1, "layer_types": ["full", "linear_attention", "full", "linear_attention"],
    "n_routed_experts": 4, "hidden_size": 64, "vocab_size": 256, "moe_intermediate_size": 32,
    "intermediate_size": 128, "num_nextn_predict_layers": 1,
}


def named(plan, gname):
    return [f for f in plan.floats if f[0] == gname]


def mat_named(plan, gname):
    return [m for m in plan.mats if m[0] == gname]


class UnquantizedMatrices(unittest.TestCase):
    def test_bf16_and_f32_sources_pack_as_bf16_float_rows(self):
        plan = exl3_pack.Plan(source(), CFG)
        plan.build(mtp=True)
        for gname, stored in (("blk.2.attn_q_a.weight", (32, 64)), ("blk.2.attn_q_b.weight", (160, 32)),
                              ("blk.2.attn_output.weight", (64, 64)), ("output.weight", (256, 64))):
            rows = named(plan, gname)
            self.assertEqual(len(rows), 1, gname)
            self.assertEqual(rows[0][2], "BF16", gname)          # a BF16 float row (index kind 4)
            self.assertEqual(rows[0][1], list(reversed(stored)), gname)
            self.assertEqual(mat_named(plan, gname), [], gname)  # not an EXL3 mat
        # an F32 source converts at pack time, the same way as BF16 ones (it used to write an F32 row,
        # which the engine's q() rejects as a type-0 slot)
        rows = named(plan, "blk.2.attn_kv_a_mqa.weight")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "BF16")
        self.assertTrue(any("stored unquantized (F32 in the source)" in n for n in plan.notes))

    def test_a_non_float_source_is_an_error(self):
        plan = exl3_pack.Plan(source(), CFG)
        st = plan.st
        st.t["model.language_model.layers.2.self_attn.q_a_proj.weight"] = \
            ("model-00001.safetensors", "Q6_K", [32, 64], 0, 0)
        with self.assertRaises(SystemExit):
            plan.build(mtp=False)

    def test_a_missing_matrix_is_an_error(self):
        plan = exl3_pack.Plan(source(), CFG)
        del plan.st.t["model.language_model.layers.2.self_attn.q_a_proj.weight"]
        with self.assertRaises(SystemExit):
            plan.build(mtp=False)


class SeparateQKV(unittest.TestCase):
    def test_unfused_projections_emit_three_float_rows(self):
        plan = exl3_pack.Plan(source(unfused_qkv=True), CFG)
        plan.build(mtp=False)
        for q in "qkv":
            rows = named(plan, "blk.3.attn_%s.weight" % q)
            self.assertEqual(len(rows), 1, q)
            self.assertEqual(rows[0][2], "BF16", q)
        self.assertEqual(mat_named(plan, "blk.3.attn_qkv.weight"), [])

    def test_fused_projections_stay_one_exl3_mat(self):
        plan = exl3_pack.Plan(source(), CFG)
        plan.build(mtp=False)
        self.assertEqual(len(mat_named(plan, "blk.1.attn_qkv.weight")), 1)


class NextNBlock(unittest.TestCase):
    def test_an_unquantized_eh_proj_packs_the_block(self):
        st = source()
        # main()'s gate: the NextN block is packed when eh_proj is an EXL3 mat OR carried unquantized
        self.assertTrue(st.has(L + "layers.4.eh_proj.trellis") or st.has(L + "layers.4.eh_proj.weight"))
        plan = exl3_pack.Plan(st, CFG)
        plan.build(mtp=True)
        # the block's floats go into the pack GGUF only (":gguf"), not index.txt / dense.bin
        block = [f for f in plan.floats if f[0].startswith("blk.4.")]
        self.assertTrue(block)
        for name, _, ty, _ in block:
            self.assertTrue(ty.endswith(":gguf"), name)
        # its projections and head are BF16 float rows (the engine quantizes the draft's q()-read ones
        # to Q8_0 at load), its experts stay in exl3.txt, and it has no EXL3 matrices
        for gname in ("blk.4.nextn.eh_proj.weight", "blk.4.attn_q_a.weight", "blk.4.attn_kv_a_mqa.weight",
                      "blk.4.attn_output.weight", "blk.4.ffn_gate_shexp.weight"):
            rows = named(plan, gname)
            self.assertTrue(rows, gname)
            self.assertEqual(rows[0][2], "BF16:gguf", gname)
        self.assertEqual([m for m in plan.mats if m[0].startswith("blk.4.")], [])
        self.assertEqual(plan.layers[-1]["layer"], 4)

    def test_no_mtp_leaves_the_block_out(self):
        plan = exl3_pack.Plan(source(), CFG)
        plan.build(mtp=False)
        self.assertEqual([f for f in plan.floats if f[0].startswith("blk.4.")], [])
        self.assertEqual([m for m in plan.mats if m[0].startswith("blk.4.")], [])


class Exl3Mats(unittest.TestCase):
    def test_exl3_matrices_keep_their_shapes(self):
        plan = exl3_pack.Plan(source(), CFG)
        plan.build(mtp=False)
        g = mat_named(plan, "blk.1.ffn_gate_shexp.weight")[0]
        self.assertEqual((g[2], g[3], g[4]), (64, 32, 1))    # k, n, K
        q = mat_named(plan, "blk.1.attn_qkv.weight")[0]
        self.assertEqual((q[2], q[3]), (64, 192))            # the fused KDA projection


if __name__ == "__main__":
    unittest.main()
