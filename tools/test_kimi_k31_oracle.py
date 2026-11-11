"""tools/test_kimi_k31_oracle.py - the Kimi K3.1 Phase 1 gate: ref/kimi_k31.py against an independent model.

The independent model is Moonshot's own Kimi K3 modeling code run with the real fla KDA kernels on
CUDA, plus a lightning-indexer patch (tools/kimi_k31_moonshot_oracle.py). Its fp32 outputs on the
synthetic fixture are committed in data/kimi-k31-synth-golden/; tools/kimi_k31_gate.py holds the
comparison and the measured tolerances (its docstring has the numbers and why each is safe).

  1. CPU, numpy only, CI-safe (no GPU, no torch, no Moonshot code):
       * the fixture regenerates byte-identically (sha256 of all five files = the ones the golden was
         made from), so the committed golden and this checkout describe the same weights;
       * the float64 and the float32 reference pass the gate: full-vocab logits, the layer-3 MLA seam,
         the layer-0 KDA seam, indexer scores, expert ids exactly equal on every row, indexer selections
         equal as sets;
       * the golden is tie-free (every indexer and router cut margin >= 10x the score tolerance), so how
         a reference breaks ties cannot change the verdict: a reference that breaks indexer ties toward
         the LATER key passes too;
       * the gate tells Moonshot's q_a/kv_a eps (1e-6) from the engines' (1e-5) in both directions, and
         fla's KDA q/k l2 eps (1e-6) from llama.cpp's (1e-5);
       * ties, explicitly: indexer inputs built with exact score ties at the cut, where implementations
         that break ties differently select different keys and must still agree as sets;
       * the fixture actually exercises what the negative tests rely on (indexer discards, negative
         visible scores, a router bias that changes the selection, block pushes).
  2. With torch (skipped without): the golden's indexer inputs through vLLM PR #65685's own reference
     score function and SGLang's torch fallback semantics, against the golden's and the reference's
     scores - the indexer half of the independent model is a transcription, this pins it to the engines.
  3. Live (skipped unless KIMI_K31_MOONSHOT_DIR names a directory with Moonshot's modeling_kimi_linear.py
     and configuration_kimi_k3.py, and fla + a CUDA device are available): runs
     tools/kimi_k31_moonshot_oracle.py on the regenerated fixture (both eps), gates the reference
     against that live output, and checks the live output against the committed golden.
     KIMI_K31_GPU_MEM_GB (default 1.0) caps the oracle's CUDA allocator.
  4. Negative tests, in the repository's negative-test discipline (docs/GLM5-FLASH.md, Phase 2): every
     mutation below is a minimal source edit of ref/kimi_k31.py (or, for A_log, of the fixture's
     converter), applied to a fresh copy of the module; each must make the gate FAIL. A mutation whose
     text no longer occurs in the source errors out, so the list cannot silently go stale. The table
     of what each mutation moved is printed at the end.

    python -m unittest tools.test_kimi_k31_oracle -v        (or pytest tools/test_kimi_k31_oracle.py)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ref"))

import numpy as np  # noqa: E402

import kimi_k31_gate as G  # noqa: E402
import kimi_k31_synth_gguf as synth  # noqa: E402

REF_SRC = REPO / "ref" / "kimi_k31.py"
SYNTH_SRC = HERE / "kimi_k31_synth_gguf.py"
FIXTURE_FILES = ("tiny.gguf", "hf-f32/config.json", "hf-f32/model.safetensors",
                 "hf-mxfp4/config.json", "hf-mxfp4/model.safetensors")


def load_mutant(src: pathlib.Path, edits, name: str) -> types.ModuleType:
    """A fresh copy of the module at `src` with each (old, new) text edit applied; `old` must occur
    exactly once. A tuple ("append", text) adds code at the end."""
    text = src.read_text()
    for old, new in edits:
        if old == "append":
            text += "\n" + new
            continue
        n = text.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: the text to replace occurs {n} times in {src.name} "
                                 f"(expected once) - update the mutation to the current source:\n{old}")
        text = text.replace(old, new)
    mod = types.ModuleType(f"_{src.stem}_mutant_{name}")
    mod.__file__ = str(src)
    exec(compile(text, f"{src} [mutant {name}]", "exec"), mod.__dict__)
    return mod


# A NeoX rotary embedding the "MLA given RoPE" mutant applies to q_pe and k_pe (K3.1 has none).
_ROPE = '''
def _mutant_rope(x, pos, theta=10000.0):
    d = x.shape[0]
    half = d // 2
    inv = theta ** (-np.arange(half, dtype=F64) * 2.0 / d)
    ang = inv[:, None] * np.asarray(pos, dtype=F64)[None, :]
    shape = (half,) + (1,) * (x.ndim - 2) + (len(pos),)
    c, s = np.cos(ang).reshape(shape), np.sin(ang).reshape(shape)
    x1, x2 = x[:half].astype(F64), x[half:].astype(F64)
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=0).astype(x.dtype)
'''

# name -> (what it reads as, target, edits, eps_overrides). Target "ref" edits ref/kimi_k31.py; "synth"
# edits the fixture's converter (tools/kimi_k31_synth_gguf.py to_gguf), whose GGUF the unmodified
# reference then reads.
MUTATIONS = {
    "mla_eps_1e-5": (
        "q_a/kv_a RMSNorm eps 1e-5 (the engines') instead of Moonshot's 1e-6", "ref", [], {"mla_latent": 1e-5}),
    "a_log_tail_slice": (
        "converter slices A_log's last num_heads entries (the zero padding) instead of the first", "synth",
        [('(b + "ssm_a", neg_exp(vals[s + "A_log"][:H])),', '(b + "ssm_a", neg_exp(vals[s + "A_log"][-H:])),')],
        None),
    "situ_beta_swapped": (
        "SiTU beta and linear beta read from each other's key", "ref",
        [('        self.situ_beta = float(u("activation.situ_beta", 4.0))\n'
          '        self.situ_linear_beta = float(u("activation.situ_linear_beta", 25.0))',
          '        self.situ_beta = float(u("activation.situ_linear_beta", 25.0))\n'
          '        self.situ_linear_beta = float(u("activation.situ_beta", 4.0))')], None),
    "attn_res_block_one_early": (
        "attention-residual blocks pushed one layer early (2, 5, 8: the 1-indexed rule read 0-indexed)", "ref",
        [("            banked = il % c.attn_res_block_size == 0",
          "            banked = (il + 1) % c.attn_res_block_size == 0")], None),
    "attn_res_block_one_late": (
        "attention-residual blocks pushed one layer late (1, 4, 7)", "ref",
        [("            banked = il % c.attn_res_block_size == 0",
          "            banked = (il - 1) % c.attn_res_block_size == 0")], None),
    "indexer_topk_plus_1": (
        "indexer keeps top_k + 1 keys", "ref",
        [("        k = c.idx_top_k\n", "        k = c.idx_top_k + 1\n")], None),
    "indexer_topk_minus_1": (
        "indexer keeps top_k - 1 keys", "ref",
        [("        k = c.idx_top_k\n", "        k = c.idx_top_k - 1\n")], None),
    "indexer_future_zero_fill": (
        "future keys filled with 0 instead of -inf and the top-k taken over the whole row (SGLang's "
        "_fp8_mqa_logits_torch fill without its length-limited topk_transform)", "ref",
        [("        score[future] = -np.inf ", "        score[future] = 0.0 "),
         ("            vis = score[:p + 1, t]", "            vis = score[:, t]")], None),
    "indexer_future_zero_fill_only": (
        "future keys filled with 0 but the top-k still limited to the visible keys: no output changes, only "
        "the score seam's -inf pattern", "ref",
        [("        score[future] = -np.inf ", "        score[future] = 0.0 ")], None),
    "router_bias_dropped": (
        "router picks by the unbiased sigmoid (e_score_correction_bias dropped)", "ref",
        [('        biased = (probs + self.layer_t(il, "exp_probs_b", "bias")[:, None]).astype(dt)',
          '        biased = probs.copy()')], None),
    "moe_topk_truncated_8": (
        "routed experts truncated to 8 per token (an engine's k <= 8 cap)", "ref",
        [("        K, T = c.n_expert_used, x.shape[1]", "        K, T = min(c.n_expert_used, 8), x.shape[1]")],
        None),
    "latent_norm_before_sum": (
        "routed_norm applied to each expert's output before the weighted sum, not after it", "ref",
        [('                    part[i, :, t - c0] = wd @ self.q8_in(names[2], h)',
          '                    part[i, :, t - c0] = rms_norm(wd @ self.q8_in(names[2], h), '
          'self.layer_t(il, "ffn_routed_norm"), c.eps_rms, dt)'),
         ('        r = rms_norm(moe, self.layer_t(il, "ffn_routed_norm"), c.eps_rms, dt)      # AFTER the weighted sum',
          '        r = moe')], None),
    "mla_rope": (
        "MLA given RoPE: NeoX rotation (theta 1e4) of q_pe and k_pe by position", "ref",
        [("        q_nope, q_pe = q[:c.qk_nope], q[c.qk_nope:]",
          "        q_nope, q_pe = q[:c.qk_nope], _mutant_rope(q[c.qk_nope:], pos)"),
         ("        k_pe = np.ascontiguousarray(kv_pe[c.kv_lora_rank:])                       # raw, never rotated",
          "        k_pe = _mutant_rope(np.ascontiguousarray(kv_pe[c.kv_lora_rank:]), pos)"),
         ("append", _ROPE)], None),
    # beyond the list above, the classic traps the GLM-5 harnesses pin
    "router_weights_biased": (
        "expert weights gathered from the biased score (the noaux_tc trap)", "ref",
        [("            w = probs[sel, t].astype(F64)", "            w = biased[sel, t].astype(F64)")], None),
    "kda_softplus_gate": (
        "KDA decay in Kimi-Linear's softplus form -exp(A_log) * softplus(g + dt_bias)", "ref",
        [("        g1 = (c.kda_gate_lower_bound * sigmoid(arg, dt).astype(F64)).astype(dt)",
          "        g1 = (-np.logaddexp(0.0, split_heads(pre, dk, H).astype(F64)) * exp_a[None, :, None]).astype(dt)")],
        None),
    "indexer_no_relu": (
        "indexer score without the ReLU on q.k", "ref",
        [("            score[:, t0:t1] = (np.maximum(kq, 0.0) * w[None, :, t0:t1]).sum(axis=1)",
          "            score[:, t0:t1] = (kq * w[None, :, t0:t1]).sum(axis=1)")], None),
    "indexer_weights_unscaled": (
        "indexer head weights without the (n_head * head_dim)^-0.5 scale (selection-invariant: only the "
        "score seam can see it)", "ref",
        [("        w = iw.astype(F64) * (Hi * Di) ** -0.5", "        w = iw.astype(F64)")], None),
    "kda_l2_eps_1e-5": (
        "KDA q/k l2 eps 1e-5 (llama.cpp's rms eps) instead of fla's 1e-6", "ref", [], {"kda_l2": 1e-5}),
    "router_bias_dropped_margins_squeezed": (
        "router bias dropped AND the reference's recorded router scores squeezed 1e4x, so its own cut "
        "margins all look like near-ties (judging near-ties on the golden's margin alone catches it)", "ref",
        [('        biased = (probs + self.layer_t(il, "exp_probs_b", "bias")[:, None]).astype(dt)',
          '        biased = probs.copy()'),
         ('        self.cb("ffn_moe_probs_biased", biased, il)',
          '        self.cb("ffn_moe_probs_biased", (biased * 1e-4).astype(dt), il)')], None),
}

# Changes that must NOT fail the gate: the golden is tie-free, so tie-breaking is not part of the contract.
POSITIVE = {
    "indexer_ties_to_later_key": (
        "indexer breaks exact score ties toward the LATER key (the engines leave it unspecified)", "ref",
        [("            idx = top_k_desc(vis, k)",
          "            idx = len(vis) - 1 - top_k_desc(vis[::-1].copy(), k)")], None),
    "router_ties_to_later_expert": (
        "router breaks exact score ties toward the LATER expert", "ref",
        [("            sel = top_k_desc(biased[:, t], K)",
          "            sel = len(biased) - 1 - top_k_desc(biased[::-1, t].copy(), K)")], None),
}

# A converter that slices A_log by a head count other than linear_attn_config.num_heads. KDA heads must
# equal the MLA heads in this format (one attention.head_count), so slicing by num_attention_heads is the
# same slice and cannot be expressed; any other count (here the stored length, head_dim) writes an ssm_a of
# the wrong length, which the reference must refuse rather than broadcast.
A_LOG_WRONG_COUNT = [('(b + "ssm_a", neg_exp(vals[s + "A_log"][:H])),',
                      '(b + "ssm_a", neg_exp(vals[s + "A_log"][:g["kda_head_dim"]])),')]


class KimiK31Gate(unittest.TestCase):
    """CPU, numpy only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-gate-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.fix = pathlib.Path(cls.tmp.name) / "fixture"
        cls.manifest = synth.build(cls.fix)
        cls.golden = G.load_golden()
        cls.meta = cls.golden[0]
        cls.ref_mod = load_mutant(REF_SRC, [], "baseline")       # a fresh copy: its own COVER counters
        cls.models = {}

    @classmethod
    def tearDownClass(cls):
        for m in cls.models.values():
            m.close()                                 # Windows will not unlink an open GGUF

    def model(self, dtype="float64", **kw):
        key = (dtype, tuple(sorted((kw.get("eps_overrides") or {}).items())))
        if key not in self.models:
            self.models[key] = self.ref_mod.KimiK31(self.fix / "tiny.gguf", dtype=dtype, **kw)
        return self.models[key]

    # ---- 1. the fixture is the one the golden was made from
    def test_fixture_regenerates_byte_identically(self):
        want = self.meta["fixture"]["manifest_files"]
        for f in FIXTURE_FILES:
            self.assertEqual(G.sha256(self.fix / f), want[f]["sha256"],
                             f"{f}: the regenerated fixture differs from the one the golden was made from "
                             "(tools/kimi_k31_synth_gguf.py or numpy's generator changed): regenerate the golden")
            self.assertEqual(self.manifest["files"][f]["sha256"], want[f]["sha256"])
        self.assertEqual(G.sha256(self.fix / "hf-f32" / "model.safetensors"), self.meta["fixture"]["hf_f32_model_sha256"])
        self.assertEqual(self.manifest["test_tokens"], self.meta["sequences"][0]["tokens"])
        self.assertEqual([(s["name"], s["tokens"], s["decode_from"]) for s in self.manifest["test_sequences"]],
                         [(s["name"], s["tokens"], s["decode_from"]) for s in self.meta["sequences"]])
        self.assertEqual(self.manifest["seed"], synth.SEED)

    def test_golden_is_real_fla_fp32(self):
        for run, eps in ((G.GOLD, 1e-6), (G.ENGINE_EPS, 1e-5)):
            r = self.meta["runs"][run]
            self.assertTrue(r["fla"]["fla"].startswith("fla-core"), r["fla"])
            self.assertEqual(r["fla"]["fp32_dots"], "ieee")
            self.assertEqual(r["device"]["device"], "cuda")
            self.assertEqual(r["dtype"], "float32")
            self.assertEqual(r["settings"]["q_a_layernorm_eps"], eps)
            self.assertEqual(r["settings"]["kv_a_layernorm_eps"], eps)
            self.assertEqual(r["settings"]["index_k_norm_eps"], 1e-5)
            self.assertTrue(r["moonshot"]["matches_hf_moonshotai_Kimi-K3_f831ab6"])
        self.assertLess((G.GOLDEN_DIR / "golden.npz").stat().st_size, 1_600_000)
        self.assertNotIn("git", json.dumps(self.meta["runs"]))          # provenance is file sha256s, not commits

    # ---- 1. the gate passes, with the numbers the tolerances rest on
    def _gate(self, dtype, run=G.GOLD, **kw):
        rep = G.gate(self.model(dtype, **kw), self.golden, run=run)
        print(f"\n[{dtype}{' ' + str(kw) if kw else ''} vs golden {run}] " + rep.summary(), file=sys.stderr)
        return rep

    def test_gate_passes_float64(self):
        rep = self._gate("float64")
        self.assertTrue(rep.ok, rep.summary())
        self.assertEqual(rep.stats["runs"], 6)
        self.assertEqual(rep.stats["moe_rows"], 1872)                       # 8 MoE layers x 234 tokens
        self.assertEqual(rep.stats["idx_rows"], 702)                        # 3 MLA layers x 234 tokens
        self.assertEqual(rep.stats.get("moe_rows_differ", 0), 0)
        self.assertEqual(rep.stats.get("idx_rows_differ", 0), 0)

    def test_gate_passes_float32(self):
        rep = self._gate("float32")
        self.assertTrue(rep.ok, rep.summary())
        self.assertEqual(rep.stats.get("moe_rows_differ", 0), 0)
        self.assertEqual(rep.stats.get("idx_rows_differ", 0), 0)

    def test_golden_is_tie_free(self):
        """The engines leave tie-breaking unspecified, so the golden must not depend on it: on every row of
        every sequence judged on logits, the golden's own indexer and router cut margins are at least
        TIE_FREE (10) x the score tolerances. gate() enforces this itself; here it is stated with the numbers."""
        rep = G.gate(self.model("float64"), self.golden)
        self.assertTrue(rep.ok, rep.summary())
        self.assertGreaterEqual(rep.stats["idx_min_margin"], G.TIE_FREE * G.IDX_SCORE_ATOL)
        self.assertGreaterEqual(rep.stats["moe_min_margin"], G.TIE_FREE * G.MOE_TIE)
        self.assertEqual(rep.idx_exact_tie_rows, 0)
        self.assertEqual(rep.moe_near_ties, [])
        self.assertEqual(rep.idx_ties, [])
        self.assertTrue(all(s.get("tie_free") for s in self.meta["sequences"]), self.meta["sequences"])
        # and gate() refuses a golden that is not: squeeze one stored margin under the bound
        meta, arrs = self.golden
        bad = dict(arrs)
        k = f"{G.GOLD}/seq0.prefill/dsa.3.cut_margin"
        bad[k] = arrs[k].copy()
        bad[k][10] = 0.0
        rb = G.gate(self.model("float64"), (meta, bad), seqs=("seq0",), modes=("prefill",))
        self.assertIn("tie-free", rb.checks_failed())

    def test_tie_breaking_does_not_change_the_verdict(self):
        """Breaking indexer or router ties toward the later key instead of the earlier one passes the gate
        with every selection identical - on the round-1 fixture (4 indexer heads, unpicked sequences) the
        indexer variant failed with logits off by 2.6."""
        for name, (what, _, edits, _) in POSITIVE.items():
            mod = load_mutant(REF_SRC, edits, name)
            m = mod.KimiK31(self.fix / "tiny.gguf", dtype="float64")
            try:
                rep = G.gate(m, self.golden)
            finally:
                m.close()
            print(f"\n[{name}] " + rep.summary(), file=sys.stderr)
            self.assertTrue(rep.ok, f"{name} ({what}):\n" + rep.summary())
            self.assertEqual(rep.stats.get("idx_rows_differ", 0) + rep.stats.get("moe_rows_differ", 0), 0)

    def test_mla_eps_is_resolved_both_ways(self):
        """The golden uses Moonshot's q_a/kv_a eps 1e-6, the ENGINE_EPS run the engines' 1e-5. The reference
        with the matching eps passes, with the other eps fails - on the layer-3 seam (the logits max cannot
        see it)."""
        for dtype in ("float64", "float32"):
            for eps, good, bad in ((1e-6, G.GOLD, G.ENGINE_EPS), (1e-5, G.ENGINE_EPS, G.GOLD)):
                kw = {"eps_overrides": {"mla_latent": eps}}
                ok = self._gate(dtype, run=good, **kw)
                ko = self._gate(dtype, run=bad, **kw)
                self.assertTrue(ok.ok, f"{dtype} eps {eps} vs golden {good}: " + ok.summary())
                self.assertFalse(ko.ok, f"{dtype} eps {eps} vs golden {bad} should fail")
                self.assertIn("seam", ko.checks_failed())
                self.assertGreater(ko.stats["seam_rms"], 2 * ok.stats["seam_rms"])

    def test_fixture_exercises_the_guarded_cases(self):
        cov = self.ref_mod.COVER
        cov.clear()
        G.gate(self.model("float64"), self.golden, seqs=("seq0", "seq2"), modes=("prefill",))
        for k in ("idx_queries_over_top_k", "idx_negative_visible_scores", "idx_zero_fill_would_pick_future",
                  "moe_bias_changed_selection", "attn_res_block_pushes"):
            self.assertGreater(cov[k], 0, f"the fixture never hits {k}: {dict(cov)}")
        # tie-free by construction: exact ties are exercised by KimiK31Ties instead
        self.assertEqual(cov["idx_ties_at_cut"], 0, dict(cov))


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---- the engines' own indexer score functions (KimiK31IndexerEngines, KimiK31Ties)

def _vllm_logits_ref(q_deq, w, k_deq):
    """sum_h relu(q_h . k) * w_h for every (row, key)."""
    # Verbatim from vLLM PR #65685 @ 9d6ac73f, tests/models/kimi_k3/test_dsa_indexer_fused.py:551-553
    # (SPDX-License-Identifier: Apache-2.0, Copyright contributors to the vLLM project).
    import torch
    return (torch.einsum("thd,nd->thn", q_deq, k_deq).relu() * w[..., None]).sum(1)


def _sglang_mqa_logits(q, k, weights, ks, ke):
    """SGLang kimi-k3.1-support @ accf045c, python/sglang/srt/layers/attention/dsa/utils.py:465-494
    (_fp8_mqa_logits_torch, the pure-torch fallback for DeepGEMM's fp8_mqa_logits; Apache-2.0), for
    unquantized inputs: the same per-head relu(q_h k^T) accumulation weighted by the head gates and the
    same [ks, ke) validity window, without the bf16 cast of q and k (which only exists to read FP8 values)
    and with k_scale = 1, and with -inf instead of 0 outside the window (SGLang's topk_transform ranks only
    [ks, ke); -inf does the same for a plain top-k)."""
    import torch
    num_q, num_heads, head_dim = q.shape
    num_kv = k.shape[0]
    k_t = k.reshape(num_kv, head_dim).t()
    logits = torch.zeros(num_q, num_kv, dtype=q.dtype)
    for h in range(num_heads):
        scores = torch.mm(q[:, h], k_t)
        logits.addcmul_(torch.relu(scores), weights[:, h:h + 1])
    positions = torch.arange(num_kv).unsqueeze(0)
    valid = (positions >= ks.unsqueeze(1)) & (positions < ke.unsqueeze(1))
    return logits.masked_fill_(~valid, float("-inf"))


def _vllm_indexer_inputs(x, qr, w: dict, n_heads: int, head_dim: int, eps: float):
    """q, k, head weights the way vLLM PR #65685's KimiK3DSAIndexer builds them (vllm/models/kimi_k3/nvidia/
    dsa.py), unfused and without FP8: index_q = wq_index(q_c) with q_c the q_a_layernorm output (:337,
    :801-815); index_k, index_weights = wkv_index(hidden_states).split([head_dim, n_heads]) (:340), the merged
    [wk; weights_proj] GEMM on the attention input; index_k = KimiK3DSAIndexerKNorm (fp32 LayerNorm with weight
    and bias, cast back; :80-101); weights = index_weights * index_softmax_scale (head_dim^-0.5, :185) *
    indexer_weight_scale (n_heads^-0.5, :131), as score() folds them (:370). `w` holds checkpoint tensors
    (self_attn.indexer.{wq_b,wk,weights_proj}.weight, k_norm.{weight,bias}) as torch tensors."""
    import torch
    import torch.nn.functional as F
    index_q = (qr @ w["wq_b"].t()).view(-1, n_heads, head_dim)
    index_k, index_weights = (x @ torch.cat([w["wk"], w["weights_proj"]]).t()).split([head_dim, n_heads], dim=-1)
    index_k = F.layer_norm(index_k.double(), (head_dim,), w["k_norm.weight"].double(), w["k_norm.bias"].double(),
                           eps).type_as(index_k)
    weights = index_weights.double() * (head_dim ** -0.5 * n_heads ** -0.5)
    return index_q, index_k, weights


def _indexer_weights(il: int) -> dict:
    """The checkpoint's (hf-f32) indexer tensors of MLA layer il, float64 torch tensors."""
    import torch
    vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
    pre = f"{synth.P}layers.{il}.self_attn.indexer."
    return {n: torch.from_numpy(vals[pre + n + ("" if n.startswith("k_norm") else ".weight")].astype(np.float64))
            for n in ("wq_b", "wk", "weights_proj", "k_norm.weight", "k_norm.bias")}


@unittest.skipUnless(_have_torch(), "needs torch (the engines' reference functions are torch code)")
class KimiK31IndexerEngines(unittest.TestCase):
    """The independent model's indexer half is a transcription (Moonshot ships no K3.1 code), written from the
    same notes as the reference, so a shared misreading would pass both. This pins it to the engines' own code:
    the golden's recorded indexer inputs (each MLA layer's attention input and q_a_layernorm output, seq0
    prefill) go through vLLM's wiring and reference score function, and through SGLang's torch fallback, and
    must give the golden's scores and selections and the reference's scores."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.meta, cls.arrs = G.load_golden()
        cls.gold = G.part(cls.arrs, G.GOLD, G.INDEXER_INPUT_SEQ, "prefill")
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-engines-")
        cls.addClassCleanup(cls.tmp.cleanup)
        path = pathlib.Path(cls.tmp.name) / "tiny.gguf"
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        md, ts = synth.to_gguf(vals, synth.TINY)
        synth.write_gguf(path, md, ts)
        cls.ref = load_mutant(REF_SRC, [], "engines")
        cls.m = cls.ref.KimiK31(path, dtype="float64")
        cls.addClassCleanup(cls.m.close)

    def test_engine_score_functions_give_the_golden_and_the_reference(self):
        torch, g, c = self.torch, synth.TINY, self.m.cfg
        H, D, k = g["idx_heads"], g["idx_dim"], g["idx_topk"]
        eps = self.meta["runs"][G.GOLD]["settings"]["index_k_norm_eps"]
        worst = {}
        layers = c.mla_layers()
        self.assertEqual(sorted(int(key.split(".")[1]) for key in self.gold if key.endswith(".qr")), layers)
        for il in layers:
            x = torch.from_numpy(self.gold[f"layers.{il}.attn_in"].astype(np.float64))      # [T, hidden]
            qr = torch.from_numpy(self.gold[f"dsa.{il}.qr"].astype(np.float64))             # [T, q_lora]
            T = x.shape[0]
            q, kk, w = _vllm_indexer_inputs(x, qr, _indexer_weights(il), H, D, eps)
            vllm = _vllm_logits_ref(q, w, kk)
            vllm = vllm.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf")).numpy()
            ks, ke = torch.zeros(T, dtype=torch.long), torch.arange(1, T + 1)
            sgl = _sglang_mqa_logits(q, kk, w, ks, ke).numpy()
            # the reference's own indexer on the same inputs (a fresh state: every key is this call's)
            self.m.dumps = {}
            sel = self.m.indexer(x.numpy().T, qr.numpy().T, il, self.m.new_state(), np.arange(T))
            ref = self.m.dumps[f"indexer_score-{il}"].T
            gold = self.gold[f"dsa.{il}.scores"].astype(np.float64)
            for name, a in (("sglang", sgl), ("reference", ref), ("golden", gold)):
                self.assertTrue(np.array_equal(np.isneginf(a), np.isneginf(vllm)), f"layer {il} {name}: -inf pattern")
            fin = np.isfinite(vllm)
            for name, a in (("sglang", sgl), ("reference", ref), ("golden", gold)):
                worst[name] = max(worst.get(name, 0.0), float(np.abs(a[fin] - vllm[fin]).max()))
            gsel = self.gold[f"dsa.{il}.selected"]
            for t in range(T):
                top = np.argsort(-vllm[t, :t + 1], kind="stable")[:k]
                self.assertEqual(sorted(top.tolist()), sorted(int(i) for i in gsel[t] if i >= 0), f"layer {il} query {t}")
                self.assertEqual(sorted(top.tolist()), sorted(int(i) for i in sel[t]), f"layer {il} query {t}")
        print(f"\n[engine score functions on the golden's indexer inputs] max |diff| vs vLLM _logits_ref: {worst}",
              file=sys.stderr)
        self.assertLess(worst["sglang"], 1e-12)
        self.assertLess(worst["reference"], 1e-12)       # same inputs, float64 on both sides
        self.assertLess(worst["golden"], 2e-5)            # the golden computed them in fp32 on the GPU

    def test_formula_variants_are_told_apart(self):
        """The misreadings a shared transcription could hide each move the scores far beyond the tolerance."""
        torch, g = self.torch, synth.TINY
        H, D, il = g["idx_heads"], g["idx_dim"], self.m.cfg.mla_layers()[0]
        eps = self.meta["runs"][G.GOLD]["settings"]["index_k_norm_eps"]
        x = torch.from_numpy(self.gold[f"layers.{il}.attn_in"].astype(np.float64))
        qr = torch.from_numpy(self.gold[f"dsa.{il}.qr"].astype(np.float64))
        w = _indexer_weights(il)
        T = x.shape[0]
        fut = torch.ones(T, T, dtype=torch.bool).triu(1)
        gold = self.gold[f"dsa.{il}.scores"].astype(np.float64)
        fin = np.isfinite(gold)

        def dev(q, kk, wt, relu_first=True):
            if relu_first:
                s = _vllm_logits_ref(q, wt, kk)
            else:
                s = (torch.einsum("thd,nd->thn", q, kk) * wt[..., None]).sum(1).relu()
            return float(np.abs(s.masked_fill(fut, float("-inf")).numpy()[fin] - gold[fin]).max())
        q, kk, wt = _vllm_indexer_inputs(x, qr, w, H, D, eps)
        self.assertLess(dev(q, kk, wt), 2e-5)
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        q_a = torch.from_numpy(vals[f"{synth.P}layers.{il}.self_attn.q_a_proj.weight"].astype(np.float64))
        raw_k = (x @ w["wk"].t())
        variants = {
            "relu after the head sum": dev(q, kk, wt, relu_first=False),
            "weights unscaled": dev(q, kk, wt / (D ** -0.5 * H ** -0.5)),
            "k_norm without bias": dev(q, _vllm_indexer_inputs(x, qr, dict(w, **{"k_norm.bias": w["k_norm.bias"] * 0}),
                                                               H, D, eps)[1], wt),
            "k without the LayerNorm": dev(q, raw_k, wt),
            "q from the un-normed q latent": dev(_vllm_indexer_inputs(x, x @ q_a.t(), w, H, D, eps)[0], kk, wt),
        }
        print(f"\n[indexer formula variants vs the golden scores, max |diff|] {variants}", file=sys.stderr)
        for name, d in variants.items():
            self.assertGreater(d, 100 * G.IDX_SCORE_ATOL, name)


class KimiK31Ties(unittest.TestCase):
    """Exact ties at the indexer cut, built on purpose: the golden is tie-free, so the tie handling is checked
    here. The engines leave tie-breaking unspecified, so implementations that break ties differently must
    agree as SETS (ref/kimi_k31.py topk_sets_agree, which the gate uses), and the gate must refuse a golden
    whose cut sits on a tie."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-ties-")
        cls.addClassCleanup(cls.tmp.cleanup)
        path = pathlib.Path(cls.tmp.name) / "tiny.gguf"
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        md, ts = synth.to_gguf(vals, synth.TINY)
        synth.write_gguf(path, md, ts)
        cls.K = load_mutant(REF_SRC, [], "ties")
        cls.m = cls.K.KimiK31(path, dtype="float64")
        cls.addClassCleanup(cls.m.close)
        toks = synth.TEST_TOKENS["seq2"]
        cls.m.forward(toks)
        cls.il = cls.m.cfg.mla_layers()[0]
        x, qr = cls.m.dumps[f"attn_norm-{cls.il}"], cls.m.dumps[f"qr-{cls.il}"]
        # keys in groups of four identical attention inputs: identical index keys, so every query scores the
        # four exactly alike, and a top-6 cut often falls inside a group
        cls.T = x.shape[1]
        group = (np.arange(cls.T) // 4) * 4
        cls.x, cls.qr = np.ascontiguousarray(x[:, group]), qr
        cls.m.dumps = {}
        cls.sel = cls.m.indexer(cls.x, cls.qr, cls.il, cls.m.new_state(), np.arange(cls.T))
        cls.score = cls.m.dumps[f"indexer_score-{cls.il}"]            # (n_kv, T)
        k = cls.m.cfg.idx_top_k
        cls.tie_rows = [t for t in range(k, cls.T)
                        if (lambda o: o[k - 1] == o[k])(np.sort(cls.score[:t + 1, t])[::-1])]

    def later_key(self, t):
        vis = self.score[:t + 1, t]
        return len(vis) - 1 - self.K.top_k_desc(vis[::-1].copy(), self.m.cfg.idx_top_k)

    def test_ties_at_the_cut_agree_as_sets(self):
        self.assertGreater(len(self.tie_rows), 10, "the construction must put exact ties at the cut")
        differ = 0
        for t in range(self.T):
            a, b = self.sel[t], self.later_key(t)
            differ += int(set(a.tolist()) != set(b.tolist()))
            self.assertTrue(self.K.topk_sets_agree(self.score[:, t], a, b, tol=0.0), f"query {t}")
        self.assertGreater(differ, 5, "breaking ties the other way must actually select other keys")
        # a non-tied swap is not a tie: drop a selected key for one scoring strictly below the cut
        t = self.tie_rows[0]
        vis = self.score[:t + 1, t]
        cut = vis[self.sel[t]].min()
        below = [int(i) for i in np.flatnonzero(vis < cut)]
        if below:
            wrong = list(self.sel[t][:-1]) + [below[0]]
            self.assertFalse(self.K.topk_sets_agree(self.score[:, t], self.sel[t], wrong, tol=0.0))

    def test_gate_accepts_a_tie_only_difference_and_refuses_a_tied_golden(self):
        """compare() with this tied row as the golden: a reference that keeps the other tied keys is reported as
        a tie, not an indexer-set failure, and the golden itself fails the tie-free check."""
        k, il = self.m.cfg.idx_top_k, self.il
        sc = self.score.T                                                  # [T, n_kv]
        o = np.sort(np.where(np.isfinite(sc), sc, -np.inf), axis=1)[:, ::-1]
        margin = np.array([o[t, k - 1] - o[t, k] if t + 1 > k else np.inf for t in range(self.T)])
        gsel = np.full((self.T, k), -1, dtype=np.int16)
        for t in range(self.T):
            gsel[t, :len(self.sel[t])] = np.sort(self.sel[t])
        gold = {"logits": np.zeros((self.T, 4), np.float32), f"dsa.{il}.selected": gsel,
                f"dsa.{il}.cut_margin": margin.astype(np.float32)}
        ref = {"logits": np.zeros((self.T, 4)), f"dsa.{il}.selected": [sorted(self.later_key(t).tolist())
                                                                       for t in range(self.T)],
               f"dsa.{il}.scores": sc}
        rep = G.Report(stats={"dtype": "float64", "runs": 1})
        acc = {"lg_ss": 0.0, "lg_n": 0, "seam_ss": 0.0, "seam_n": 0, "kda_seam_ss": 0.0, "kda_seam_n": 0}
        G.compare(ref, gold, {f"dsa.{il}.scores": sc}, "ties", rep, acc, G.TOL["float64"])
        self.assertGreater(len(rep.idx_ties), 5, rep.summary())
        self.assertEqual(rep.checks_failed(), ["tie-free"], rep.summary())

    @unittest.skipUnless(_have_torch(), "needs torch")
    def test_independent_indexer_and_engines_on_ties(self):
        """The independent model's indexer (earlier-key rule, like the reference) selects the same keys; vLLM's
        reference scores with torch.topk (no tie rule) agree as sets."""
        import torch
        sys.path.insert(0, str(HERE))
        import kimi_k31_moonshot_indexer as kidx
        g = synth.TINY
        H, D, k = g["idx_heads"], g["idx_dim"], g["idx_topk"]
        ix = kidx.LightningIndexer(g["n_embd"], g["q_lora"], H, D, k, self.m.cfg.eps_indexer_ln).double()
        score = torch.from_numpy(np.ascontiguousarray(self.score.T))[None]
        _, keep = ix.select(score, torch.arange(self.T))
        q, kk, w = _vllm_indexer_inputs(torch.from_numpy(self.x.T.copy()), torch.from_numpy(self.qr.T.copy()),
                                        _indexer_weights(self.il), H, D, self.m.cfg.eps_indexer_ln)
        vllm = _vllm_logits_ref(q, w, kk).masked_fill(torch.ones(self.T, self.T, dtype=torch.bool).triu(1),
                                                      float("-inf"))
        self.assertLess(float(np.abs(vllm.numpy()[np.isfinite(self.score.T)] - self.score.T[np.isfinite(self.score.T)]).max()),
                        1e-12)
        for t in range(self.T):
            self.assertEqual(set(keep[0, t].nonzero().flatten().tolist()), set(self.sel[t].tolist()), f"query {t}")
            top = vllm[t, :t + 1].topk(min(k, t + 1)).indices.numpy()
            self.assertTrue(self.K.topk_sets_agree(self.score[:, t], self.sel[t], top, tol=1e-12), f"query {t}")


MOONSHOT_DIR = os.environ.get("KIMI_K31_MOONSHOT_DIR")


@unittest.skipUnless(MOONSHOT_DIR, "set KIMI_K31_MOONSHOT_DIR to a directory with Moonshot's Kimi-K3 modeling files")
class KimiK31GateLive(unittest.TestCase):
    """The live oracle on the regenerated fixture (fla + CUDA), then the same gate."""

    @classmethod
    def setUpClass(cls):
        probe = subprocess.run([sys.executable, "-c", "import importlib.util, torch; "
                                "print(int(torch.cuda.is_available() and importlib.util.find_spec('fla') is not None))"],
                               capture_output=True, text=True, timeout=300)
        if probe.stdout.strip() != "1":
            raise unittest.SkipTest(f"needs torch with CUDA and fla: {probe.stdout.strip()} {probe.stderr.strip()[-300:]}")
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-live-")
        cls.addClassCleanup(cls.tmp.cleanup)
        root = pathlib.Path(cls.tmp.name)
        fix = root / "fixture"
        synth.build(fix)
        mem = os.environ.get("KIMI_K31_GPU_MEM_GB", "1.0")
        for run, extra in (("1e-6", []), ("1e-5", ["--mla-norm-eps", "1e-5"])):
            cmd = [sys.executable, str(HERE / "kimi_k31_moonshot_oracle.py"), "run",
                   "--modeling-dir", MOONSHOT_DIR, "--fixture", str(fix), "--device", "cuda",
                   "--fla", "real", "--gpu-mem-gb", mem, "--out", str(root / f"live-{run}")] + extra
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                raise AssertionError(f"oracle run {run} failed ({p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
            dev = json.loads((root / f"live-{run}" / "run.json").read_text())["device"]
            print(p.stdout.strip() + f" [{dev.get('gpu')}, peak {dev.get('peak_allocated_mib')} MiB]", file=sys.stderr)
        G.export(root / "live-1e-6", root / "live-1e-5", None, root / "golden")
        cls.live = G.load_golden(root / "golden")
        cls.committed = G.load_golden()
        ref = load_mutant(REF_SRC, [], "live")
        cls.model = ref.KimiK31(fix / "tiny.gguf", dtype="float64")

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_reference_passes_against_the_live_oracle(self):
        rep = G.gate(self.model, self.live)
        print("\n[live] " + rep.summary(), file=sys.stderr)
        self.assertTrue(rep.ok, rep.summary())
        self.assertFalse(G.gate(self.model, self.live, run=G.ENGINE_EPS).ok)

    def test_live_oracle_reproduces_the_committed_golden(self):
        """fla is not bit-reproducible: the live run must agree with the committed one within the gate's
        float tolerances and exactly on every selection."""
        (_, live), (_, comm) = self.live, self.committed
        self.assertEqual(sorted(live), sorted(comm))
        worst = {}
        for k in comm:
            a, b = comm[k], live[k]
            if a.dtype.kind in "iu" or k.endswith(".cut_margin"):
                if a.dtype.kind in "iu":
                    self.assertTrue(np.array_equal(a, b), f"{k}: selections differ between runs")
                continue
            fin = np.isfinite(a)
            self.assertTrue(np.array_equal(fin, np.isfinite(b)), k)
            d = float(np.abs(a[fin].astype(np.float64) - b[fin]).max()) if fin.any() else 0.0
            leaf = k.split("/")[-1]
            kind = leaf.split(".")[0] if not leaf.endswith(".attn_in") else "indexer inputs (attn_in)"
            worst[kind] = max(worst.get(kind, 0.0), d)
            # attn_in of the later MLA layers carries the noise of every layer before it (3.3e-5 between
            # two runs), so it gets a bound of its own
            tol = {"logits": G.TOL["float64"]["logits_atol"], "layers": G.TOL["float64"]["seam_atol"],
                   "indexer inputs (attn_in)": 1e-4, "dsa": G.IDX_SCORE_ATOL}[kind]
            self.assertLessEqual(d, tol, f"{k}: live vs committed {d:.3e} > {tol:.0e}")
        print(f"\n[live vs committed golden] max |diff| {worst}", file=sys.stderr)


class KimiK31GateNegative(unittest.TestCase):
    """Every mutation must make the gate fail (float64 reference)."""

    table: list = []

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-neg-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = pathlib.Path(cls.tmp.name)
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        md, ts = synth.to_gguf(vals, synth.TINY)
        cls.gguf = cls.root / "tiny.gguf"
        synth.write_gguf(cls.gguf, md, ts)
        cls.golden = G.load_golden()
        cls.table = []

    @classmethod
    def tearDownClass(cls):
        if cls.table:
            w = max(len(r[0]) for r in cls.table)
            lines = [f"{'mutation':{w}s}  {'caught by':48s}  logits max  expert rows  indexer rows"]
            for name, kinds, lmax, er, ir in sorted(cls.table):
                lines.append(f"{name:{w}s}  {','.join(kinds) or 'NOTHING':48s}  {lmax:10.3e}  {er:11d}  {ir:12d}")
            print("\nKimi K3.1 gate, negative tests (float64 reference vs the golden):\n" + "\n".join(lines),
                  file=sys.stderr)

    def run_mutation(self, name):
        what, target, edits, eps = MUTATIONS[name]
        if target == "synth":
            smod = load_mutant(SYNTH_SRC, edits, name)
            vals, _, _ = smod.make_weights(smod.TINY, smod.SEED)
            md, ts = smod.to_gguf(vals, smod.TINY)
            path = self.root / f"{name}.gguf"
            smod.write_gguf(path, md, ts)
            rmod = load_mutant(REF_SRC, [], name)
        else:
            path = self.gguf
            rmod = load_mutant(REF_SRC, edits, name)
        m = rmod.KimiK31(path, dtype="float64", eps_overrides=eps)
        try:
            rep = G.gate(m, self.golden)
        finally:
            m.close()
        self.table.append((name, rep.checks_failed(), rep.stats.get("logits_max", float("nan")),
                           rep.stats.get("moe_rows_differ", 0), rep.stats.get("idx_rows_differ", 0)))
        self.assertFalse(rep.ok, f"mutation {name} ({what}) passed the gate:\n{rep.summary()}")
        return rep

    def test_mutation_mla_eps_is_caught_on_the_seam(self):
        rep = self.run_mutation("mla_eps_1e-5")
        self.assertIn("seam", rep.checks_failed())

    def test_mutation_future_zero_fill_selects_future_keys(self):
        rep = self.run_mutation("indexer_future_zero_fill")
        self.assertTrue(any("future keys" in f for f in rep.failures), rep.summary())

    def test_mutation_zero_fill_alone_changes_only_the_scores(self):
        rep = self.run_mutation("indexer_future_zero_fill_only")
        self.assertEqual(rep.checks_failed(), ["indexer-scores"])
        self.assertTrue(all("future" in f for f in rep.failures), rep.summary())

    def test_mutation_unscaled_indexer_is_seen_by_the_scores_only(self):
        rep = self.run_mutation("indexer_weights_unscaled")
        self.assertEqual(rep.checks_failed(), ["indexer-scores"])

    def test_mutation_topk_8_shrinks_the_expert_sets(self):
        rep = self.run_mutation("moe_topk_truncated_8")
        self.assertEqual(rep.stats.get("moe_rows_differ", 0), 1872)

    def test_mutation_kda_l2_eps_is_caught_on_the_kda_seam(self):
        rep = self.run_mutation("kda_l2_eps_1e-5")
        self.assertIn("kda-seam", rep.checks_failed())

    def test_mutation_squeezed_router_margins_do_not_excuse_expert_mismatches(self):
        rep = self.run_mutation("router_bias_dropped_margins_squeezed")
        self.assertIn("experts", rep.checks_failed())
        self.assertEqual(rep.moe_near_ties, [])                 # judged on the golden's margins, not its own

    def test_mutation_a_log_sliced_by_the_wrong_head_count_is_refused(self):
        smod = load_mutant(SYNTH_SRC, A_LOG_WRONG_COUNT, "a_log_wrong_count")
        vals, _, _ = smod.make_weights(smod.TINY, smod.SEED)
        md, ts = smod.to_gguf(vals, smod.TINY)
        path = self.root / "a_log_wrong_count.gguf"
        smod.write_gguf(path, md, ts)
        rmod = load_mutant(REF_SRC, [], "a_log_wrong_count")
        with self.assertRaisesRegex(ValueError, r"ssm_a: shape \(%d,\), expected \(%d,\)"
                                    % (synth.TINY["kda_head_dim"], synth.TINY["n_head"])):
            rmod.KimiK31(path, dtype="float64").close()
        self.table.append(("a_log_sliced_by_head_dim", ["refused (ssm_a shape)"], float("nan"), 0, 0))


def _add_mutation_tests():
    special = {"mla_eps_1e-5", "indexer_future_zero_fill", "indexer_future_zero_fill_only", "indexer_weights_unscaled",
               "moe_topk_truncated_8", "kda_l2_eps_1e-5", "router_bias_dropped_margins_squeezed"}
    for name in MUTATIONS:
        if name in special:
            continue

        def t(self, name=name):
            self.run_mutation(name)
        t.__doc__ = MUTATIONS[name][0]
        setattr(KimiK31GateNegative, f"test_mutation_{name.replace('-', '_')}", t)


_add_mutation_tests()


if __name__ == "__main__":
    unittest.main()
