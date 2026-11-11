"""tools/test_kimi_k31_ref.py - the Kimi K3.1 reference forward pass (ref/kimi_k31.py), checked on
the synthetic fixture (tools/kimi_k31_synth_gguf.py, seed 3101, its 32 suggested test tokens).

Self-consistency, no GPU, no network, numpy only, a few seconds:
  * the config reads every key from the GGUF, falls back to the released K3.1 values, and refuses
    what the reference does not implement;
  * prefill, chunked prefill and token-by-token decode agree (KDA conv history and recurrent
    state, MLA latent / key-part / indexer caches) to 1e-10 in the float64 mode and to float32
    rounding in the default mode, selections compared as sets with tie tolerance;
  * the indexer is causal (-inf, never 0, on future keys), keeps min(top_k, t+1) keys, and with
    top_k >= the sequence it IS dense causal MLA (bit-identical); with the fixture's top_k it is
    dense exactly on the rows that see <= top_k keys and differs after;
  * the absorbed MLA equals the materialized kv_b form, the KDA layer equals a tokens-first
    transcription of the recurrence, and the indexer score equals a tokens-first transcription
    of the published formula (all three written here independently of the reference's code,
    compared against the float64 mode to 1e-9);
  * the attention-residual candidate counts follow the block layout, the MoE routing picks by
    biased score and weights by the renormalized unbiased one, changing a token never changes an
    earlier position, and the two eps presets give different but close logits.

    python -m unittest tools.test_kimi_k31_ref        (or unittest discover, like the other tool tests)

KIMI_K31_GGUF=/path/tiny.gguf runs against an existing file instead of generating one.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))

import numpy as np  # noqa: E402

import kimi_k31_synth_gguf as synth  # noqa: E402
import kimi_k31 as K  # noqa: E402  (ref/kimi_k31.py)

F64 = np.float64


# ---- independent tokens-first helpers (the checkpoint's own (T, ...) layout, float64)
def _rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_last(s):
    z = np.exp(s - s.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class KimiK31RefTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-ref-")
        env = os.environ.get("KIMI_K31_GGUF")
        if env:
            cls.gguf = pathlib.Path(env)
        else:
            cls.gguf = pathlib.Path(cls.tmp.name) / "tiny.gguf"
            vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
            md, ts = synth.to_gguf(vals, synth.TINY)
            synth.write_gguf(cls.gguf, md, ts)
        cls.tokens = synth.sample_tokens(synth.TINY, synth.SEED)
        cls.m = K.KimiK31(cls.gguf, cache_weights=True)
        cls.cfg = cls.m.cfg
        cls.logits = cls.m.forward(cls.tokens)
        cls.d = dict(cls.m.dumps)
        cls.m64 = K.KimiK31(cls.gguf, cache_weights=True, dtype="float64")
        cls.logits64 = cls.m64.forward(cls.tokens)
        cls.d64 = dict(cls.m64.dumps)
        cls.T = len(cls.tokens)
        cls.mla = cls.cfg.mla_layers()
        cls.moe = list(range(cls.cfg.n_layer_dense_lead, cls.cfg.n_layers))
        cls._others: list = []

    @classmethod
    def tearDownClass(cls):
        for m in [cls.m, cls.m64] + cls._others:
            m.close()                       # Windows will not unlink an open GGUF
        cls.tmp.cleanup()

    def other(self, **kw) -> K.KimiK31:
        m = K.KimiK31(self.gguf, cache_weights=True, **kw)
        self._others.append(m)
        return m

    # ================================ config ================================

    def test_config_from_metadata(self):
        c = self.cfg
        self.assertEqual((c.n_vocab, c.n_embd, c.n_layers, c.n_head), (320, 256, 9, 6))
        self.assertEqual(self.mla, [3, 7, 8])
        self.assertEqual((c.kda_head_dim, c.ssm_d_conv, c.kda_gate_lower_bound), (32, 4, -5.0))
        self.assertEqual((c.q_lora_rank, c.kv_lora_rank, c.qk_nope, c.n_rot, c.n_head_v_mla), (160, 512, 128, 64, 128))
        self.assertEqual((c.idx_n_head, c.idx_key_dim, c.idx_top_k), (12, 32, 6))
        self.assertEqual((c.n_expert, c.n_expert_used, c.n_expert_shared, c.n_ff_exp, c.n_expert_latent),
                         (24, 10, 2, 64, 128))
        self.assertEqual((c.n_layer_dense_lead, c.n_ff, c.attn_res_block_size), (1, 448, 3))
        self.assertEqual((c.situ_beta, c.situ_linear_beta), (4.0, 25.0))
        e = c.eps()
        self.assertAlmostEqual(e["rms"], 1e-5, delta=1e-12)
        self.assertAlmostEqual(e["indexer_ln"], 1e-5, delta=1e-12)
        self.assertEqual((e["mla_latent"], e["kda_l2"], e["router_renorm"]), (1e-6, 1e-6, 1e-20))
        ll = K.KimiK31Config(self.m.g.g.metadata, eps="llama.cpp").eps()
        self.assertEqual((ll["mla_latent"], ll["kda_l2"]), (e["rms"], e["rms"]))
        ov = K.KimiK31Config(self.m.g.g.metadata, eps_overrides={"mla_latent": 3e-6}).eps()
        self.assertEqual(ov["mla_latent"], 3e-6)

    def test_config_released_defaults(self):
        c = K.KimiK31Config({"general.architecture": "kimi-k31"})
        self.assertEqual((c.n_vocab, c.n_embd, c.n_layers, c.n_head), (163840, 7168, 93, 96))
        self.assertEqual(c.mla_layers(), list(range(3, 92, 4)) + [92])
        self.assertEqual(len(c.mla_layers()), 24)
        self.assertEqual((c.idx_n_head, c.idx_key_dim, c.idx_top_k), (32, 128, 2048))
        self.assertEqual((c.n_expert, c.n_expert_used, c.n_expert_latent, c.n_ff_exp), (896, 16, 3584, 3072))
        self.assertEqual((c.kv_lora_rank, c.n_rot, c.n_head_k_mla, c.qk_nope, c.q_lora_rank), (512, 64, 192, 128, 1536))
        self.assertEqual((c.attn_res_block_size, c.kda_head_dim, c.d_inner), (12, 128, 12288))

    def test_config_refusals(self):
        base = dict(self.m.g.g.metadata)
        cases = [
            ({"general.architecture": "kimi-k3"}, ValueError),
            ({"kimi-k31.attention.indexer.kpool": 4}, NotImplementedError),
            ({"kimi-k31.attention.indexer.types": [1, 1, 1, 0, 1, 1, 1, 1, 1]}, NotImplementedError),
            ({"kimi-k31.attention.indexer.block_size": 64}, NotImplementedError),
            ({"kimi-k31.expert_gating_func": 1}, NotImplementedError),
            ({"kimi-k31.expert_group_count": 8}, NotImplementedError),
            ({"kimi-k31.attention.key_length": 512}, ValueError),
            ({"kimi-k31.attention.value_length": 576}, ValueError),
            ({"kimi-k31.attn_res.block_size": 0}, NotImplementedError),
            ({"kimi-k31.attention.indexer.top_k": 0}, ValueError),
            ({"kimi-k31.attention.head_count_kv": [0, 1]}, ValueError),
        ]
        for patch, exc in cases:
            with self.subTest(patch=patch), self.assertRaises(exc):
                K.KimiK31Config({**base, **patch})
        with self.assertRaises(ValueError):
            K.KimiK31Config(base, eps="bf16")
        with self.assertRaises(ValueError):
            K.KimiK31Config(base, eps_overrides={"q_a": 1e-5})
        K.KimiK31Config({**base, "kimi-k31.attention.indexer.types": [1] * 9})   # all full: accepted

    def test_missing_tensor_refused(self):
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        md, ts = synth.to_gguf(vals, synth.TINY, vocab=False)
        p = pathlib.Path(self.tmp.name) / "missing.gguf"
        synth.write_gguf(p, md, [(n, a) for n, a in ts if n != "blk.7.indexer.proj.weight"])
        with self.assertRaisesRegex(ValueError, "blk.7.indexer.proj.weight"):
            K.KimiK31(p)

    # ================================ whole model ================================

    def test_logits(self):
        self.assertEqual(self.logits.shape, (320, self.T))
        self.assertTrue(np.isfinite(self.logits).all())
        self.assertGreater(float(self.logits.std()), 0.1)
        self.assertTrue(np.array_equal(self.logits, self.d["result_output"]))
        self.assertEqual((self.logits.dtype, self.logits64.dtype), (np.float32, np.float64))
        # the float32 seams track the float64 pass to float32 rounding
        scale = float(np.abs(self.logits64).max())
        self.assertLess(float(np.abs(self.logits - self.logits64).max()), 1e-4 * scale)

    def test_causal_in_tokens(self):
        """Changing the last token changes nothing before it (bit-exact) and its own logits a lot."""
        toks = list(self.tokens)
        toks[-1] = (toks[-1] + 1) % self.cfg.n_vocab
        lg = self.other().forward(toks)
        self.assertTrue(np.array_equal(lg[:, :-1], self.logits[:, :-1]))
        self.assertGreater(float(np.abs(lg[:, -1] - self.logits[:, -1]).max()), 1e-2)

    def test_eps_presets_differ_slightly(self):
        lg = self.other(eps="llama.cpp").forward(self.tokens)
        diff = float(np.abs(lg - self.logits).max())
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1e-2 * float(np.abs(self.logits).max()))

    # ================================ prefill vs decode ================================

    def _decode(self, m: K.KimiK31, chunks: list[int]):
        st = m.new_state()
        logits, sel, ids, at = [], {il: [] for il in self.mla}, {il: [] for il in self.moe}, 0
        for n in chunks:
            logits.append(m.forward(self.tokens[at:at + n], st))
            for il in self.mla:
                top = m.dumps[f"top_k-{il}"]
                sel[il] += [top[:, j] for j in range(n)]
            for il in self.moe:
                ids[il] += [m.expert_ids(il)[:, j] for j in range(n)]
            at += n
        self.assertEqual(st.n_past, self.T)
        return np.concatenate(logits, axis=1), sel, ids, st

    def _check_against_prefill(self, chunks: list[int], dtype: str, rel: float):
        """`rel`: allowed max |logit difference| relative to the largest logit."""
        full, d = (self.logits64, self.d64) if dtype == "float64" else (self.logits, self.d)
        lg, sel, ids, st = self._decode(self.other(dtype=dtype), chunks)
        scale = float(np.abs(full).max())
        self.assertLess(float(np.abs(lg - full).max()), rel * scale, f"chunks {chunks}")
        for il in self.mla:
            score = d[f"indexer_score-{il}"]
            for t in range(self.T):
                self.assertTrue(K.topk_sets_agree(score[:, t], d[f"top_k-{il}"][:, t], sel[il][t], tol=rel * 10),
                                f"layer {il} query {t}")
        for il in self.moe:
            biased = d[f"ffn_moe_probs_biased-{il}"]
            for t in range(self.T):
                self.assertTrue(K.topk_sets_agree(biased[:, t], d[f"ffn_moe_topk-{il}"][:, t], ids[il][t],
                                                  tol=rel * 10), f"MoE layer {il} token {t}")
        return st

    def test_token_by_token_decode_matches_prefill(self):
        st_dec = self._check_against_prefill([1] * self.T, "float64", 1e-10)
        st_pre = self.m64.new_state()
        self.other(dtype="float64").forward(self.tokens, st_pre)
        for il in range(self.cfg.n_layers):
            if self.cfg.is_recr(il):          # the KDA conv history and recurrent state
                self.assertEqual(len(st_dec.conv[il]), 3)
                for a, b in zip(st_pre.conv[il], st_dec.conv[il]):
                    self.assertEqual(a.shape, (self.cfg.d_inner, self.cfg.ssm_d_conv - 1))
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(st_pre.ssm[il], st_dec.ssm[il], rtol=1e-10, atol=1e-12)
                self.assertGreater(float(np.abs(st_pre.ssm[il]).max()), 1e-3)
            else:                             # the MLA latent, key part and indexer caches
                for name in ("kv", "k_pe", "idx_k"):
                    a, b = getattr(st_pre, name)[il], getattr(st_dec, name)[il]
                    self.assertEqual(a.shape[1], self.T)
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)

    def test_chunked_prefill_matches_prefill(self):
        self._check_against_prefill([5, 1, 2, 9, 15], "float64", 1e-10)   # chunks shorter than the conv history too

    def test_float32_decode_within_rounding(self):
        """The default float32 mode: token-by-token decode differs from prefill only by float32
        rounding (matrix-vector vs matrix-matrix products), measured at about 1e-5 relative."""
        self._check_against_prefill([1] * self.T, "float32", 1e-4)

    # ================================ the indexer ================================

    def test_indexer_causal_topk(self):
        k = self.cfg.idx_top_k
        for il in self.mla:
            score, top, mask = (self.d[f"{n}-{il}"] for n in ("indexer_score", "top_k", "kq_mask_top_k"))
            self.assertEqual(score.shape, (self.T, self.T))
            self.assertEqual(top.shape, (k, self.T))
            for t in range(self.T):
                sel = [int(i) for i in top[:, t] if i >= 0]
                self.assertEqual(len(sel), min(k, t + 1))
                self.assertEqual(len(set(sel)), len(sel))
                self.assertLessEqual(max(sel), t)                           # never a future key
                self.assertTrue(np.all(np.isneginf(score[t + 1:, t])))     # -inf, not 0
                self.assertTrue(np.all(np.isfinite(score[:t + 1, t])))
                if t + 1 <= k:
                    self.assertEqual(set(sel), set(range(t + 1)))           # dense while <= top_k visible
                ref = np.argsort(-score[:t + 1, t].astype(F64), kind="stable")[:k]
                self.assertTrue(K.topk_sets_agree(score[:, t], ref, sel, tol=1e-6))
                self.assertEqual(set(np.flatnonzero(mask[:, t] == 0.0)), set(sel))
                self.assertTrue(np.all(np.isneginf(np.delete(mask[:, t], sel))))
        # the fixture exercises the trap of rows where a 0-filled mask would rank a future key above the
        # k-th visible score; its test sequence is picked tie-free, so exact ties at the cut are built on
        # purpose in test_indexer_ties_compare_as_sets
        ties = fill_traps = 0
        for il in self.mla:
            score = self.d[f"indexer_score-{il}"]
            for t in range(k, self.T):
                o = np.sort(score[:t + 1, t])[::-1]
                ties += int(o[k - 1] == o[k])
                fill_traps += int(o[k - 1] < 0.0)
        self.assertEqual(ties, 0)
        self.assertGreater(fill_traps, 0)

    def test_indexer_ties_compare_as_sets(self):
        """Exact ties at the cut (keys in groups of four identical attention inputs): the reference keeps the
        earlier key (ggml_top_k's order), which the engines do not promise, so a selection that keeps the
        later tied keys instead must agree as a set, and one that swaps in a lower-scoring key must not."""
        c, m, k = self.cfg, self.m64, self.cfg.idx_top_k
        il = self.mla[0]
        group = (np.arange(self.T) // 4) * 4
        x = np.ascontiguousarray(self.d64[f"attn_norm-{il}"][:, group])
        m.dumps = {}
        sel = m.indexer(x, self.d64[f"qr-{il}"], il, m.new_state(), np.arange(self.T))
        score = m.dumps[f"indexer_score-{il}"]
        ties = differ = 0
        for t in range(k, self.T):
            vis = score[:t + 1, t]
            o = np.sort(vis)[::-1]
            ties += int(o[k - 1] == o[k])
            later = len(vis) - 1 - K.top_k_desc(vis[::-1].copy(), k)
            differ += int(set(later.tolist()) != set(sel[t].tolist()))
            self.assertTrue(K.topk_sets_agree(score[:, t], sel[t], later, tol=0.0))
            below = np.flatnonzero(vis < vis[sel[t]].min())
            if below.size:
                self.assertFalse(K.topk_sets_agree(score[:, t], sel[t], list(sel[t][:-1]) + [int(below[0])], tol=0.0))
        self.assertGreater(ties, 10)
        self.assertGreater(differ, 5)

    def test_indexer_matches_published_formula(self):
        """score[t,s] = sum_h w[t,h] relu(q[t,h].k[s]), q = wq_b(q_a_norm(q_a(x))), k = LayerNorm(wk(x)),
        w = proj(x) * H^-0.5 * D^-0.5, -inf for s > t; written tokens-first from the formula."""
        c, m, d = self.cfg, self.m64, self.d64
        Hi, Di = c.idx_n_head, c.idx_key_dim
        for il in self.mla:
            X = d[f"attn_norm-{il}"].T
            W = lambda n, s="weight": m.t(f"blk.{il}.{n}.{s}").astype(F64)  # noqa: E731
            qr = _rms(X @ W("attn_q_a").T, W("attn_q_a_norm"), c.eps_mla_latent)
            q = (qr @ W("indexer.attn_q_b").T).reshape(self.T, Hi, Di)
            kk = X @ W("indexer.attn_k").T
            mu, var = kk.mean(-1, keepdims=True), kk.var(-1, keepdims=True)
            kk = (kk - mu) / np.sqrt(var + c.eps_indexer_ln) * W("indexer.k_norm") + W("indexer.k_norm", "bias")
            w = (X @ W("indexer.proj").T) * Hi ** -0.5 * Di ** -0.5
            ref = (np.maximum(np.einsum("thd,sd->ths", q, kk), 0.0) * w[:, :, None]).sum(1)   # (T, S)
            ref[np.triu_indices(self.T, 1)] = -np.inf
            got = d[f"indexer_score-{il}"].T
            self.assertTrue(np.array_equal(np.isneginf(got), np.isneginf(ref)))
            fin = np.isfinite(ref)
            np.testing.assert_allclose(got[fin], ref[fin], rtol=1e-9, atol=1e-12)
            for t in range(self.T):
                sel = np.argsort(-ref[t, :t + 1], kind="stable")[:c.idx_top_k]
                self.assertTrue(K.topk_sets_agree(ref[t], sel, d[f"top_k-{il}"][:, t], tol=1e-9))

    def test_topk_ge_seq_is_dense_mla(self):
        """top_k >= every query's visible keys: the indexer selects everything and the model IS
        dense causal MLA, bit for bit. With the fixture's top_k the rows that see <= top_k keys
        are still bit-identical to dense and the rest differ."""
        dense = self.other(indexer=False).forward(self.tokens)
        big = self.other(index_top_k=10 ** 6)
        self.assertTrue(np.array_equal(big.forward(self.tokens), dense))
        k = self.cfg.idx_top_k
        self.assertTrue(np.array_equal(self.logits[:, :k], dense[:, :k]))
        self.assertGreater(float(np.abs(self.logits[:, k:] - dense[:, k:]).max()), 1e-3)

    # ================================ MLA, KDA: independent transcriptions ================================

    def test_mla_absorbed_equals_materialized(self):
        """Moonshot's form: K_h = [kv_b_h(c)[:nope] | k_rot shared, unrotated], V_h = kv_b_h(c)[nope:],
        softmax(Q K^T / sqrt(192) + mask) V, with kv_b re-assembled from the GGUF's k_b / v_b split
        and the reference's own selection mask."""
        c, m, d = self.cfg, self.m64, self.d64
        H, nope, rope, vh, kvl = c.n_head, c.qk_nope, c.n_rot, c.n_head_v_mla, c.kv_lora_rank
        for il in self.mla:
            X = d[f"attn_norm-{il}"].T
            W = lambda n: m.t(f"blk.{il}.{n}.weight").astype(F64)  # noqa: E731
            qr = _rms(X @ W("attn_q_a").T, W("attn_q_a_norm"), c.eps_mla_latent)
            q = (qr @ W("attn_q_b").T).reshape(self.T, H, nope + rope)
            ckv = X @ W("attn_kv_a_mqa").T
            lat = _rms(ckv[:, :kvl], W("attn_kv_a_norm"), c.eps_mla_latent)
            k_rot = ckv[:, kvl:]
            kv_b = np.concatenate([W("attn_k_b").transpose(0, 2, 1), W("attn_v_b")], axis=1)   # (H, nope+v, kvl)
            kv = np.einsum("tc,hrc->thr", lat, kv_b)
            key = np.concatenate([kv[..., :nope], np.broadcast_to(k_rot[:, None, :], (self.T, H, rope))], -1)
            s = np.einsum("thd,shd->hts", q, key) * (nope + rope) ** -0.5
            p = _softmax_last(s + d[f"kq_mask_top_k-{il}"].T[None])
            o = np.einsum("hts,shv->thv", p, kv[..., nope:]).reshape(self.T, H * vh)
            np.testing.assert_allclose(d[f"kqv_out-{il}"].T, o, rtol=1e-9, atol=1e-12)
            out = (o * _sig(X @ W("attn_gate").T)) @ W("attn_output").T
            np.testing.assert_allclose(d[f"mla_out-{il}"].T, out, rtol=1e-9, atol=1e-12)

    def test_kda_matches_recurrence_transcription(self):
        """Per head, tokens-first: conv taps w[:, -1] on the current token, l2 norms, q * dk^-0.5,
        g = lb*sigmoid(exp(A_log)*(f_b(f_a x) + dt_bias)) with A_log = log(-ssm_a),
        S <- diag(exp g) S; S += k (beta (v - S^T k))^T; o = S^T q; RMSNorm * sigmoid(g_proj x); o_proj."""
        c, m, d = self.cfg, self.m64, self.d64
        H, dk, Wc = c.n_head, c.kda_head_dim, c.ssm_d_conv
        for il in [i for i in range(c.n_layers) if c.is_recr(i)]:
            X = d[f"attn_norm-{il}"].T
            W = lambda n, s="weight": m.t(f"blk.{il}.{n}.{s}").astype(F64)  # noqa: E731

            def conv(s):
                lin = X @ W(f"attn_{s}").T
                w = W(f"ssm_conv1d_{s}").reshape(c.d_inner, Wc)
                y = np.zeros_like(lin)
                for t in range(self.T):
                    for j in range(Wc):
                        if t - (Wc - 1) + j >= 0:
                            y[t] += w[:, j] * lin[t - (Wc - 1) + j]
                return (y * _sig(y)).reshape(self.T, H, dk)

            q, k, v = conv("q"), conv("k"), conv("v")
            l2 = lambda a: a / np.sqrt((a * a).sum(-1, keepdims=True) + c.eps_kda_l2)  # noqa: E731
            q, k = l2(q) * dk ** -0.5, l2(k)
            a_log = np.log(-m.t(f"blk.{il}.ssm_a").astype(F64))
            g = ((X @ W("ssm_f_a").T) @ W("ssm_f_b").T).reshape(self.T, H, dk)
            g = c.kda_gate_lower_bound * _sig(np.exp(a_log)[None, :, None] * (g + W("ssm_dt", "bias").reshape(H, dk)))
            beta = _sig(X @ W("ssm_beta").T)
            o = np.zeros((self.T, H, dk))
            for h in range(H):
                S = np.zeros((dk, dk))
                for t in range(self.T):
                    S = np.exp(g[t, h])[:, None] * S
                    S = S + np.outer(k[t, h], beta[t, h] * (v[t, h] - S.T @ k[t, h]))
                    o[t, h] = S.T @ q[t, h]
            np.testing.assert_allclose(d[f"kda_scan_out-{il}"].transpose(2, 1, 0), o, rtol=1e-9, atol=1e-13)
            o = _rms(o, W("ssm_norm"), c.eps_rms) * _sig((X @ W("ssm_g").T).reshape(self.T, H, dk))
            out = o.reshape(self.T, H * dk) @ W("attn_output").T
            np.testing.assert_allclose(d[f"kda_out-{il}"].T, out, rtol=1e-9, atol=1e-12)

    # ================================ attention residuals, MoE ================================

    def test_attn_res_layout(self):
        bs = self.cfg.attn_res_block_size
        self.assertNotIn("res_probs_attn-0", self.d)                     # layer 0 reads the embeddings
        for il in range(self.cfg.n_layers):
            if il > 0:
                self.assertEqual(self.d[f"res_probs_attn-{il}"].shape, (math.ceil(il / bs) + 1, self.T))
            self.assertEqual(self.d[f"res_probs_ffn-{il}"].shape, (il // bs + 2, self.T))
            # a block push restarts the partial sum from the attention output alone
            attn = self.d[f"{'kda_out' if self.cfg.is_recr(il) else 'mla_out'}-{il}"]
            want = attn if il % bs == 0 else (self.d[f"l_out-{il - 1}"] + attn).astype(np.float32)
            self.assertTrue(np.array_equal(self.d[f"prefix_sum_attn-{il}"], want))
        self.assertEqual(self.d["res_probs"].shape, (len(range(0, self.cfg.n_layers, bs)) + 1, self.T))
        for name, p in self.d.items():
            if name.startswith("res_probs"):
                np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-6, err_msg=name)
        self.assertEqual(self.d["attn_norm-0"].shape, (self.cfg.n_embd, self.T))

    def test_attn_res_matches_transcription(self):
        """Moonshot's _apply_attn_res, tokens-first: v = [blocks..., partial],
        s_i = sum_d w_d v_id / sqrt(mean_d v_id^2 + eps), out = softmax(s) @ v over the RAW v.
        The block pushed at layer p is the embeddings (p = 0) or the partial sum after layer p-1;
        the attention site of layer il sees the pushes before il, the FFN site those up to il."""
        c, m, d = self.cfg, self.m64, self.d64
        bs, n = c.attn_res_block_size, c.n_layers
        emb = d["inp_embd"].T

        def mix(cands, w):
            v = np.stack(cands)                                           # (n_cand, T, D)
            s = (v * w).sum(-1) / np.sqrt(np.mean(v * v, axis=-1) + c.eps_rms)
            p = np.exp(s - s.max(0)) / np.exp(s - s.max(0)).sum(0)
            return np.einsum("nt,ntd->td", p, v)

        def pushed(upto):
            return [emb if p == 0 else d[f"l_out-{p - 1}"].T for p in range(0, upto + 1, bs)]

        np.testing.assert_allclose(d["attn_norm-0"].T, _rms(emb, m.t("blk.0.attn_norm.weight"), c.eps_rms),
                                   rtol=1e-12, atol=1e-14)               # layer 0 reads the embeddings
        for il in range(n):
            if il > 0:
                want = mix(pushed(il - 1) + [d[f"l_out-{il - 1}"].T], m.t(f"blk.{il}.attn_res_score.weight"))
                np.testing.assert_allclose(d[f"res_mix_attn-{il}"].T, want, rtol=1e-9, atol=1e-12)
            want = mix(pushed(il) + [d[f"prefix_sum_attn-{il}"].T], m.t(f"blk.{il}.ffn_res_score.weight"))
            np.testing.assert_allclose(d[f"res_mix_ffn-{il}"].T, want, rtol=1e-9, atol=1e-12)
        want = mix(pushed(n - 1) + [d[f"l_out-{n - 1}"].T], m.t("output_res_score.weight"))
        np.testing.assert_allclose(d["res_mix"].T, want, rtol=1e-9, atol=1e-12)
        logits = _rms(want, m.t("output_norm.weight"), c.eps_rms) @ m.t("output.weight").T
        np.testing.assert_allclose(self.logits64.T, logits, rtol=1e-9, atol=1e-12)

    def test_ffn_matches_transcription(self):
        """Dense SiTU-GLU (layer 0) and the latent MoE, tokens-first: sigmoid router on the full x,
        pick by probs + bias, weight by the renormalized unbiased probs, experts on z = routed_down(x),
        RMSNorm AFTER the weighted sum, routed_up, plus shared experts on the full x;
        SiTU(g, u) = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)."""
        c, m, d = self.cfg, self.m64, self.d64
        b, lb = c.situ_beta, c.situ_linear_beta

        def situ(g, u):
            return b * np.tanh(g / b) * _sig(g) * lb * np.tanh(u / lb)

        for il in range(c.n_layers):
            X = d[f"ffn_norm-{il}"].T
            W = lambda n, s="weight": m.t(f"blk.{il}.{n}.{s}")  # noqa: E731
            if il < c.n_layer_dense_lead:
                out = situ(X @ W("ffn_gate").T, X @ W("ffn_up").T) @ W("ffn_down").T
            else:
                probs = _sig(X @ W("ffn_gate_inp").T)                     # (T, E)
                pick = probs + W("exp_probs_b", "bias")
                z = X @ W("ffn_routed_down").T                            # (T, latent)
                ge, ue, de = W("ffn_gate_exps"), W("ffn_up_exps"), W("ffn_down_exps")
                r = np.zeros((self.T, c.n_expert_latent))
                for t in range(self.T):
                    idx = np.argsort(-pick[t], kind="stable")[:c.n_expert_used]
                    self.assertEqual(set(idx.tolist()), set(d[f"ffn_moe_topk-{il}"][:, t].tolist()))
                    w = probs[t, idx] / (probs[t, idx].sum() + 1e-20) * c.expert_weights_scale
                    for wi, e in zip(w, idx):
                        r[t] += wi * (de[e] @ situ(ge[e] @ z[t], ue[e] @ z[t]))
                np.testing.assert_allclose(d[f"ffn_moe_out-{il}"].T, r, rtol=1e-9, atol=1e-12)
                r = _rms(r, W("ffn_routed_norm"), c.eps_rms) @ W("ffn_routed_up").T
                out = r + situ(X @ W("ffn_gate_shexp").T, X @ W("ffn_up_shexp").T) @ W("ffn_down_shexp").T
            np.testing.assert_allclose(d[f"ffn_out-{il}"].T, out, rtol=1e-9, atol=1e-12)

    def test_moe_routing(self):
        c = self.cfg
        bias_changed = 0
        for il in self.moe:
            ids = self.d[f"ffn_moe_topk-{il}"]
            probs, biased = self.d[f"ffn_moe_probs-{il}"], self.d[f"ffn_moe_probs_biased-{il}"]
            self.assertEqual(ids.shape, (c.n_expert_used, self.T))
            for t in range(self.T):
                sel = ids[:, t]
                self.assertEqual(len(set(sel.tolist())), c.n_expert_used)
                self.assertTrue(K.topk_sets_agree(biased[:, t], np.argsort(-biased[:, t], kind="stable")[:10], sel))
                np.testing.assert_array_equal(self.d[f"ffn_moe_weights-{il}"][:, t], probs[sel, t])
                bias_changed += int(set(sel.tolist()) != set(np.argsort(-probs[:, t])[:c.n_expert_used].tolist()))
            np.testing.assert_allclose(self.d[f"ffn_moe_weights_norm-{il}"].sum(axis=0), 1.0, atol=1e-6)
            np.testing.assert_allclose(self.d[f"ffn_moe_weights_scaled-{il}"],
                                       self.d[f"ffn_moe_weights_norm-{il}"] * c.expert_weights_scale, rtol=1e-7)
            self.assertEqual(self.d[f"ffn_moe_out-{il}"].shape, (c.n_expert_latent, self.T))
        self.assertGreater(bias_changed, 0)                               # the bias decides real selections


if __name__ == "__main__":
    unittest.main()
