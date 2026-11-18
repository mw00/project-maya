"""Tests for the Kimi K3.1 independent oracle pieces (tools/kimi_k31_moonshot_oracle.py and its helpers).

Unit tests of the indexer patch and the pure-torch fla stand-in need torch; they are skipped without it
(the CI tool tests run with numpy only). The end-to-end selftest additionally needs transformers, einops,
safetensors, Moonshot's modeling files and a fixture:

    KIMI_K31_MOONSHOT_DIR=<dir with modeling_kimi_linear.py> KIMI_K31_FIXTURE=<kimi_k31_synth_gguf.py output> \\
        python -m unittest tools.test_kimi_k31_moonshot_oracle
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import torch
except ImportError:          # CI tool tests run without torch
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class IndexerPatch(unittest.TestCase):
    def setUp(self):
        import kimi_k31_moonshot_indexer as kidx
        self.kidx = kidx
        torch.manual_seed(0)

    def test_select_is_causal_and_counts(self):
        ix = self.kidx.LightningIndexer(16, 8, 2, 4, topk=3, k_norm_eps=1e-5)
        T = 7
        score = torch.randn(1, T, T)
        _, keep = ix.select(score, torch.arange(T))
        for t in range(T):
            sel = keep[0, t].nonzero().flatten()
            self.assertEqual(sel.numel(), min(3, t + 1))
            self.assertTrue(bool((sel <= t).all()))

    def test_negative_scores_never_pick_future_keys(self):
        ix = self.kidx.LightningIndexer(16, 8, 2, 4, topk=4, k_norm_eps=1e-5)
        T = 6
        score = -torch.rand(1, T, T) - 1.0                      # every visible score negative
        _, keep = ix.select(score, torch.arange(T))
        self.assertFalse(bool(keep[0].triu(1).any()))

    def test_ties_break_toward_the_earlier_key(self):
        ix = self.kidx.LightningIndexer(16, 8, 2, 4, topk=2, k_norm_eps=1e-5)
        score = torch.tensor([[[0.5, 0.0, 0.0, 0.0, -1.0]]])    # one query at position 4
        _, keep = ix.select(score, torch.tensor([4]))
        self.assertEqual(keep[0, 0].nonzero().flatten().tolist(), [0, 1])
        # the same row, scored with a longer (future-padded) key range, selects the same keys
        long = torch.cat([score, torch.full((1, 1, 3), 7.0)], dim=-1)
        _, keep2 = ix.select(long, torch.tensor([4]))
        self.assertEqual(keep2[0, 0].nonzero().flatten().tolist(), [0, 1])

    def test_scores_formula(self):
        H, D, hid, ql = 3, 4, 10, 6
        ix = self.kidx.LightningIndexer(hid, ql, H, D, topk=2, k_norm_eps=1e-5)
        x, qr = torch.randn(1, 5, hid), torch.randn(1, 5, ql)
        keys = ix.index_keys(x)
        ref_k = torch.nn.functional.layer_norm(x @ ix.wk.weight.T, (D,), ix.k_norm.weight, ix.k_norm.bias, 1e-5)
        torch.testing.assert_close(keys, ref_k)
        q = (qr @ ix.wq_b.weight.T).view(1, 5, H, D)
        w = (x @ ix.weights_proj.weight.T) / (H * D) ** 0.5
        ref = sum(w[..., h, None] * torch.relu(q[:, :, h] @ keys.transpose(1, 2)) for h in range(H))
        torch.testing.assert_close(ix.scores(x, qr, keys), ref)

    def test_fold_mask(self):
        keep = torch.tensor([[[True, False], [True, True]]])
        add = self.kidx._fold_mask(None, keep, torch.float32)
        self.assertEqual(add.shape, (1, 1, 2, 2))
        self.assertEqual(add[0, 0, 0, 1].item(), float("-inf"))
        self.assertEqual(add[0, 0, 1].tolist(), [0.0, 0.0])
        boolm = self.kidx._fold_mask(torch.ones(1, 1, 2, 3, dtype=torch.bool), keep, torch.float32)
        self.assertEqual(boolm[0, 0].tolist(), [[True, False], [True, True]])
        fl = self.kidx._fold_mask(torch.zeros(1, 1, 2, 2), keep, torch.float32)
        self.assertEqual(fl[0, 0, 0, 1].item(), float("-inf"))

    def test_refusals(self):
        base = {"linear_attn_config": {"full_attn_layers": [4, 8]},
                "dsa_config": {"dsa_layers": [4, 8], "index_topk": 6, "index_head_dim": 32, "index_num_heads": 4,
                               "index_rope_dim": 0, "index_k_norm_type": "layernorm", "index_topk_freq": 1}}
        self.assertEqual(self.kidx.dsa_layers_0idx(base), [3, 7])
        for patch in ({"index_topk_freq": 2}, {"index_rope_dim": 64}, {"index_k_norm_type": "rmsnorm"},
                      {"dsa_layers": [4]}, {"index_topk": 0}):
            bad = {**base, "dsa_config": {**base["dsa_config"], **patch}}
            with self.assertRaises(ValueError):
                self.kidx.dsa_layers_0idx(bad)


@unittest.skipIf(torch is None, "needs torch")
class FlaStandin(unittest.TestCase):
    def setUp(self):
        import kimi_k31_fla_standin as st
        self.st = st
        torch.manual_seed(1)

    def _inputs(self, B=1, T=9, H=2, K=8):
        return dict(q=torch.randn(B, T, H, K), k=torch.randn(B, T, H, K), v=torch.randn(B, T, H, K),
                    g=torch.randn(B, T, H, K), beta=torch.randn(B, T, H))

    def test_kda_split_with_state_equals_one_pass(self):
        x = self._inputs()
        H, K = 2, 8
        kw = dict(A_log=torch.randn(K), dt_bias=torch.randn(H * K), use_qk_l2norm_in_kernel=True,
                  use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5.0,
                  transpose_state_layout=True, output_final_state=True)
        o, s = self.st.chunk_kda(**x, **kw)
        a = {n: t[:, :5] for n, t in x.items()}
        b = {n: t[:, 5:] for n, t in x.items()}
        o1, s1 = self.st.chunk_kda(**a, **kw)
        o2, s2 = self.st.fused_recurrent_kda(**b, initial_state=s1, **kw)
        torch.testing.assert_close(torch.cat([o1, o2], 1), o)
        torch.testing.assert_close(s2, s)
        self.assertEqual(s.shape, (1, H, K, K))

    def test_kda_reads_only_the_first_heads_of_A_log(self):
        x = self._inputs()
        kw = dict(dt_bias=torch.zeros(16), use_gate_in_kernel=True, lower_bound=-5.0)
        a = torch.randn(8)
        o1, _ = self.st.chunk_kda(**x, A_log=a, **kw)
        o2, _ = self.st.chunk_kda(**x, A_log=torch.cat([a[:2], torch.zeros(6)]), **kw)
        torch.testing.assert_close(o1, o2)

    def test_kda_gate_is_lower_bound_sigmoid(self):
        T, H, K = 1, 1, 2
        q = torch.tensor([[[[1.0, 0.0]]]]); k = torch.tensor([[[[1.0, 0.0]]]]); v = torch.tensor([[[[2.0, 3.0]]]])
        g = torch.zeros(1, T, H, K); beta = torch.ones(1, T, H)
        h0 = torch.ones(1, H, K, K)                                   # [N,H,K,V] (no transpose)
        o, s = self.st.chunk_kda(q, k, v, g, beta, A_log=torch.zeros(1), dt_bias=torch.zeros(2),
                                 use_gate_in_kernel=True, lower_bound=-5.0, initial_state=h0,
                                 output_final_state=True, scale=1.0)
        decay = torch.exp(torch.tensor(-5.0 * 0.5))                    # -5 * sigmoid(exp(0) * 0)
        S = h0[0, 0] * decay
        u = (v[0, 0, 0] - S.T @ k[0, 0, 0]) * 1.0                       # beta = 1 (no sigmoid flag)
        S = S + torch.outer(k[0, 0, 0], u)
        torch.testing.assert_close(o[0, 0, 0], S.T @ q[0, 0, 0])

    def test_short_conv_cache(self):
        conv = self.st.ShortConvolution(hidden_size=5, kernel_size=4)
        x = torch.randn(1, 7, 5)
        y, st = conv(x, output_final_state=True)
        ref = torch.nn.functional.silu(torch.nn.functional.conv1d(
            x.transpose(1, 2), conv.weight, padding=3, groups=5)[..., :7]).transpose(1, 2)
        torch.testing.assert_close(y, ref)
        y1, c = conv(x[:, :4], output_final_state=True)
        outs = [y1]
        for t in range(4, 7):
            yt, c = conv(x[:, t:t + 1], cache=c, output_final_state=True)
            outs.append(yt)
        torch.testing.assert_close(torch.cat(outs, 1), y)
        torch.testing.assert_close(c, st)

    def test_gated_norm(self):
        n = self.st.FusedRMSNormGated(6, eps=1e-5, activation="sigmoid")
        with torch.no_grad():
            n.weight.uniform_(0.5, 1.5)
        x, g = torch.randn(3, 6), torch.randn(3, 6)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * n.weight * torch.sigmoid(g)
        torch.testing.assert_close(n(x, g), ref)


def _e2e_ready():
    if torch is None:
        return False
    try:
        import einops  # noqa: F401
        import safetensors  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    d, f = os.environ.get("KIMI_K31_MOONSHOT_DIR"), os.environ.get("KIMI_K31_FIXTURE")
    return bool(d and f and (Path(d) / "modeling_kimi_linear.py").is_file()
                and (Path(f) / "hf-f32" / "model.safetensors").is_file())


@unittest.skipUnless(_e2e_ready(), "needs torch, transformers, einops, safetensors, KIMI_K31_MOONSHOT_DIR "
                                   "and KIMI_K31_FIXTURE")
class EndToEnd(unittest.TestCase):
    def test_selftest_cpu(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run([sys.executable, str(HERE / "kimi_k31_moonshot_oracle.py"), "selftest",
                            "--modeling-dir", os.environ["KIMI_K31_MOONSHOT_DIR"],
                            "--fixture", os.environ["KIMI_K31_FIXTURE"], "--device", "cpu"],
                           capture_output=True, text=True, env=env, timeout=900)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("selftest: all passed", r.stdout)


if __name__ == "__main__":
    unittest.main()
