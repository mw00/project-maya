"""tools/test_kimi_k31_golden.py - the K3.1 router and SiTU golden vectors (tools/kimi_k31_golden.py).

numpy only, no GPU, a few seconds:
  * the golden router IS the oracle's: on the synthetic fixture, kimi_k31_golden.route() applied to the dumped
    router logits and the layer's bias equals moe_ffn's own probs, biased scores, top-k ids (in order) and weights,
    bit for bit, in every MoE layer; golden SiTU equals ref/kimi_k31.py situ bit for bit and an independent float64
    transcription of Moonshot's formula;
  * the cases cover what a kernel can get wrong: bitwise ties straddling the cut (decided by the lower id), an
    all-equal token (ids 0..k-1), bias that changes the set, all-tiny sigmoids (the +1e-20 vs 2^-14 split),
    sigmoids that round to exactly 1.0f, +-inf logits, a non-unit routed scale, and SiTU over its whole range;
  * a plain float32 kernel (the emulation) passes both gates, with the budgets measured from it;
  * every wrong reading fails a gate: softmax, weights from the biased score, no renormalisation, a 2^-14 clamp,
    higher-id ties, picks by the unbiased score, 8 picks, the scale dropped, renormalising over all experts; SwiGLU,
    swapped betas, a linear up branch, no sigmoid, no gate tanh, a 2^-11 tanh approximation;
  * the files are deterministic and read back to the same arrays.

    python -m unittest tools.test_kimi_k31_golden
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

import kimi_k31_golden as G  # noqa: E402
import kimi_k31_synth_gguf as synth  # noqa: E402
import kimi_k31 as K  # noqa: E402  (ref/kimi_k31.py)

F32, F64 = np.float32, np.float64


def read_raw(d: pathlib.Path) -> dict:
    """Read write_raw_dumps' layout back: manifest dims are the file's own (outer first) order; undo the reversal."""
    out = {}
    for line in (d / "manifest.txt").read_text().splitlines():
        name, *dims = line.split()
        dims = [int(x) for x in dims]
        dt = np.int32 if name.endswith(".ids") else F32
        a = np.fromfile(d / f"{name}.f32", dtype=dt).reshape(dims)
        out[name] = np.ascontiguousarray(a.transpose(tuple(range(a.ndim - 1, -1, -1))))
    return out


# ------------------------------------------------------------------ wrong readings (each must fail a gate)

def _sel(scores, k, high_first=False):
    if high_first:
        E = scores.shape[0]
        return (E - 1 - np.argsort(-scores[::-1].astype(F64), kind="stable")[:k]).astype(np.int64)
    return np.argsort(-scores.astype(F64), kind="stable")[:k].astype(np.int64)


def wrong_route(kind, logits, bias, k, scale):
    logits = np.asarray(logits, dtype=F32)
    E, T = logits.shape
    p64 = K.sigmoid(logits, F64)
    if kind == "softmax":
        with np.errstate(invalid="ignore"):
            z = logits.astype(F64) - logits.astype(F64).max(axis=0, keepdims=True)
            p64 = np.exp(z) / np.exp(z).sum(axis=0, keepdims=True)
        p64 = np.nan_to_num(p64)
    p = p64.astype(F32)
    biased = (p + bias[:, None]).astype(F32)
    kk = 8 if kind == "k8" else k
    ids = np.empty((kk, T), dtype=np.int64)
    w = np.empty((kk, T), dtype=F32)
    for t in range(T):
        src = p[:, t] if kind == "unbiased_picks" else biased[:, t]
        sel = _sel(src, kk, high_first=(kind == "high_id_ties"))
        ids[:, t] = sel
        x = (biased[sel, t] if kind == "biased_weights" else p[sel, t]).astype(F64)
        if kind == "no_renorm":
            pass
        elif kind == "clamp_2m14":
            x = x / max(x.sum(), 2.0 ** -14)
        elif kind == "renorm_all":
            x = x / (p[:, t].astype(F64).sum() + G.RENORM_EPS)
        elif kind == "no_eps":
            x = x / x.sum()
        elif kind == "max_sum_eps":
            x = x / max(x.sum(), G.RENORM_EPS)
        else:
            x = x / (x.sum() + G.RENORM_EPS)
        w[:, t] = x * (1.0 if kind == "no_scale" else scale)
    return ids, w


def wrong_situ(kind, g, u):
    g64, u64 = np.asarray(g, dtype=F64), np.asarray(u, dtype=F64)
    with np.errstate(over="ignore", invalid="ignore"):
        sg = 1.0 / (1.0 + np.exp(-g64))
        if kind == "swiglu":
            r = g64 * sg * u64
        elif kind == "swapped_betas":
            r = 25.0 * np.tanh(g64 / 25.0) * sg * 4.0 * np.tanh(u64 / 4.0)
        elif kind == "linear_up":
            r = 4.0 * np.tanh(g64 / 4.0) * sg * u64
        elif kind == "no_sigmoid":
            r = 4.0 * np.tanh(g64 / 4.0) * 25.0 * np.tanh(u64 / 25.0)
        elif kind == "no_gate_tanh":
            r = g64 * sg * 25.0 * np.tanh(u64 / 25.0)
        elif kind == "approx_tanh":          # tanh.approx.f32-like: ~2^-11 relative error
            def at(x):
                t = np.tanh(x)
                return t * (1.0 + 2.0 ** -11 * np.sign(np.sin(1e3 * x)))
            r = 4.0 * at(g64 / 4.0) * sg * 25.0 * at(u64 / 25.0)
        else:
            raise ValueError(kind)
        return r.astype(F32)


class GoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-golden-")
        env = os.environ.get("KIMI_K31_GGUF")
        if env:
            cls.gguf = pathlib.Path(env)
        else:
            cls.gguf = pathlib.Path(cls.tmp.name) / "tiny.gguf"
            vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
            md, ts = synth.to_gguf(vals, synth.TINY)
            synth.write_gguf(cls.gguf, md, ts)
        cls.m = K.KimiK31(cls.gguf, cache_weights=True)
        cls.cfg = cls.m.cfg
        cls.m.forward(synth.sample_tokens(synth.TINY, synth.SEED))
        cls.d = dict(cls.m.dumps)
        cls.out = pathlib.Path(cls.tmp.name) / "golden"
        cls.meta = G.write(cls.out)
        cls.files = read_raw(cls.out)
        cls.rcases = G.route_cases()
        cls.scases = G.situ_cases()

    @classmethod
    def tearDownClass(cls):
        cls.m.close()
        cls.tmp.cleanup()

    # -------------------------------------------------------------- the golden functions are the oracle's
    def test_route_equals_oracle_moe_ffn(self):
        c = self.cfg
        layers = list(range(c.n_layer_dense_lead, c.n_layers))
        self.assertTrue(layers)
        for il in layers:
            logits = self.d[f"ffn_moe_logits-{il}"]
            bias = self.m.layer_t(il, "exp_probs_b", "bias")
            r = G.route(logits, bias, c.n_expert_used, c.expert_weights_scale, c.expert_weights_norm,
                        c.eps_router_renorm)
            np.testing.assert_array_equal(r["probs"], self.d[f"ffn_moe_probs-{il}"], err_msg=f"probs {il}")
            np.testing.assert_array_equal(r["biased"], self.d[f"ffn_moe_probs_biased-{il}"], err_msg=f"biased {il}")
            np.testing.assert_array_equal(r["ids"], self.d[f"ffn_moe_topk-{il}"], err_msg=f"ids {il}")
            for key, seam in (("w_raw", "ffn_moe_weights"), ("w_norm", "ffn_moe_weights_norm"),
                              ("w", "ffn_moe_weights_scaled")):
                np.testing.assert_array_equal(r[key], self.d[f"{seam}-{il}"], err_msg=f"{key} {il}")

    def test_situ_equals_oracle_and_moonshot_formula(self):
        for name, (g, u) in self.scases.items():
            with np.errstate(over="ignore"):
                ref = K.situ(g, u, 4.0, 25.0, F32)
            np.testing.assert_array_equal(G.situ(g, u), ref, err_msg=name)
            # independent float64 transcription of configuration_kimi_k3's SiTU: 4 tanh(g/4) sigma(g) 25 tanh(u/25)
            g64, u64 = g.astype(F64), u.astype(F64)
            with np.errstate(over="ignore", invalid="ignore"):
                ind = 4.0 * np.tanh(g64 / 4.0) * (np.exp(-np.logaddexp(0.0, -g64))) * 25.0 * np.tanh(u64 / 25.0)
                r64 = G.situ(g, u, dt=F64)
                fin = np.isfinite(r64) & np.isfinite(ind)
                np.testing.assert_allclose(r64[fin], ind[fin], rtol=1e-12, atol=1e-300, err_msg=name)
                np.testing.assert_array_equal(np.isinf(r64), np.isinf(ind), err_msg=name)

    def test_released_constants(self):
        self.assertEqual((G.E_RELEASED, G.K_RELEASED, G.SITU_BETA, G.SITU_LINEAR_BETA, G.RENORM_EPS),
                         (896, 16, 4.0, 25.0, 1e-20))
        cfg = K.KimiK31Config({"general.architecture": "kimi-k31"})
        self.assertEqual((cfg.n_expert_used, cfg.situ_beta, cfg.situ_linear_beta, cfg.eps_router_renorm,
                          cfg.expert_weights_scale), (16, 4.0, 25.0, 1e-20, 1.0))

    # -------------------------------------------------------------- coverage
    def test_cases_cover_the_traps(self):
        f = self.files
        k = G.K_RELEASED
        # ties: the 9-way block straddles the cut and the lower ids win it
        lg, bs = f["route.ties.logits"], f["route.ties.bias"]
        ids = f["route.ties.ids"]
        for t in range(8):
            col = lg[:, t]
            blk = np.flatnonzero(col == F32(1.25))
            self.assertEqual(len(blk), 9)
            chosen = sorted(set(ids[:, t]) & set(blk))
            self.assertEqual(chosen, sorted(blk)[:4], f"token {t}: the lower ids win the tie")
        self.assertEqual(sorted(ids[:, 9]), list(range(k)), "all equal -> ids 0..15")
        self.assertEqual(list(ids[:, 9]), list(range(k)), "and in ascending id order")
        self.assertTrue(np.all(f["route.ties.margin"] == 0))
        # bias changes the set in every bias_flip token, and weights still come from unbiased probs
        bf = self.meta["router"]["cases"]["bias_flip"]
        self.assertEqual(bf["bias_changed_set"], bf["T"])
        # tiny: ordinary weights (sum 1) although every prob is < 1e-8
        w = f["route.tiny.w"].astype(F64)
        self.assertTrue(np.all(f["route.tiny.w_raw"] < 1e-8))
        np.testing.assert_allclose(w.sum(axis=0), 1.0, rtol=1e-6)
        # saturate: probs exactly 1.0f; inf: exact 0 and 1
        self.assertTrue(np.all(f["route.saturate.probs"] == F32(1.0)))
        self.assertTrue(np.any(f["route.saturate.margin"] == 0))
        pr = f["route.inf.probs"]
        self.assertTrue(np.any(pr == F32(1.0)) and np.any(pr == F32(0.0)))
        self.assertEqual(self.meta["router"]["cases"]["scaled"]["scale"], 2.5)
        np.testing.assert_allclose(f["route.scaled.w"].astype(F64).sum(axis=0), 2.5, rtol=1e-6)
        # SiTU: both saturations, signed zeros, denormal inputs, and inf
        g = f["situ.edges.g"]
        self.assertTrue(np.any(np.signbit(g) & (g == 0)) and np.any((g != 0) & (np.abs(g) < 1.2e-38)))
        self.assertTrue(np.any(np.isinf(f["situ.inf.g"])) and np.any(np.isinf(f["situ.inf.u"])))
        self.assertGreater(f["situ.gate_sweep.g"].max(), 90)
        self.assertLess(f["situ.gate_sweep.g"].min(), -90)
        self.assertGreater(np.abs(f["situ.up_sweep.u"]).max(), 300)

    # -------------------------------------------------------------- the gates
    def gold(self, name):
        lg, bs, k, scale = self.rcases[name]
        return G.gold_route(lg, bs, k, scale), (lg, bs, k, scale)

    def test_float32_kernel_passes(self):
        rb, sb = self.meta["router"]["budget_ulp"], self.meta["situ"]["budget_ulp"]
        self.assertLessEqual(rb, 16)
        self.assertLessEqual(sb, 16)
        for name in self.rcases:
            gold, (lg, bs, k, scale) = self.gold(name)
            ids, w = G.emulate_route(lg, bs, k, scale)
            ok, st = G.check_route(gold, ids, w, rb)
            self.assertTrue(ok, f"route {name}: {st}")
        for name, (g, u) in self.scases.items():
            ok, st = G.check_situ({"ref64": G.situ(g, u, dt=F64), "g": g}, G.emulate_situ(g, u), sb)
            self.assertTrue(ok, f"situ {name}: {st}")

    def test_golden_files_pass_their_own_gate(self):
        rb = self.meta["router"]["budget_ulp"]
        for name in self.rcases:
            gold, _ = self.gold(name)
            ok, st = G.check_route(gold, self.files[f"route.{name}.ids"], self.files[f"route.{name}.w"], rb)
            self.assertTrue(ok, f"{name}: {st}")
            self.assertEqual(st["worst_ulp"] <= 0.5, True, f"{name}: golden f32 weights are the rounded f64 ({st})")

    def test_wrong_router_readings_fail(self):
        rb = self.meta["router"]["budget_ulp"]
        kinds = ["softmax", "biased_weights", "no_renorm", "clamp_2m14", "high_id_ties", "unbiased_picks", "k8",
                 "no_scale", "renorm_all", "no_eps", "max_sum_eps"]
        for kind in kinds:
            failed_in = []
            for name in self.rcases:
                gold, (lg, bs, k, scale) = self.gold(name)
                ids, w = wrong_route(kind, lg, bs, k, scale)
                ok, _ = G.check_route(gold, ids, w, rb)
                if not ok:
                    failed_in.append(name)
            self.assertTrue(failed_in, f"wrong reading {kind!r} passed every case")
        # the targeted cases catch their bug
        for kind, case in (("clamp_2m14", "tiny"), ("high_id_ties", "ties"), ("unbiased_picks", "bias_flip"),
                           ("no_scale", "scaled"), ("no_eps", "eps_dom"), ("max_sum_eps", "eps_dom")):
            gold, (lg, bs, k, scale) = self.gold(case)
            ids, w = wrong_route(kind, lg, bs, k, scale)
            self.assertFalse(G.check_route(gold, ids, w, rb)[0], f"{kind} must fail on {case}")

    def test_wrong_situ_readings_fail(self):
        sb = self.meta["situ"]["budget_ulp"]
        for kind in ["swiglu", "swapped_betas", "linear_up", "no_sigmoid", "no_gate_tanh", "approx_tanh"]:
            failed_in = [name for name, (g, u) in self.scases.items()
                         if not G.check_situ({"ref64": G.situ(g, u, dt=F64), "g": g}, wrong_situ(kind, g, u), sb)[0]]
            self.assertTrue(failed_in, f"wrong reading {kind!r} passed every case")

    def test_truncated_to_8_experts_fails(self):
        """The plan's mutation: a kernel that keeps only the first 8 picks (and renormalises over them)."""
        rb = self.meta["router"]["budget_ulp"]
        for name in ("rand", "decode1", "window16"):
            gold, (lg, bs, k, scale) = self.gold(name)
            ids, w = G.emulate_route(lg, bs, k, scale)
            ids8 = ids.copy()
            ids8[8:] = ids8[:8]                     # 16 slots, only 8 distinct experts
            self.assertFalse(G.check_route(gold, ids8, w, rb)[0], name)
            w8 = w.copy()
            w8[8:] = 0                              # 8 experts kept, weights of the rest zeroed
            self.assertFalse(G.check_route(gold, ids, w8, rb)[0], name)

    # -------------------------------------------------------------- files
    def test_files_deterministic_and_round_trip(self):
        out2 = pathlib.Path(self.tmp.name) / "golden2"
        meta2 = G.write(out2)
        self.assertEqual(meta2["files"], self.meta["files"])
        j = json.loads((self.out / "golden.json").read_text())
        self.assertEqual(j["files"], self.meta["files"])
        arrays, _ = G.build()
        self.assertEqual(set(arrays), set(self.files))
        for n, a in arrays.items():
            want = a.astype(np.int32) if a.dtype == np.int64 else a.astype(F32)
            np.testing.assert_array_equal(self.files[n], want, err_msg=n)
        for n in ("route.rand.logits", "route.rand.ids", "situ.random.out"):
            self.assertIn(n, self.files)
        self.assertEqual(self.files["route.rand.logits"].shape, (896, 64))
        self.assertEqual(self.files["route.rand.ids"].shape, (16, 64))


if __name__ == "__main__":
    unittest.main()
