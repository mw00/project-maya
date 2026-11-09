"""tools/strata/test_parity_compare.py - the B7.4a / M2 parity numbers and gates (tools/strata/parity_compare.py) on
synthetic logits: the KL equals generate.cpp's STRATA_GLM_SCORE_TOPK loop, full and top-k references agree, row offsets
line up, near-ties are found, and the gate applies BUILD-PLAN section 6 (M2, M2-alt) and B7.4a."""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llama_to_score_ref as conv  # noqa: E402
import parity_compare as pc  # noqa: E402
from test_llama_to_score_ref import engine_score_topk  # noqa: E402

V, N, C = 300, 40, 10          # vocab, tokens, STRATA_GLM_SCORE context


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class Fixture:
    def __init__(self, d: pathlib.Path, noise: float, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.ref = (rng.standard_normal((N, V)) * 3.0).astype(np.float32)
        self.ref[5, :] = 0.0
        self.ref[5, 7], self.ref[5, 9] = 4.0, 4.0 - 1e-4            # a near-tie at position 5
        self.ref[C + 3, :] = 0.0
        self.ref[C + 3, 11], self.ref[C + 3, 12] = 5.0, 5.0 - 2e-4  # and one inside the scored range
        self.eng = (self.ref + noise * rng.standard_normal((N, V))).astype(np.float32)
        self.tokens = [int(i) for i in self.ref.argmax(axis=1)]
        self.d = d
        self.ref_bin = d / f"ref{seed}.bin"
        self.ref.tofile(self.ref_bin)
        self.topk = d / f"ref{seed}.topk"
        conv.convert(str(self.ref_bin), V, self.tokens, C, str(self.topk), "topk", 64, None)
        self.dump = d / f"eng{seed}.lsm"        # the engine's STRATA_GLM_SCORE_DUMP: rows C .. N-2
        log_softmax(self.eng[C:N - 1]).astype("<f4").tofile(self.dump)


class Compare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = pathlib.Path(cls.tmp.name)
        cls.fx = Fixture(cls.d, noise=0.05)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_topk_matches_engine_loop(self):
        r = pc.compare(pc.Rows(f"{self.fx.topk}@{C}", V), pc.Rows(f"{self.fx.dump}@{C}", V))
        kl, same, n = engine_score_topk(self.fx.topk, self.fx.eng, self.fx.tokens, C)
        self.assertEqual((r["first"], r["last"], r["n"]), (C, N - 2, n))
        self.assertAlmostEqual(r["kl_mean"], kl, places=7)
        self.assertAlmostEqual(r["top1_pct"], same, places=6)
        self.assertGreater(r["kl_mean"], 0.0)
        self.assertLessEqual(r["kl_p50"], r["kl_p99"])
        self.assertLessEqual(r["kl_p99"], r["kl_max"])

    def test_full_reference_equals_topk(self):
        a = pc.compare(pc.Rows(f"{self.fx.topk}@{C}", V), pc.Rows(f"{self.fx.dump}@{C}", V))
        b = pc.compare(pc.Rows(str(self.fx.ref_bin), V), pc.Rows(f"{self.fx.dump}@{C}", V))   # raw logits from 0
        self.assertEqual((a["first"], a["last"]), (b["first"], b["last"]))
        np.testing.assert_allclose(a["kl"], b["kl"], rtol=1e-4, atol=1e-9)
        self.assertEqual(a["top1_same"], b["top1_same"])

    def test_identity_and_offsets(self):
        r = pc.compare(pc.Rows(str(self.fx.ref_bin), V), pc.Rows(str(self.fx.ref_bin), V))
        self.assertEqual((r["first"], r["last"]), (0, N - 1))
        self.assertLess(r["kl_max"], 1e-9)
        self.assertEqual(r["top1_pct"], 100.0)
        self.assertIn(5, r["near_ties"])
        self.assertEqual(r["top1_miss_outside_ties"], [])
        # a misaligned FIRST shows up as a large KL
        bad = pc.compare(pc.Rows(str(self.fx.ref_bin), V), pc.Rows(f"{self.fx.dump}@{C + 1}", V))
        good = pc.compare(pc.Rows(str(self.fx.ref_bin), V), pc.Rows(f"{self.fx.dump}@{C}", V))
        self.assertGreater(bad["kl_mean"], 10 * good["kl_mean"])

    def test_positions_window_and_ties(self):
        r = pc.compare(pc.Rows(f"{self.fx.topk}@{C}", V), pc.Rows(f"{self.fx.dump}@{C}", V), positions=(C, C + 8))
        self.assertEqual((r["first"], r["last"], r["n"]), (C, C + 7, 8))
        self.assertIn(C + 3, r["near_ties"])
        with self.assertRaises(ValueError):
            pc.compare(pc.Rows(f"{self.fx.topk}@{C}", V), pc.Rows(f"{self.fx.dump}@{C}", V), positions=(100, 120))

    def test_errors(self):
        with self.assertRaises(ValueError):
            pc.Rows(str(self.fx.ref_bin), V + 1)
        with self.assertRaises(ValueError):
            pc.compare(pc.Rows(str(self.fx.ref_bin), V), pc.Rows(f"{self.fx.topk}@{C}", V))

    def test_cli(self):
        out = self.d / "c.json"
        rc = quiet(pc.main, ["compare", "--ref", f"{self.fx.topk}@{C}", "--test", f"{self.fx.dump}@{C}",
                             "--n-vocab", str(V), "--name", "chat", "--out", str(out)])
        self.assertEqual(rc, 0)
        j = json.loads(out.read_text())
        self.assertEqual(j["name"], "chat")
        self.assertEqual(quiet(pc.main, ["compare", "--ref", "/nonexistent.topk", "--test", str(self.fx.dump)]), 2)


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = pathlib.Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def res(self, name, kls, same=None):
        n = len(kls)
        same = n if same is None else same
        p = self.d / f"{name}.json"
        p.write_text(json.dumps({"name": name, "kl": kls, "top1_same": same, "n": n}))
        return p.name

    def gate(self, m):
        mp = self.d / "manifest.json"
        mp.write_text(json.dumps(m))
        return pc.gate(json.loads(mp.read_text()), self.d)

    def test_b74a(self):
        floor = self.res("floor", [0.004] * 100, 99)
        ok = self.gate({"gate": "B7.4a", "floor": [floor], "a_vs_r": [self.res("a1", [0.007] * 100, 97)]})
        self.assertEqual(ok["verdict"], "PASS", ok)
        # max(2F, 0.01) = 0.01: 0.012 fails (a); 94% fails top-1
        bad = self.gate({"gate": "B7.4a", "floor": [floor], "a_vs_r": [self.res("a2", [0.012] * 100, 94)]})
        self.assertEqual(bad["verdict"], "FAIL")
        self.assertEqual(sorted(bad["failed"]), ["a", "top1"])
        # a high floor raises the bound to 2F
        hi = self.gate({"gate": "B7.4a", "floor": [self.res("f2", [0.008] * 10)], "a_vs_r": ["a2.json"]})
        self.assertEqual(hi["failed"], ["top1"])

    def test_m2(self):
        extra = {}
        for cid in "efhi":
            p = self.d / f"x{cid}.json"
            p.write_text(json.dumps({"pass": True} if cid != "h" else {"gate": {"pass": True}}))
            extra[cid] = p.name
        base = {"gate": "M2", "floor": [self.res("mf", [0.003] * 200, 196)],
                "a_vs_r": [self.res("ma", [0.004] * 198 + [0.1, 0.12], 195)],
                "slow_vs_fast": [self.res("sf", [0.001] * 50)], "prod_vs_strict": [self.res("ps", [0.002] * 50)],
                "extra": extra}
        g = self.gate(base)
        self.assertEqual(g["verdict"], "PASS", g)
        md = pc.gate_markdown(g)
        self.assertIn("M2 gate: **PASS**", md)
        # (c): 195/200 = 97.5% vs floor agreement 98% - 1.0 = 97.0 passes; 193 fails
        g2 = self.gate(dict(base, a_vs_r=[self.res("mb", [0.004] * 200, 193)]))
        self.assertEqual(g2["failed"], ["c"])
        # (b) p99 > 0.15
        g3 = self.gate(dict(base, a_vs_r=[self.res("mc", [0.001] * 190 + [0.5] * 10, 200)]))
        self.assertIn("b", g3["failed"])
        # (d), (g)
        g4 = self.gate(dict(base, slow_vs_fast=[self.res("sf2", [0.003] * 50)],
                            prod_vs_strict=[self.res("ps2", [0.006] * 50)]))
        self.assertEqual(sorted(g4["failed"]), ["d", "g"])
        # an extra that failed, one not run
        fe = self.d / "fe.json"
        fe.write_text(json.dumps({"pass": False}))
        g5 = self.gate(dict(base, extra={"e": fe.name}))
        self.assertEqual(g5["verdict"], "FAIL")
        self.assertEqual(g5["failed"], ["e"])
        self.assertEqual(sorted(g5["missing"]), ["f", "h", "i"])
        g6 = self.gate(dict(base, extra={}))
        self.assertEqual(g6["verdict"], "INCOMPLETE")

    def test_m2_alt_and_pooling(self):
        # pooled over prompts: one bad prompt among good ones moves the pooled mean
        g = self.gate({"gate": "M2-alt", "floor": [self.res("af", [0.002] * 100)],
                       "a_vs_r": [self.res("p1", [0.002] * 100), self.res("p2", [0.03] * 100)]})
        self.assertAlmostEqual(g["a_vs_r"]["kl_mean"], 0.016)
        self.assertIn("a", g["failed"])
        self.assertIn("R_cuda", g["criteria"][0]["what"])

    def test_cli_exit(self):
        mp = self.d / "m.json"
        mp.write_text(json.dumps({"gate": "B7.4a", "a_vs_r": [self.res("c1", [0.001] * 10)]}))
        out = self.d / "v.json"
        self.assertEqual(quiet(pc.main, ["gate", str(mp), "--json", str(out), "--md", str(self.d / "v.md")]), 0)
        self.assertEqual(json.loads(out.read_text())["verdict"], "PASS")
        mp.write_text(json.dumps({"gate": "M3"}))
        self.assertEqual(quiet(pc.main, ["gate", str(mp)]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
