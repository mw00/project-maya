"""tools/strata/test_seam_bisect.py - seam_bisect.py (B7.4b triage, M2 (e), B7.6) on dumps of the oracle's SMALL run
written in the three on-disk forms: the oracle's .npy (T on the last axis), a llama.cpp-style directory (one call,
every token, ids as .i32) and an engine-style directory (the last token only, ids as int32 bytes in .f32)."""
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))
import seam_bisect as sb  # noqa: E402

try:
    import glm_dsa_synth_gguf as synth  # noqa: E402
    from glm_dsa import GlmDsa, Options  # noqa: E402
    _ERR = None
except ImportError as e:
    _ERR = e

T = 20


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a)


def column(arr: np.ndarray, seam: str, t: int, width: int) -> np.ndarray:
    if seam == "top_k":
        cells = np.nonzero(arr[t])[0].astype(np.int32)
        out = np.full(width, -1, dtype=np.int32)
        out[: cells.size] = cells
        return out
    if seam == "indexer_score":
        return np.asarray(arr[t], dtype=np.float32)
    v = np.asarray(arr[..., t]).ravel(order="F")
    return v.astype(np.int32) if seam == "ffn_moe_topk" else v.astype(np.float32)


class Bisect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _ERR is not None:
            raise unittest.SkipTest(f"gguf-py not importable: {_ERR}")
        cls.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(cls.tmp.name)
        gguf = d / "small.gguf"
        synth.write(gguf, "SMALL", synth.DEFAULT_SEED, None)
        m = GlmDsa(gguf, Options(mode="hf"))
        m.collect = True
        m.forward([(i * 37 + 11) % m.cfg.n_vocab for i in range(T)])
        cls.top_k = m.cfg.top_k
        dumps = dict(m.dumps)
        m.close()
        cls.ora, cls.lla, cls.eng = d / "oracle", d / "llama", d / "engine"
        for p in (cls.ora, cls.lla, cls.eng):
            p.mkdir()
        cls.names = []
        for key, arr in dumps.items():
            seam, _ = sb.split_name(key)
            if seam in ("indexer_score",):
                continue
            np.save(cls.ora / f"{key}.npy", arr)
            cols = [column(arr, seam, t, cls.top_k) for t in range(T)]
            ids = seam in sb.ID_SEAMS
            np.concatenate(cols).tofile(cls.lla / f"{key}.{'i32' if ids else 'f32'}")
            cols[-1].tofile(cls.eng / f"{key}.f32")
            cls.names.append(key)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_identical_runs(self):
        r = sb.bisect(self.eng, self.lla, "1", "auto")
        self.assertIsNone(r["first_divergence"], [x for x in r["rows"] if not x["ok"]][:3])
        self.assertGreater(r["compared"], 50)
        r2 = sb.bisect(self.eng, self.ora, "1", "auto")
        self.assertIsNone(r2["first_divergence"], [x for x in r2["rows"] if not x["ok"]][:3])
        r3 = sb.bisect(self.lla, self.ora, str(T), "auto", token=5)
        self.assertIsNone(r3["first_divergence"])

    def test_first_divergence_in_forward_order(self):
        bad = pathlib.Path(self.tmp.name) / "bad"
        bad.mkdir()
        for p in self.eng.iterdir():
            (bad / p.name).write_bytes(p.read_bytes())
        for key in ("attn_out-3", "l_out-5"):
            v = np.fromfile(bad / f"{key}.f32", dtype="<f4")
            (v * 1.2).astype("<f4").tofile(bad / f"{key}.f32")
        r = sb.bisect(bad, self.lla, "1", "auto")
        self.assertEqual(r["first_divergence"], "attn_out-3")
        rows = {x["seam"]: x for x in r["rows"]}
        self.assertTrue(rows["kqv_out-3"]["ok"])
        self.assertFalse(rows["l_out-5"]["ok"])
        # a wrong selection is an id-seam divergence
        sel = np.fromfile(bad / "top_k-2.f32", dtype="<i4")
        sel[sel >= 0] = (sel[sel >= 0] + 1) % T
        sel.tofile(bad / "top_k-2.f32")
        r = sb.bisect(bad, self.lla, "1", "auto")
        self.assertEqual(r["first_divergence"], "top_k-2")
        # --layers restricts
        r = sb.bisect(bad, self.lla, "1", "auto", layers=(4, 8))
        self.assertEqual(r["first_divergence"], "l_out-5")

    def test_order(self):
        keys = ["l_out-1", "attn_out-1", "result_output", "attn_norm-2", "top_k-1", "result_norm"]
        self.assertEqual(sorted(keys, key=sb.order_key),
                         ["top_k-1", "attn_out-1", "l_out-1", "attn_norm-2", "result_norm", "result_output"])

    def test_cli(self):
        out = pathlib.Path(self.tmp.name) / "b.json"
        self.assertEqual(quiet(sb.main, ["bisect", "--a", str(self.eng), "--b", str(self.lla), "--json", str(out)]), 0)
        self.assertIsNone(json.loads(out.read_text())["first_divergence"])
        self.assertEqual(quiet(sb.main, ["bisect", "--a", "/nonexistent", "--b", str(self.lla)]), 2)


class Overlap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.a, self.b = self.d / "a", self.d / "b"
        self.a.mkdir()
        self.b.mkdir()
        W, P = 16, 40                       # top-k width, positions 0..39
        rng = np.random.default_rng(3)
        for il in (0, 2):
            ra, rb = [], []
            for p in range(P):
                cells = np.sort(rng.choice(1000, W, replace=False)).astype(np.int32)
                other = cells.copy()
                if il == 2 and p == 30:
                    other[:2] = [2000, 2001]           # 14 / 16 = 87.5%
                ra.append(cells)
                rb.append(other[rng.permutation(W)])   # order does not matter
            np.concatenate(ra).tofile(self.a / f"top_k-{il}.f32")
            # B starts 5 positions later (its row 0 is position 5)
            np.concatenate(rb[5:]).tofile(self.b / f"top_k-{il}.i32")
            moe_a = np.asarray([[1, 2, 3, 4]] * P, dtype=np.int32)
            moe_b = moe_a.copy()
            moe_b[3] = [1, 2, 3, 9]
            moe_a.tofile(self.a / f"ffn_moe_topk-{il}.f32")
            moe_b.tofile(self.b / f"ffn_moe_topk-{il}.i32")
        self.W = W

    def tearDown(self):
        self.tmp.cleanup()

    def test_overlap(self):
        r = sb.set_compare(self.a, self.b, "top_k", [0, 2], (10, 40), self.W, 0, 5, 99.5, {})
        self.assertTrue(r["layers"]["0"]["pass"])
        self.assertEqual(r["layers"]["0"]["mean_pct"], 100.0)
        self.assertFalse(r["pass"])
        self.assertEqual(r["layers"]["2"]["below_gate"], [30])
        self.assertAlmostEqual(r["layers"]["2"]["min_pct"], 87.5)
        r2 = sb.set_compare(self.a, self.b, "top_k", [0, 2], (10, 40), self.W, 0, 5, 99.5, {2: {30}})
        self.assertTrue(r2["pass"])
        self.assertEqual(r2["layers"]["2"]["excluded_ties"], 1)
        with self.assertRaises(ValueError):
            sb.set_compare(self.a, self.b, "top_k", [0], (100, 120), self.W, 0, 5, 99.5, {})

    def test_moe_and_cli(self):
        ties = self.d / "ties.json"
        ties.write_text(json.dumps({"0": [3], "2": [3]}))
        self.assertEqual(quiet(sb.main, ["moe", "--a", str(self.a), "--b", str(self.b), "--layers", "0,2",
                                         "--positions", "0:8", "--n-used", "4"]), 1)
        out = self.d / "m.json"
        self.assertEqual(quiet(sb.main, ["moe", "--a", str(self.a), "--b", str(self.b), "--layers", "0,2",
                                         "--positions", "0:8", "--n-used", "4", "--ties", str(ties),
                                         "--json", str(out)]), 0)
        self.assertTrue(json.loads(out.read_text())["pass"])
        self.assertEqual(quiet(sb.main, ["overlap", "--a", str(self.a), "--b", str(self.b), "--layers", "0",
                                         "--positions", "10:40", "--b-first", "5", "--top-k", str(self.W)]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
