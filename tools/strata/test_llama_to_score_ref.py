"""tools/strata/test_llama_to_score_ref.py - the score-reference converter against a Python port of the engine's
MAYA_GLM_SCORE comparison (src/program/generate.cpp), on synthetic logits (B7.3)."""
from __future__ import annotations

import contextlib
import io
import pathlib
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llama_to_score_ref as conv  # noqa: E402


def engine_score_topk(ref_path, logits, tokens, c):
    """generate.cpp's MAYA_GLM_SCORE_TOPK loop, transcribed: mean KL(ref top-k + rest || engine), same-top %."""
    with open(ref_path, "rb") as f:
        k = struct.unpack("<i", f.read(4))[0]
        kl, same, n = 0.0, 0, 0
        for i in range(c, len(tokens) - 1):
            lg = logits[i].astype(np.float64)
            mx = lg.max()
            ls = lg - (mx + np.log(np.exp(lg - mx).sum()))
            am = int(np.argmax(lg))
            tid = np.frombuffer(f.read(4 * k), dtype="<i4")
            tlp = np.frombuffer(f.read(4 * k), dtype="<f4").astype(np.float64)
            p = np.exp(tlp)
            kk = float((p * (tlp - ls[tid])).sum())
            pr, pt = p.sum(), np.exp(ls[tid]).sum()
            rr, rt = max(1.0 - pr, 1e-12), max(1.0 - pt, 1e-12)
            if pr < 1.0:
                kk += rr * (np.log(rr) - np.log(rt))
            kl += kk
            same += int(tid[0]) == am
            n += 1
        assert f.read() == b""
    return kl / n, 100.0 * same / n, n


def engine_score_full(ref_path, logits, tokens, c, vocab):
    rl_all = np.fromfile(ref_path, dtype="<f4").reshape(-1, vocab).astype(np.float64)
    kl, same, n = 0.0, 0, 0
    for j, i in enumerate(range(c, len(tokens) - 1)):
        lg = np.full(vocab, -1e30)
        lg[:logits.shape[1]] = logits[i]
        mx = lg.max()
        ls = lg - (mx + np.log(np.exp(lg - mx).sum()))
        rl = rl_all[j]
        p = np.exp(rl)
        m = p > 0
        kl += float((p[m] * (rl[m] - ls[m])).sum())
        same += int(np.argmax(rl)) == int(np.argmax(lg))
        n += 1
    return kl / n, 100.0 * same / n, n


class ConverterTest(unittest.TestCase):
    V, N, C = 1000, 40, 8

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.td.name)
        rng = np.random.default_rng(3)
        self.logits = (rng.standard_normal((self.N, self.V)) * 3).astype(np.float32)
        self.tokens = [int(x) for x in rng.integers(0, self.V, self.N)]
        # a greedy continuation: after the context every next token is the reference's argmax
        for i in range(self.C, self.N - 1):
            self.tokens[i + 1] = int(np.argmax(self.logits[i]))
        self.dump = self.d / "ref.bin"
        self.logits.tofile(self.dump)
        (self.d / "tokens.txt").write_text(" ".join(map(str, self.tokens)))

    def tearDown(self):
        self.td.cleanup()

    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = conv.main(["--dump", str(self.dump), "--n-vocab", str(self.V), "--tokens", str(self.d / "tokens.txt"),
                            "--context", str(self.C), *args])
        return rc, out.getvalue()

    def test_topk_self_kl_zero_and_layout(self):
        out = self.d / "ref.topk"
        rc, msg = self.run_main("--out", str(out), "--topk", "64")
        self.assertEqual(rc, 0)
        self.assertIn("top1 100.00%", msg)
        self.assertEqual(out.stat().st_size, 4 + (self.N - 1 - self.C) * 64 * 8)
        kl, same, n = engine_score_topk(out, self.logits, self.tokens, self.C)
        self.assertEqual(n, self.N - 1 - self.C)
        self.assertLess(abs(kl), 1e-6)
        self.assertEqual(same, 100.0)
        # ids best first, log-probs descending
        with open(out, "rb") as f:
            f.read(4)
            ids = np.frombuffer(f.read(256), "<i4")
            lp = np.frombuffer(f.read(256), "<f4")
        self.assertEqual(int(ids[0]), int(np.argmax(self.logits[self.C])))
        self.assertTrue(np.all(np.diff(lp) <= 0))

    def test_topk_kl_matches_full_vocab_kl_when_k_is_vocab(self):
        out = self.d / "ref.all"
        self.assertEqual(self.run_main("--out", str(out), "--topk", str(self.V))[0], 0)
        noisy = self.logits + np.random.default_rng(5).standard_normal(self.logits.shape).astype(np.float32) * 0.3
        kl_k, _, _ = engine_score_topk(out, noisy, self.tokens, self.C)
        full = self.d / "ref.full"
        self.assertEqual(self.run_main("--out", str(full), "--format", "full")[0], 0)
        kl_f, _, _ = engine_score_full(full, noisy, self.tokens, self.C, self.V)
        self.assertGreater(kl_f, 0.01)
        self.assertAlmostEqual(kl_k, kl_f, places=5)

    def test_top64_rest_bucket_is_bounded_by_full_kl(self):
        out = self.d / "ref.topk"
        self.run_main("--out", str(out))
        noisy = self.logits + np.random.default_rng(6).standard_normal(self.logits.shape).astype(np.float32) * 0.3
        kl_k, _, _ = engine_score_topk(out, noisy, self.tokens, self.C)
        full = self.d / "ref.full"
        self.run_main("--out", str(full), "--format", "full")
        kl_f, _, _ = engine_score_full(full, noisy, self.tokens, self.C, self.V)
        self.assertGreater(kl_k, 0.0)
        self.assertLessEqual(kl_k, kl_f + 1e-9)        # a coarser partition never has a larger KL

    def test_full_padding_to_engine_vocab(self):
        out = self.d / "ref.full"
        rc, _ = self.run_main("--out", str(out), "--format", "full", "--engine-vocab", str(self.V + 24))
        self.assertEqual(rc, 0)
        rows = np.fromfile(out, "<f4").reshape(-1, self.V + 24)
        self.assertEqual(rows.shape[0], self.N - 1 - self.C)
        self.assertTrue(np.all(np.isneginf(rows[:, self.V:])))
        self.assertTrue(np.allclose(np.exp(rows[:, :self.V].astype(np.float64)).sum(1), 1.0, atol=1e-5))
        kl, same, _ = engine_score_full(out, self.logits, self.tokens, self.C, self.V + 24)
        self.assertLess(abs(kl), 1e-5)
        self.assertEqual(same, 100.0)

    def test_errors(self):
        self.assertEqual(self.run_main("--out", str(self.d / "x"), "--n-vocab", "999")[0], 2)   # size mismatch
        (self.d / "tokens.txt").write_text("1 2 3")
        self.assertEqual(self.run_main("--out", str(self.d / "x"))[0], 2)                       # token count
        (self.d / "tokens.txt").write_text(" ".join(map(str, self.tokens)))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(conv.main(["--dump", str(self.dump), "--n-vocab", str(self.V), "--context",
                                        str(self.N), "--out", str(self.d / "x")]), 2)            # nothing to score

    def test_tokens_json_list(self):
        p = self.d / "t.json"
        p.write_text("[1, 2, 3]")
        self.assertEqual(conv.read_tokens(str(p)), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
