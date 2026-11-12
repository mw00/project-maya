"""tools/test_glm_dsa_slice_oracle.py - the B7.5 slice oracle (ref/glm_dsa.py --layers A:B --hidden-in) and the
NextN probe (tools/glm_dsa_mtp_probe.py) on the SMALL synthetic preset, CPU only (about 20 s).

  * a slice A:n_layer fed the full forward's l_out-(A-1) reproduces the full forward's logits at --positions (raw
    float32 rows, (n_embd, T) and (T, n_embd) .npy inputs), and two chained slices equal one;
  * the slice's --dump-dir seams (MoE ids, selections) equal the full run's on the same layers;
  * the comparison report: KL(slice || engine) ~ 0 against the full logits, the B7.5 gate passes when engine and
    llama agree and fails (exit 1) when the "engine" is a misreading (neox_rope);
  * a slice starting on a shared indexer layer is refused;
  * the NextN probe from the oracle's own trunk, from the post-norm hidden (result_norm) and from the trunk's last
    l_out equals ref/glm_dsa.py --mtp-out; a wrong draft fails its gate.

    python3 tools/test_glm_dsa_slice_oracle.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ref"))

try:
    import glm_dsa_synth_gguf as synth  # noqa: E402
    from glm_dsa import GlmDsa, Options, kl_top_rest, load_hidden, parse_positions, slice_gate  # noqa: E402
    _IMPORT_ERR = None
except ImportError as e:  # gguf-py missing
    _IMPORT_ERR = e

PY = sys.executable
T = 24
_TMP: tempfile.TemporaryDirectory | None = None
D: pathlib.Path | None = None
GGUF: pathlib.Path | None = None


def toks(n: int, v: int) -> list[int]:
    return [(i * 37 + 11) % v for i in range(n)]


def run(*args: str, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    r = subprocess.run([PY, *args], capture_output=True, text=True, env=env, cwd=str(ROOT))
    if r.returncode not in ok:
        raise AssertionError(f"{args[0]} exit {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


def setUpModule():
    global _TMP, D, GGUF
    if _IMPORT_ERR is not None:
        raise unittest.SkipTest(f"gguf-py not importable: {_IMPORT_ERR}")
    _TMP = tempfile.TemporaryDirectory(prefix="strata-slice-")
    D = pathlib.Path(_TMP.name)
    GGUF = D / "small.gguf"
    synth.write(GGUF, "SMALL", synth.DEFAULT_SEED, None)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


class Positions(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_positions(None, 5).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(parse_positions("1:3", 5).tolist(), [1, 2])
        self.assertEqual(parse_positions("4,0,4", 5).tolist(), [0, 4])
        self.assertEqual(parse_positions("-2", 5).tolist(), [3, 4])
        self.assertEqual(parse_positions("101:103", 5, pos0=100).tolist(), [1, 2])
        with self.assertRaises(ValueError):
            parse_positions("5", 5)

    def test_gate(self):
        self.assertTrue(slice_gate(0.010, 0.008)[0])
        self.assertFalse(slice_gate(0.030, 0.010)[0])     # |diff| 0.02 > 0.01
        self.assertFalse(slice_gate(0.006, 0.003)[0])     # 2x the llama KL
        self.assertTrue(slice_gate(0.006, 0.003, floor=0.01)[0])


class Slice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = GlmDsa(GGUF, Options(mode="hf"))
        m.collect = True
        cls.cfg = m.cfg
        cls.tokens = toks(T, m.cfg.n_vocab)
        cls.full = m.forward(cls.tokens)                       # (vocab, T)
        cls.dumps = dict(m.dumps)
        m.close()
        cls.full_f32 = D / "full.f32"
        np.ascontiguousarray(cls.full.T.astype(np.float32)).tofile(cls.full_f32)
        mut = GlmDsa(GGUF, Options(mode="hf", mutations=frozenset({"neox_rope"})))
        cls.mut_f32 = D / "mut.f32"
        np.ascontiguousarray(mut.forward(cls.tokens).T.astype(np.float32)).tofile(cls.mut_f32)
        mut.close()
        # the residual entering layer 2 (a full layer) as raw float32 rows, ggml order
        cls.h1 = cls.dumps["l_out-1"]
        cls.h1_f32 = D / "l_out-1.f32"
        np.ascontiguousarray(cls.h1.T.astype(np.float32)).tofile(cls.h1_f32)
        np.save(D / "h1_et.npy", cls.h1.astype(np.float32))
        np.save(D / "h1_te.npy", cls.h1.T.astype(np.float32))

    def test_layer_layout(self):
        c = self.cfg
        self.assertTrue(c.is_full(2) and not c.is_full(3) and c.is_full(6), c.idx_full)

    def test_hidden_formats(self):
        a = load_hidden(self.h1_f32, self.cfg.n_embd)
        b = load_hidden(D / "h1_et.npy", self.cfg.n_embd)
        c = load_hidden(D / "h1_te.npy", self.cfg.n_embd)
        self.assertEqual(a.shape, (self.cfg.n_embd, T))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)

    def test_slice_reproduces_full(self):
        out, rep, dd = D / "slice.f32", D / "slice.json", D / "slice_dumps"
        r = run("ref/glm_dsa.py", str(GGUF), "--layers", f"2:{self.cfg.n_layer}", "--hidden-in", str(self.h1_f32),
                "--positions", "4:20", "--logits-out", str(out), "--compare-engine", str(self.full_f32),
                "--compare-llama", str(self.full_f32), "--report", str(rep), "--dump-dir", str(dd))
        got = np.fromfile(out, dtype="<f4").reshape(-1, self.cfg.n_vocab)
        want = self.full[:, 4:20].T
        self.assertEqual(got.shape, want.shape)
        # the slice starts from float32-rounded hidden states: logits agree to float32 noise
        self.assertLess(float(np.abs(got - want).max()), 1e-3 * (1.0 + float(np.abs(want).max())))
        self.assertEqual(got.argmax(axis=1).tolist(), want.argmax(axis=1).tolist())
        j = json.loads(rep.read_text())
        self.assertEqual(j["positions"], list(range(4, 20)))
        self.assertLess(j["engine"]["kl_mean"], 1e-6)
        self.assertTrue(j["gate"]["pass"], r.stdout)
        self.assertIn("B7.5 slice gate: PASS", r.stdout)
        # the slice's MoE ids and selections equal the full run's on the same layers
        for il in range(2, self.cfg.n_layer):
            if self.cfg.is_moe(il):
                np.testing.assert_array_equal(np.load(dd / f"ffn_moe_topk-{il}.npy"), self.dumps[f"ffn_moe_topk-{il}"])
            np.testing.assert_array_equal(np.load(dd / f"top_k-{il}.npy"), self.dumps[f"top_k-{il}"])

    def test_gate_fails_on_a_misreading(self):
        rep = D / "mut.json"
        r = run("ref/glm_dsa.py", str(GGUF), "--layers", f"2:{self.cfg.n_layer}", "--hidden-in", str(self.h1_f32),
                "--positions", "-8", "--compare-engine", str(self.mut_f32), "--compare-llama", str(self.full_f32),
                "--report", str(rep), ok=(1,))
        j = json.loads(rep.read_text())
        self.assertFalse(j["gate"]["pass"])
        self.assertGreater(j["engine"]["kl_mean"], 0.01)
        self.assertIn("FAIL", r.stdout)

    def test_chained_slices(self):
        mid = D / "mid.npy"
        run("ref/glm_dsa.py", str(GGUF), "--layers", "2:6", "--hidden-in", str(D / "h1_et.npy"),
            "--hidden-out", str(mid))
        h = np.load(mid)
        np.testing.assert_allclose(h, self.dumps["l_out-5"], rtol=1e-4, atol=1e-5)
        out = D / "chain.f32"
        run("ref/glm_dsa.py", str(GGUF), "--layers", f"6:{self.cfg.n_layer}", "--hidden-in", str(mid),
            "--logits-out", str(out))
        got = np.fromfile(out, dtype="<f4").reshape(-1, self.cfg.n_vocab)
        self.assertEqual(got.argmax(axis=1).tolist(), self.full.argmax(axis=0).tolist())
        k = kl_top_rest(self.full, got.T.astype(np.float64))
        self.assertLess(float(k.max()), 1e-6)

    def test_shared_layer_start_refused(self):
        r = run("ref/glm_dsa.py", str(GGUF), "--layers", "3:5", "--hidden-in", str(D / "h1_et.npy"), ok=(1,))
        self.assertIn("shared indexer layer", r.stderr)

    def test_head_in_chunks(self):
        m = GlmDsa(GGUF, Options(mode="hf"), cache=False)
        x = self.dumps[f"l_out-{self.cfg.n_layer - 1}"]
        lg = m.head(x, [0, 5, 23], chunk_rows=100)                 # 512 rows in 6 chunks, no weight cache
        m.close()
        np.testing.assert_allclose(lg, self.full[:, [0, 5, 23]], rtol=1e-9, atol=1e-9)


class MtpProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = GlmDsa(GGUF, Options(mode="hf"))
        m.collect = True
        cls.n_vocab = m.cfg.n_vocab
        cls.tokens = toks(T, m.cfg.n_vocab)
        cls.tok_csv = ",".join(map(str, cls.tokens))
        cls.trunk_logits = m.forward(cls.tokens)
        cls.dumps = dict(m.dumps)
        cls.n_layer = m.cfg.n_layer
        m.close()
        cls.ref_mtp = D / "mtp.f32"
        run("ref/glm_dsa.py", str(GGUF), "--tokens", cls.tok_csv, "--mtp", "--mtp-out", str(cls.ref_mtp))
        np.ascontiguousarray(cls.dumps["result_norm"].T.astype(np.float32)).tofile(D / "h_nextn.f32")
        np.save(D / "l_out_last.npy", cls.dumps[f"l_out-{cls.n_layer - 1}"].astype(np.float32))
        wrong = np.fromfile(cls.ref_mtp, dtype="<f4").reshape(-1, cls.n_vocab)[:, ::-1].copy()
        cls.wrong = D / "wrong.f32"
        wrong.tofile(cls.wrong)

    def _probe(self, *src, ok=(0,)):
        rep = D / ("probe_%d.json" % abs(hash(src)))
        r = run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", self.tok_csv, *src, "--positions", "0:16",
                "--compare", f"oracle_cli={self.ref_mtp}", "--report", str(rep), ok=ok)
        return r, json.loads(rep.read_text())

    def test_oracle_trunk(self):
        out = D / "probe_draft.f32"
        r, j = self._probe("--oracle-trunk", "--draft-out", str(out))
        self.assertLess(j["compare"]["oracle_cli"]["kl_mean"], 1e-9, r.stdout)
        self.assertEqual(j["compare"]["oracle_cli"]["top1_same"], 16)
        np.testing.assert_allclose(np.fromfile(out, dtype="<f4"), np.fromfile(self.ref_mtp, dtype="<f4"), rtol=1e-6,
                                   atol=1e-6)

    def test_hidden_post_norm(self):
        r, j = self._probe("--hidden", str(D / "h_nextn.f32"))
        self.assertTrue(j["compare"]["oracle_cli"]["pass"], r.stdout)
        self.assertLess(j["compare"]["oracle_cli"]["kl_mean"], 1e-6)

    def test_trunk_out(self):
        r, j = self._probe("--trunk-out", str(D / "l_out_last.npy"))
        self.assertTrue(j["compare"]["oracle_cli"]["pass"], r.stdout)

    def test_wrong_draft_fails(self):
        rep = D / "probe_wrong.json"
        run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", self.tok_csv, "--oracle-trunk",
            "--compare", f"bad={self.wrong}", "--compare-ids", "ids=" + ",".join(["0"] * 23), "--report", str(rep),
            ok=(1,))
        j = json.loads(rep.read_text())
        self.assertFalse(j["compare"]["bad"]["pass"])
        self.assertEqual(j["compare"]["ids"]["n"], 16)

    def test_head_vs_trunk_agreement(self):
        # B11.3 / B15.4: teacher-forced agreement of the draft at row t with the trunk's greedy token at t + 1
        draft = np.fromfile(self.ref_mtp, dtype="<f4").reshape(-1, self.n_vocab)[:16].argmax(axis=1)
        trunk = self.trunk_logits.argmax(axis=0)
        want = int((draft == trunk[1:17]).sum())
        rep = D / "agree.json"
        run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", self.tok_csv, "--oracle-trunk", "--agree",
            "--positions", "0:16", "--compare", f"cli={self.ref_mtp}", "--report", str(rep))
        j = json.loads(rep.read_text())
        self.assertEqual(j["agree"]["oracle"]["n"], 16)
        self.assertEqual(j["agree"]["oracle"]["same"], want)
        self.assertEqual(j["agree"]["cli"]["same"], want)
        # the trunk's greedy ids from a logits file (rows from position 0) or an id list give the same
        lf = D / "trunk_logits.f32"
        np.ascontiguousarray(self.trunk_logits.T.astype(np.float32)).tofile(lf)
        for src in (["--trunk-logits", str(lf)], ["--trunk-ids", ",".join(map(str, trunk.tolist()))],
                    ["--trunk-ids", ",".join(map(str, trunk[1:].tolist())) + "@1"]):
            rep2 = D / "agree2.json"
            run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", self.tok_csv, "--hidden",
                str(D / "h_nextn.f32"), "--positions", "0:16", "--agree", *src, "--report", str(rep2))
            self.assertEqual(json.loads(rep2.read_text())["agree"]["oracle"]["same"], want, src)
        # --min-agree gates each source: a draft that always proposes the trunk's own next greedy token (ids
        # trunk[t + 1]) agrees on every row; the random synthetic head agrees on `want` of 16
        rep3 = D / "agree3.json"
        perfect = ",".join(map(str, trunk[1:17].tolist()))
        run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", self.tok_csv, "--oracle-trunk",
            "--compare-ids", f"perfect={perfect}", "--min-agree", "0.9", "--positions", "0:16", "--report", str(rep3),
            ok=(0,) if want >= 15 else (1,))
        j3 = json.loads(rep3.read_text())
        self.assertEqual(j3["agree"]["perfect"]["same"], 16)
        self.assertTrue(j3["agree"]["perfect"]["pass"])
        self.assertEqual(j3["agree"]["oracle"]["pass"], want >= 15)
        # --agree without a trunk source is a usage error
        r = run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", self.tok_csv, "--hidden", str(D / "h_nextn.f32"),
                "--agree", ok=(2,))
        self.assertIn("--agree needs", r.stderr)

    def test_usage_error(self):
        r = run("tools/glm_dsa_mtp_probe.py", str(GGUF), "--tokens", "5", "--oracle-trunk", ok=(2,))
        self.assertIn("at least 2 tokens", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
