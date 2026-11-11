"""tools/test_kimi_k31_ref_layers.py - ref/kimi_k31.py on real-weight terms: the truncated model
(--layers N), lazy reads, --act-q8, --tokens-file, --seams and --dry-run, on the synthetic fixture
(tools/kimi_k31_synth_gguf.py, seed 3101, its 32 suggested test tokens; 9 layers, MLA at 3, 7, 8,
block pushes at 0, 3, 6).

  * the oracle's bits are the pinned ones (tools/kimi_k31_ref_pin.py, data/kimi-k31-ref-pin.json:
    every seam and the logits of a float32 / float64 prefill and decode, made by the oracle before
    any of these options), where this machine rounds as the pin's did; --layers 9 is the full run;
  * --layers N (N = 1, 4, 7) is, bit for bit, the full run of a GGUF holding only the first N layers
    (same output mix, norm and head), and every seam of layers < N equals the 9-layer run's;
    without the head it stops at l_out-(N-1) (trunk_out) and res_blocks-(N-1) = [embedding, the
    pushed blocks], a seam only truncated runs write;
  * lazy: without the weight cache the experts read are exactly the routed ones, each once per
    expert-loop chunk; no routed tensor, embedding or head is read whole; no layer >= N is read;
    neither the chunking nor the head slabs change a result;
  * the indexer's query blocks and situ()'s row blocks give the one-block values bit for bit;
    memory_estimate() bounds the forward's measured allocations;
  * --act-q8 32 rounds exactly as a scalar transcription of q8_quant_block (float32 scale and
    division, half to even); it rounds the inputs of products with iq::mv-type weights only (none
    on the F32 fixture); with F32 counted as one, a GEMV, the expert sum and the head equal
    hand-written products of rounded inputs; any block but 32 is refused;
  * --tokens-file (text, JSON, npy) gives the --tokens run's dumps; --seams keeps only matches; the
    CLI streams seams to disk (the same files) and checks the last layer whichever seams it keeps;
  * --dry-run on the fixture probes slices only and passes; it fails over its RSS limit or when the
    forward's memory estimate is; its read map counts expert reads per token chunk.

    python -m pytest tools/test_kimi_k31_ref_layers.py      (numpy only, ~30 s)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ref"))

import numpy as np  # noqa: E402

import kimi_k31_ref_pin as pin  # noqa: E402
import kimi_k31_synth_gguf as synth  # noqa: E402
import kimi_k31 as K  # noqa: E402  (ref/kimi_k31.py)

F32, F64 = np.float32, np.float64
CUTS = (1, 4, 7)          # dense layer only; through the first MLA layer and the push at 3; past the push at 6


def _first_layers(g: dict, n: int) -> dict:
    """The fixture geometry cut to its first n layers (the 1-indexed MLA list cut with it)."""
    return dict(g, n_layers=n, full_attn_layers=[i for i in g["full_attn_layers"] if i <= n])


def _equal_dumps(tc: unittest.TestCase, a: dict, b: dict, keys=None) -> None:
    keys = sorted(a) if keys is None else keys
    for k in keys:
        tc.assertIn(k, b, k)
        tc.assertEqual(a[k].dtype, b[k].dtype, k)
        if not np.array_equal(a[k], b[k]):
            tc.fail(f"{k}: differs" if a[k].shape != b[k].shape else
                    f"{k}: max |diff| {np.nanmax(np.abs(a[k].astype(F64) - b[k])):.3e}")


def _layer_of(key: str) -> int:
    head, _, tail = key.rpartition("-")
    return int(tail) if head and tail.isdigit() else -1


def _q8_scalar(x: np.ndarray, block: int) -> np.ndarray:
    """q8_quant_block one value at a time: float32 max, scale and division, round half to even."""
    v = np.asarray(x, dtype=F32).ravel()
    out = np.empty(v.size, dtype=F64)
    for b0 in range(0, v.size, block):
        am = F32(0.0)
        for j in range(block):
            am = max(am, F32(abs(v[b0 + j])))
        sc = F32(am / F32(127.0))
        for j in range(block):
            if am > 0:
                r = F32(v[b0 + j] / sc)
                q = np.floor(F64(r))
                d = F64(r) - q
                q = q + 1 if d > 0.5 or (d == 0.5 and q % 2 == 1) else q    # half to even
            else:
                q = 0.0
            out[b0 + j] = q * F64(sc)
    return out.reshape(np.shape(x))


class KimiK31RefLayersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-layers-")
        cls.addClassCleanup(cls.tmp.cleanup)
        d = pathlib.Path(cls.tmp.name)
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        cls.gguf = d / "tiny.gguf"
        synth.write_gguf(cls.gguf, *synth.to_gguf(vals, synth.TINY))
        cls.cut_gguf = {}
        for n in CUTS:
            cls.cut_gguf[n] = d / f"tiny-first{n}.gguf"
            synth.write_gguf(cls.cut_gguf[n], *synth.to_gguf(vals, _first_layers(synth.TINY, n)))
        cls.tokens = synth.sample_tokens(synth.TINY, synth.SEED)
        cls.T = len(cls.tokens)
        cls.full = {}
        for dt in ("float32", "float64"):
            m = cls.model(dtype=dt)
            cls.full[dt] = (m.forward(cls.tokens), dict(m.dumps))
            m.close()
        cls.cfg = K.KimiK31Config(K.Gguf(str(cls.gguf)).g.metadata)

    @classmethod
    def model(cls, path=None, **kw):
        kw.setdefault("cache_weights", True)
        return K.KimiK31(path or cls.gguf, **kw)

    # ---- the pinned bits; the truncated model
    def test_oracle_bits_are_the_pinned_ones(self):
        want = json.loads(pin.PIN.read_text())
        if want["probe"] != pin.platform_probe():
            self.skipTest("this machine rounds differently from the one the pin was made on "
                          "(tools/test_kimi_k31_oracle.py's gate holds the oracle here)")
        diff = pin.compare(pin.pin_runs(K, self.gguf, self.tokens), want["runs"])
        self.assertEqual(diff, [], f"{len(diff)} seams differ from the pin")

    def test_layers_all_is_the_full_run(self):
        for dt in ("float32", "float64"):
            m = self.model(dtype=dt, layers=self.cfg.n_layers)
            lg = m.forward(self.tokens)
            full_lg, full_d = self.full[dt]
            self.assertTrue(np.array_equal(lg, full_lg), dt)
            self.assertEqual(sorted(m.dumps), sorted(full_d))
            self.assertFalse(any(k.startswith("res_blocks") for k in m.dumps))   # not a truncated run
            _equal_dumps(self, full_d, m.dumps)

    def test_layers_n_is_a_first_n_layers_file(self):
        """--layers N with the head IS the N-layer GGUF's model; its seams are the full run's."""
        for dt in ("float32", "float64"):
            full_lg, full_d = self.full[dt]
            for n in CUTS:
                with self.subTest(dt=dt, n=n):
                    m = self.model(dtype=dt, layers=n)
                    s = self.model(self.cut_gguf[n], dtype=dt)
                    self.assertEqual(s.cfg.n_layers, n)
                    lg, lg_s = m.forward(self.tokens), s.forward(self.tokens)
                    self.assertTrue(np.array_equal(lg, lg_s))
                    cut = f"res_blocks-{n - 1}"                 # the truncated run's own seam
                    self.assertEqual(cut in m.dumps, n < self.cfg.n_layers)
                    self.assertNotIn(cut, s.dumps)
                    self.assertEqual(sorted(k for k in m.dumps if k != cut), sorted(s.dumps))
                    _equal_dumps(self, s.dumps, m.dumps)
                    trunk = [k for k in m.dumps if _layer_of(k) < n and not k.startswith("res_blocks")
                             and k not in ("res_probs", "res_mix", "result_norm", "result_output")]
                    self.assertGreaterEqual(len(trunk), 15 * n)
                    _equal_dumps(self, m.dumps, full_d, trunk)
                    self.assertFalse(any(_layer_of(k) >= n for k in m.dumps))
                    if n < self.cfg.n_layers:
                        self.assertFalse(np.array_equal(lg, full_lg))

    def test_no_head_stops_at_the_cut(self):
        n = 4
        m = self.model(layers=n, head=False)
        self.assertIsNone(m.forward(self.tokens))
        d, full = m.dumps, self.full["float32"][1]
        for k in ("res_probs", "res_mix", "result_norm", "result_output"):
            self.assertNotIn(k, d)
        self.assertNotIn("output.weight", {name for name, _ in m.reads})
        _equal_dumps(self, d, full, [f"l_out-{n - 1}"])
        self.assertTrue(np.array_equal(m.trunk_out, full[f"l_out-{n - 1}"]))
        blocks = d[f"res_blocks-{n - 1}"]               # pushed at layers 0 and 3: [embedding, P after layer 2]
        self.assertEqual(blocks.shape, (self.cfg.n_embd, 2, self.T))
        self.assertTrue(np.array_equal(blocks[:, 0], full["inp_embd"]))
        self.assertTrue(np.array_equal(blocks[:, 1], full["l_out-2"]))
        self.assertFalse(any(k.startswith("res_blocks") for k in full))         # the full run has none
        m = self.model(head=False)                      # every layer, no head: truncated after all
        m.forward(self.tokens)
        self.assertEqual(m.dumps["res_blocks-8"].shape[1], 3)                   # pushes at 0, 3, 6

    def test_truncated_decode_and_state(self):
        """Token-by-token through a truncated model's state equals its prefill; a state of another
        depth is refused."""
        n = 4
        m = self.model(dtype="float64", layers=n)
        pre = m.forward(self.tokens)
        st = m.new_state()
        steps = [m.forward(self.tokens[:5], st)] + [m.forward([t], st) for t in self.tokens[5:]]
        self.assertLess(np.abs(np.concatenate(steps, axis=1) - pre).max(), 1e-10)
        with self.assertRaisesRegex(ValueError, "4-layer run"):
            self.model(dtype="float64").forward([1], st)

    def test_layers_refused(self):
        for bad in (0, -1, self.cfg.n_layers + 1):
            with self.assertRaisesRegex(ValueError, "the file has 9 layers"):
                self.model(layers=bad)

    # ---- lazy reads
    def test_lazy_reads_only_the_routed_experts(self):
        """Without the weight cache (a real file): 3 tokens (at most 30 of the 24 x top-10 experts,
        so some are never picked) in expert-loop chunks of 2 tokens."""
        n, toks, chunk = 7, self.tokens[:3], 2
        m = self.model(layers=n, cache_weights=False, moe_token_chunk=chunk)
        m.forward(toks)
        reads, c = m.reads, self.cfg
        for (name, what), cnt in reads.items():
            self.assertLess(int(name.split(".")[1]) if name.startswith("blk.") else -1, n, name)
            if name.endswith("_exps.weight"):
                self.assertIsInstance(what, int, f"{name} read {what}")
            if name in ("token_embd.weight", "output.weight"):
                self.assertEqual(what, "rows", name)
        self.assertEqual(reads[("token_embd.weight", "rows")], len(set(toks)))
        self.assertEqual(reads[("output.weight", "rows")], c.n_vocab)
        chunks = [list(range(a, min(len(toks), a + chunk))) for a in range(0, len(toks), chunk)]
        unpicked = 0
        for il in range(c.n_layer_dense_lead, n):
            ids = m.dumps[f"ffn_moe_topk-{il}"]
            per_chunk = [set(int(e) for e in ids[:, ch].ravel()) for ch in chunks]
            want = {e: sum(e in s for s in per_chunk) for e in set().union(*per_chunk)}
            unpicked += c.n_expert - len(want)
            for base in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                name = f"blk.{il}.{base}.weight"
                got = {what: cnt for (nm, what), cnt in reads.items() if nm == name}
                self.assertEqual(got, want, name)
        self.assertGreater(unpicked, 0, "every expert was routed: the test proves nothing")
        self.assertLess(m.bytes_read, sum(p.stat().st_size for p in m.g.paths))

    def test_chunking_and_head_slabs_change_nothing(self):
        """Expert-loop chunks of 1, 7 and all tokens are the same arithmetic; head slabs of 64 rows
        are the head to rounding (float64: 1e-12)."""
        for chunk in (1, 7):
            m = self.model(moe_token_chunk=chunk)
            self.assertTrue(np.array_equal(m.forward(self.tokens), self.full["float32"][0]), chunk)
            _equal_dumps(self, self.full["float32"][1], m.dumps)
        m = self.model(dtype="float64", head_chunk_rows=64)
        lg = m.forward(self.tokens)
        ref = self.full["float64"][0]
        self.assertLess(np.abs(lg - ref).max(), 1e-12 * np.abs(ref).max())
        self.assertEqual(m.reads[("output.weight", "rows")], self.cfg.n_vocab)
        self.assertNotIn(("output.weight", None), m.reads)

    # ---- memory
    def test_indexer_query_blocks_change_nothing(self):
        """Blocks of 2, 3, 5 and 31 (+1) queries, decode included: the one-block sums, bit for bit
        (a lone query would sum its heads in another order: the last block takes it)."""
        hi = self.cfg.idx_n_head
        for dt in ("float32", "float64"):
            full_lg, full_d = self.full[dt]
            for tb in (2, 3, 5, 31):
                with self.subTest(dt=dt, tb=tb):
                    m = self.model(dtype=dt, idx_block_bytes=tb * self.T * hi * 8)
                    st = m.new_state()
                    self.assertTrue(np.array_equal(m.forward(self.tokens, st), full_lg))
                    _equal_dumps(self, full_d, m.dumps)
                    ref = self.model(dtype=dt)
                    st_r = ref.new_state()
                    ref.forward(self.tokens, st_r)
                    for t in self.tokens[:2]:
                        self.assertTrue(np.array_equal(m.forward([t], st), ref.forward([t], st_r)))
                        _equal_dumps(self, ref.dumps, m.dumps)

    def test_situ_row_blocks_change_nothing(self):
        """situ() in blocks of rows (SITU_BLOCK, here 100 values) is the one-block SiTU, bit for bit."""
        old = K.SITU_BLOCK
        try:
            K.SITU_BLOCK = 100
            for dt in ("float32", "float64"):
                m = self.model(dtype=dt)
                self.assertTrue(np.array_equal(m.forward(self.tokens), self.full[dt][0]), dt)
                _equal_dumps(self, self.full[dt][1], m.dumps)
        finally:
            K.SITU_BLOCK = old

    def test_memory_estimate_bounds_the_forward(self):
        """tracemalloc's peak over a forward as the CLI runs it (uncached weights, seams to a sink)
        stays under memory_estimate()["peak"], at 32 and 160 tokens."""
        import tracemalloc
        rng = np.random.default_rng(11)
        for T in (32, 160):
            toks = [int(x) for x in rng.integers(0, self.cfg.n_vocab, T)]
            for dt in ("float32", "float64"):
                m = self.model(dtype=dt, cache_weights=False, idx_block_bytes=1 << 16)
                m.sink = lambda name, arr: None
                est = m.memory_estimate(T)
                tracemalloc.start()
                try:
                    base = tracemalloc.get_traced_memory()[0]
                    m.forward(toks)
                    peak = tracemalloc.get_traced_memory()[1] - base
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, est["peak"], (T, dt, peak, est))
                self.assertGreater(peak, est["peak"] / 8, (T, dt, peak, est))   # an estimate, not a wild one

    # ---- --act-q8
    def test_q8_round_is_the_engine_rounding(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((128, 3)).astype(F32) * F32(3.0)
        x[32:64, 1] = 0.0                                          # a zero block stays 0
        x[:32, 2] = np.array([127.0, 2.5, 3.5, -2.5, -0.5, 0.5, 1.5] + [0.25] * 25, dtype=F32)  # sc = 1: ties
        got = K.q8_round(x, 32, F64)
        want = np.stack([_q8_scalar(x[:, j], 32) for j in range(x.shape[1])], axis=1)
        self.assertTrue(np.array_equal(got, want))
        self.assertTrue(np.array_equal(got[:7, 2], [127.0, 2.0, 4.0, -2.0, -0.0, 0.0, 2.0]))   # half to even
        self.assertTrue(np.all(got[32:64, 1] == 0.0))
        # float64 input is rounded to float32 first; a float32 result is the float64 one rounded
        self.assertTrue(np.array_equal(K.q8_round(x.astype(F64) * (1 + 1e-12), 32, F64), got))
        self.assertTrue(np.array_equal(K.q8_round(x, 32, F32), got.astype(F32)))
        err = np.abs(got - x).reshape(4, 32, 3).max(axis=1) / np.abs(x).reshape(4, 32, 3).max(axis=1).clip(1e-30)
        self.assertLessEqual(err.max(), 0.5 / 127 * (1 + 1e-6))
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            K.q8_round(x[:100], 32)

    def test_act_q8_follows_the_weight_type(self):
        """iq::mv types only: on the all-F32 fixture --act-q8 32 rounds nothing; the embedding and
        the per-head k_b / v_b never round; any block but 32 is refused."""
        m = self.model(act_q8=32)
        self.assertFalse(any(m.q8_rounds(n) for n in m.expected_tensors()))
        self.assertTrue(np.array_equal(m.forward(self.tokens), self.full["float32"][0]))
        _equal_dumps(self, self.full["float32"][1], m.dumps)
        self.assertTrue({"Q8_0", "Q4_K", "Q6_K", "IQ1_S", "IQ2_XXS"} <= K.IQ_MV_TYPES)
        self.assertFalse({"F32", "F16", "BF16", "MXFP4"} & K.IQ_MV_TYPES)
        f = self.model(act_q8=32, act_q8_types={"F32"})
        mla = self.cfg.mla_layers()[0]
        for name in ("token_embd.weight", f"blk.{mla}.attn_k_b.weight", f"blk.{mla}.attn_v_b.weight",
                     "blk.0.attn_norm.weight", "blk.0.ssm_conv1d_q.weight"):
            self.assertFalse(f.q8_rounds(name), name)
        for name in ("output.weight", "blk.0.ffn_down.weight", "blk.1.ffn_gate_exps.weight",
                     f"blk.{mla}.indexer.attn_q_b.weight", "blk.1.ffn_gate_inp.weight"):
            self.assertTrue(f.q8_rounds(name), name)
        for bad in (-1, 1, 16, 48, 256):
            with self.assertRaisesRegex(ValueError, f"act_q8 {bad}: 0 \\(off\\) or 32"):
                self.model(act_q8=bad)

    def test_act_q8_rounds_the_product_inputs(self):
        """With F32 counted as an iq::mv type (float64): a GEMV (the latent down), the first MoE
        layer's expert sum and the head equal hand-written products of rounded inputs; the
        embedding is unchanged and the change is int8-sized."""
        c = self.cfg
        il = c.n_layer_dense_lead
        m = self.model(dtype="float64", act_q8=32, act_q8_types={"F32"}, layers=il + 1)
        lg = m.forward(self.tokens)
        d, full = m.dumps, self.full["float64"][1]
        _equal_dumps(self, full, d, ["inp_embd"])
        q8 = lambda a: np.stack([_q8_scalar(a[:, t], 32) for t in range(a.shape[1])], axis=1)  # noqa: E731
        for seam, x, w in ((f"ffn_routed_in-{il}", d[f"ffn_norm-{il}"], m.layer_t(il, "ffn_routed_down")),
                           ("result_output", d["result_norm"], m.t("output.weight"))):
            want = w @ q8(x)
            self.assertLess(np.abs(d[seam] - want).max(), 1e-12 * np.abs(want).max(), seam)
        self.assertTrue(np.array_equal(lg, d["result_output"]))
        z, ids, w = d[f"ffn_routed_in-{il}"], d[f"ffn_moe_topk-{il}"], d[f"ffn_moe_weights_scaled-{il}"]
        want = np.zeros((c.n_expert_latent, self.T))
        for t in range(self.T):
            zq = _q8_scalar(z[:, t], 32)
            for i, e in enumerate(ids[:, t]):
                h = K.situ(m.expert_w(il, "ffn_gate_exps", e) @ zq, m.expert_w(il, "ffn_up_exps", e) @ zq,
                           c.situ_beta, c.situ_linear_beta, F64)
                want[:, t] += w[i, t] * (m.expert_w(il, "ffn_down_exps", e) @ _q8_scalar(h, 32))
        got, plain = d[f"ffn_moe_out-{il}"], full[f"ffn_moe_out-{il}"]
        self.assertLess(np.abs(got - want).max(), 1e-12 * np.abs(want).max())
        rel = np.abs(d["l_out-0"] - full["l_out-0"]).max() / np.abs(full["l_out-0"]).max()
        self.assertTrue(1e-5 < rel < 5e-2, rel)            # the dense layer; past it the routing moves

    # ---- the CLI: --tokens-file, --seams, --dry-run
    def run_cli(self, *args, ok=True):
        p = subprocess.run([sys.executable, str(REPO / "ref" / "kimi_k31.py"), str(self.gguf), *args],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED="0"))
        if ok and p.returncode != 0:
            self.fail(f"kimi_k31.py {' '.join(args)}: rc {p.returncode}\n{p.stdout}\n{p.stderr}")
        return p

    def test_tokens_file_formats(self):
        d = pathlib.Path(self.tmp.name)
        toks = self.tokens[:12]
        files = {"a.txt": "# a prompt\n" + ", ".join(map(str, toks[:5])) + "\n" + " ".join(map(str, toks[5:])) + "\n",
                 "b.ids": "[" + ", ".join(map(str, toks)) + "]\n",
                 "c.json": json.dumps(toks), "d.json": json.dumps({"tokens": toks, "why": "x"})}
        for name, text in files.items():
            (d / name).write_text(text)
        np.save(d / "e.npy", np.asarray(toks, dtype=np.int32))
        for name in [*files, "e.npy"]:
            self.assertEqual(K.load_tokens(d / name), toks, name)
        for name, text in {"bad.txt": "1, 2.5", "bad.json": json.dumps([1, "2"]), "empty.txt": "# none\n"}.items():
            (d / name).write_text(text)
            with self.assertRaises(ValueError, msg=name):
                K.load_tokens(d / name)
        np.save(d / "bad.npy", np.asarray([1.0, 2.0]))
        with self.assertRaises(ValueError):
            K.load_tokens(d / "bad.npy")
        # the CLI: the file and --tokens give the same dumps; --seams keeps only the matches
        a, b = d / "dump-file", d / "dump-tokens"
        self.run_cli("--tokens-file", str(d / "a.txt"), "--layers", "4", "--dump-dir", str(a))
        self.run_cli("--tokens", ",".join(map(str, toks)), "--layers", "4", "--dump-dir", str(b))
        fa = sorted(p.name for p in a.iterdir())
        self.assertEqual(fa, sorted(p.name for p in b.iterdir()))
        self.assertIn("res_blocks-3.npy", fa)
        for f in fa:
            self.assertTrue(np.array_equal(np.load(a / f), np.load(b / f)), f)
        s = d / "dump-seams"
        p = self.run_cli("--tokens-file", str(d / "c.json"), "--layers", "4", "--no-head",
                         "--seams", "l_out-3,res_blocks-*", "--dump-dir", str(s))
        self.assertEqual(sorted(x.name for x in s.iterdir()), ["l_out-3.npy", "res_blocks-3.npy"])
        self.assertTrue(np.array_equal(np.load(s / "l_out-3.npy"), np.load(a / "l_out-3.npy")))
        self.assertIn("l_out-3 (256, 12)", p.stdout)
        # the last layer is checked and summarized whichever seams are kept
        p = self.run_cli("--tokens-file", str(d / "c.json"), "--layers", "4", "--no-head", "--seams", "res_blocks-3")
        self.assertIn("l_out-3 (256, 12)", p.stdout)
        # raw dumps streamed by the CLI = write_raw_dumps of the library's dumps
        r, lib = d / "raw-cli", d / "raw-lib"
        self.run_cli("--tokens-file", str(d / "c.json"), "--layers", "4", "--dump-raw-dir", str(r))
        m = self.model(layers=4)
        m.forward(toks)
        K.write_raw_dumps(m.dumps, lib)
        self.assertEqual((r / "manifest.txt").read_text(), (lib / "manifest.txt").read_text())
        for f in lib.iterdir():
            self.assertEqual((r / f.name).read_bytes(), f.read_bytes(), f.name)
        self.assertNotEqual(self.run_cli("--act-q8", "48", ok=False).returncode, 0)
        self.assertNotEqual(self.run_cli("--tokens", "1", "--tokens-file", str(d / "c.json"), ok=False).returncode, 0)
        self.assertNotEqual(self.run_cli("--tokens", str(self.cfg.n_vocab), ok=False).returncode, 0)

    def test_check_output(self):
        m = self.model(layers=2, head=False)
        self.assertEqual(K.check_output(m, m.forward(self.tokens[:4])), 0)
        m.trunk_out = m.trunk_out.copy()
        m.trunk_out[0, 0] = np.nan
        self.assertEqual(K.check_output(m, None), 1)
        lg = self.full["float32"][0].copy()
        self.assertEqual(K.check_output(m, lg), 0)
        lg[1, 2] = np.inf
        self.assertEqual(K.check_output(m, lg), 1)

    def test_seams_filter(self):
        m = self.model(seams=["l_out-*", "top_k-3"])
        m.forward(self.tokens)
        self.assertEqual(sorted(m.dumps), sorted([f"l_out-{i}" for i in range(self.cfg.n_layers)] + ["top_k-3"]))
        _equal_dumps(self, m.dumps, self.full["float32"][1])

    def test_dry_run(self):
        m = self.model(layers=4, cache_weights=False)
        rmap = m.read_map(self.T)
        kinds = {e["name"]: e["kind"] for e in rmap}
        self.assertEqual(kinds["token_embd.weight"], "rows")
        self.assertEqual(kinds["output.weight"], "head")
        self.assertEqual(kinds["blk.1.ffn_gate_exps.weight"], "experts")
        self.assertFalse(any(e["layer"] >= 4 for e in rmap))
        self.assertEqual(sorted(kinds), sorted(m.expected_tensors()))
        for e in rmap:
            self.assertTrue(np.isfinite(m.probe(e, self.tokens)).all(), e["name"])
        big = [nm for (nm, what) in m.reads if what is None and m._nbytes(nm, m.g.info(nm).elements) > (1 << 20)]
        self.assertEqual(big, [])
        self.assertFalse(any(isinstance(w, int) and w != 0 for (_, w) in m.reads))   # expert 0 only
        # experts per chunk: 5 tokens in chunks of 2 read min(24, 10 x 2) + min(24, 10 x 2) + min(24, 10 x 1)
        mc = self.model(layers=4, cache_weights=False, moe_token_chunk=2)
        self.assertEqual(mc.expert_reads(5), 50)
        self.assertEqual(mc.expert_reads(1), 10)
        e = next(e for e in mc.read_map(5) if e["name"] == "blk.1.ffn_down_exps.weight")
        self.assertEqual(e["reads"], e["bytes"] // self.cfg.n_expert * 50)
        self.assertEqual(m.expert_reads(self.T), self.cfg.n_expert)
        p = self.run_cli("--dry-run", "--layers", "4", "--tokens", "1,2,3")
        self.assertIn("dry run: OK", p.stdout)
        self.assertIn("read map: layers 0..3 of 9 + head, 3 tokens", p.stdout)
        self.assertIn("forward memory estimate (3 tokens, float32, seams streamed): peak", p.stdout)
        p = self.run_cli("--dry-run", "--layers", "4", "--max-rss-gb", "0.001", ok=False)
        self.assertEqual(p.returncode, 1)
        self.assertIn("dry run: FAILED", p.stdout)
        # a prompt whose forward would not fit fails although the probes do (the (n_kv, T) buffers)
        n = 6000
        tf = pathlib.Path(self.tmp.name) / "long.npy"
        np.save(tf, (np.arange(n) % self.cfg.n_vocab).astype(np.int32))
        est = self.model(layers=4, cache_weights=False).memory_estimate(n)["peak"]
        self.assertGreater(est, 0.6 * 2**30)
        p = self.run_cli("--dry-run", "--layers", "4", "--tokens-file", str(tf), "--max-rss-gb", "0.5", ok=False)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("dry run: FAILED", p.stdout)
        self.assertIn("non-finite probes: 0", p.stdout)


if __name__ == "__main__":
    unittest.main()
