"""tools/test_kimi_k31_check_dumps.py - the self-test of `kimi_k31_gate.py check-dumps`: an engine's seam dumps
judged on the M1a seam ladder.

CPU, numpy only. The "engine" is ref/kimi_k31.py in float32, written out in the layout an engine writes
(tools/kimi_k31_gate.py, "An engine's seam dumps"):
  * the reference's own float32 dumps pass every rung, in both layouts (raw and npz), against the golden and
    against the float64 reference alone, under both eps presets; the q_a/kv_a 1e-5 run uses the golden's E run;
  * the dump layout agrees with the gate's own (run_reference), and REF64_TOL is re-measured: each bound is
    2-5x (max) / 1.5-3x (rms) the float32 reference's gap to the float64 one (wide enough, and not widened);
  * a 1e-3 change of one value of a judged seam (or a 1e-3 relative change of the whole seam) fails exactly the
    rungs that judge it; so does one changed expert id or indexer key, a NaN, and a NaN or +inf where the
    indexer scores must be -inf;
  * dumps that cannot be judged are refused before anything is compared: a seam on a layer that has no such
    seam, a name ref/kimi_k31.py does not dump (the golden's own key names included), an unknown or missing
    run, a missing seam, a shape other than the reference's (token rows, rank, any axis but the count of
    experts / keys / residual blocks), a non-canonical layer number, a manifest that does not match its files,
    an integer seam stored as float, a file that cannot be read, a GGUF that is not the golden's fixture
    (refused by size, unhashed);
  * the plan's eleven M1a mutation switches (tools/test_kimi_k31_oracle.py MUTATIONS: ten applied to the
    reference, the A_log slice to the fixture's converter) and indexer top-k + 1 each fail first on the rung
    expected for them, every earlier rung passes, and the first failing rung's seams move by at least 10x
    their bounds (the table of what each moved is printed at the end);
  * the CLI: dump-ref + check-dumps (and its alias check-engine) exit 0, a perturbed file 1, a refused
    directory or an unreadable .npz 2; a partial --rungs run never prints the bare PASS line; a GGUF of another
    geometry than --geometry is refused (on big, unhashed); a --gguf that is no GGUF (missing, empty, junk, cut
    inside its header) is refused, exit 2 and no traceback, on both geometries and by both commands; --golden on
    big is refused; an error once the checks pass (in reference_dumps or check_dumps) is raised, not refused;
  * with KIMI_K31_BIG_FIXTURE (the directory of a kimi_k31_synth_gguf.py --geometry big build): check-dumps on
    the big fixture, which has no golden - the reference's float32 dumps pass every rung against the float64
    reference under both eps presets, each DUMP_TOL / DUMP_REF64_TOL bound is 1.9-2.5x the gap they measure
    (2x rounded up where measured, less a margin for the BLAS thread count), a 1e-3 change fails its rung, one
    eps preset's dumps judged under the other fail first on rung 4, and dump-ref + check-dumps --geometry big pass
    through the CLI (~6 minutes).

    python -m pytest tools/test_kimi_k31_check_dumps.py
"""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ref"))

import numpy as np  # noqa: E402

import kimi_k31  # noqa: E402
import kimi_k31_gate as G  # noqa: E402
import kimi_k31_synth_gguf as synth  # noqa: E402
import test_kimi_k31_oracle as O  # noqa: E402  (MUTATIONS, load_mutant)

GATE = HERE / "kimi_k31_gate.py"

# The plan's eleven M1a mutation switches (its names: kda_softplus_gate, alog_tail_slice, attnres_push_early /
# _late, situ_betas_swapped, routed_norm_before_sum, experts_trunc8, router_bias_dropped,
# future_keys_zero_fullrow, indexer_no_relu, idx_topk_minus1) under tools/test_kimi_k31_oracle.py's names, plus
# indexer_topk_plus_1, and the first rung each must fail on. BAR pushed early or late changes layer 0 itself:
# no block is pushed there, so its FFN reads the partial sum unmixed.
M1A_MUTATIONS = {
    "kda_softplus_gate": 1,
    "a_log_tail_slice": 1,                    # the converter's mutation: the unmodified reference reads its GGUF
    "attn_res_block_one_early": 2,
    "attn_res_block_one_late": 2,
    "situ_beta_swapped": 2,
    "latent_norm_before_sum": 3,
    "moe_topk_truncated_8": 3,
    "router_bias_dropped": 3,
    "indexer_future_zero_fill": 4,
    "indexer_no_relu": 4,
    "indexer_topk_plus_1": 4,
    "indexer_topk_minus_1": 4,
}


def copy_runs(runs: dict) -> dict:
    return {tag: {k: v.copy() for k, v in d.items()} for tag, d in runs.items()}


def failing(res) -> list[int]:
    return [rung.n for rung, rep in res if not rep.ok]


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-dumps-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = pathlib.Path(cls.tmp.name)
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        md, ts = synth.to_gguf(vals, synth.TINY)
        cls.gguf = cls.root / "tiny.gguf"
        synth.write_gguf(cls.gguf, md, ts)
        cls.golden = G.load_golden()
        cls.models = {}
        cls.m64 = cls.model("float64")
        cls.expected = cls.dumps(cls.m64)
        cls.runs32 = cls.dumps(cls.model("float32"))

    @classmethod
    def tearDownClass(cls):
        for m in cls.models.values():
            m.close()

    @classmethod
    def model(cls, dtype, eps="moonshot", **ov):
        key = (dtype, eps, tuple(sorted(ov.items())))
        if key not in cls.models:
            cls.models[key] = kimi_k31.KimiK31(cls.gguf, dtype=dtype, eps=eps, eps_overrides=ov or None)
        return cls.models[key]

    @classmethod
    def dumps(cls, m) -> dict:
        return {tag: G.reference_dumps(m, toks, df) for tag, toks, df in G.golden_runs(cls.golden[0])}

    def check(self, runs, m64=None, **kw):
        if m64 is None:
            m64, kw["expected"] = self.m64, self.expected
        return G.check_dumps(runs, m64, self.golden, **kw)

    def refused(self, runs, pattern, **kw):
        with self.assertRaisesRegex(G.DumpsRefused, pattern):
            self.check(runs, **kw)


class CheckDumpsPass(_Fixture):
    def test_reference_float32_dumps_pass_every_rung(self):
        res = self.check(self.runs32)
        self.assertEqual(failing(res), [], "\n".join(G.rung_summary(r, p) for r, p in res))
        self.assertEqual([r.n for r, _ in res], [1, 2, 3, 4, 5, 6])
        st = res[-1][1].stats
        self.assertEqual((st["moe_rows"], st.get("moe_rows_differ", 0)), (1872, 0))
        self.assertEqual((st["idx_rows"], st.get("idx_rows_differ", 0)), (702, 0))
        self.assertEqual(st["golden_run"], G.GOLD)
        self.assertEqual(st["seams"]["logits"]["vs"], {"golden " + G.GOLD})
        self.assertEqual(res[1][1].stats["seams"]["l_out-0"]["vs"], {G.REF64})

    def test_raw_and_npz_layouts_round_trip(self):
        for fmt in ("raw", "npz"):
            with self.subTest(fmt=fmt):
                d = self.root / f"layout-{fmt}"
                G.write_dumps(self.runs32, d, fmt)
                back = G.load_dumps(d)
                self.assertEqual(sorted(back), sorted(self.runs32))
                for tag, arrs in self.runs32.items():
                    self.assertEqual(sorted(back[tag]), sorted(arrs))
                    for name, a in arrs.items():
                        b = back[tag][name]
                        self.assertEqual(b.shape, a.shape, name)
                        self.assertTrue(np.issubdtype(b.dtype, np.integer) == G._is_int(name), name)
                        np.testing.assert_array_equal(b, a.astype(b.dtype), err_msg=f"{tag}/{name}")
                self.assertEqual(failing(self.check(back)), [])

    def test_raw_layout_is_ggml_order(self):
        """manifest dims are ggml's (features first); the data is features fastest, like an engine's buffer."""
        d = self.root / "ggml-order"
        G.write_dumps({"seq1.prefill": self.runs32["seq1.prefill"]}, d)
        lines = dict(line.split(" ", 1) for line in (d / "seq1.prefill" / "manifest.txt").read_text().splitlines())
        c = self.m64.cfg
        self.assertEqual(lines["kda_out-0"], f"{c.n_embd} 5")
        self.assertEqual(lines["kda_g1-0"], f"{c.kda_head_dim} {c.n_head} 5")
        self.assertEqual(lines["top_k-3"], f"{c.idx_top_k} 5")
        raw = np.fromfile(d / "seq1.prefill" / "kda_out-0.f32", dtype=np.float32)
        np.testing.assert_array_equal(raw[:c.n_embd], self.runs32["seq1.prefill"]["kda_out-0"][0])

    def test_layout_agrees_with_the_gates_own(self):
        for tag, toks, df in G.golden_runs(self.golden[0]):
            mine = G.to_layout(self.expected[tag], self.m64.cfg)
            theirs = G.run_reference(self.m64, toks, df)
            for key, v in theirs.items():
                with self.subTest(run=tag, key=key):
                    if key.endswith(".selected"):
                        self.assertEqual([sorted(int(i) for i in r if i >= 0) for r in mine[key]],
                                         [sorted(r) for r in v])
                    else:
                        np.testing.assert_array_equal(mine[key], v)

    def test_against_the_f64_reference_alone_and_both_eps(self):
        res = self.check(self.runs32, reference="ref64")
        self.assertEqual(failing(res), [])
        self.assertEqual(res[-1][1].stats["seams"]["logits"]["vs"], {G.REF64})
        llama64 = self.model("float64", "llama.cpp")
        llama32 = self.dumps(self.model("float32", "llama.cpp"))
        res = self.check(llama32, m64=llama64)
        self.assertIsNone(res[0][1].stats["golden_run"])
        self.assertEqual(failing(res), [])
        # the eps are resolved both ways: each preset's dumps fail the other's rung 1 (the KDA l2 eps)
        self.assertIn(1, failing(self.check(llama32)))
        self.assertIn(1, failing(self.check(self.runs32, m64=llama64)))
        # q_a/kv_a 1e-5 alone: the golden's E run judges the layer-3 seam
        e64 = self.model("float64", mla_latent=1e-5)
        res = self.check(self.dumps(self.model("float32", mla_latent=1e-5)), m64=e64)
        self.assertEqual(failing(res), [])
        self.assertEqual(res[3][1].stats["seams"]["layers.3.attn_out"]["vs"], {"golden " + G.ENGINE_EPS})
        self.assertIn(4, failing(self.check(self.runs32, m64=e64)))

    def test_ref64_bounds_are_measured(self):
        """Each REF64_TOL bound sits 2-5x (max) / 1.5-3x (pooled rms) above the float32 reference's gap."""
        res = self.check(self.runs32, reference="ref64")
        seen = set()
        for _, rep in res:
            for label, s in rep.stats["seams"].items():
                base = G._split(label)[0]
                if base in G.REF64_TOL:
                    seen.add(base)
                    atol, rms = G.REF64_TOL[base]
                    print(f"{label}: f32 vs f64 max {s['max']:.2e} (bound {atol:.1e}, {atol / s['max']:.1f}x) "
                          f"rms {s['rms']:.2e} (bound {rms:.1e}, {rms / s['rms']:.1f}x)", file=sys.stderr)
                    self.assertGreaterEqual(atol, 2.0 * s["max"], label)
                    self.assertGreaterEqual(rms, 1.5 * s["rms"], label)
                    # and not widened: a looser bound needs a new measurement (measured 3.0-3.1x / 2.0-2.1x)
                    self.assertLessEqual(atol, 5.0 * s["max"], label)
                    self.assertLessEqual(rms, 3.0 * s["rms"], label)
        self.assertEqual(seen, set(G.REF64_TOL))

    def test_rungs_can_be_climbed_one_at_a_time(self):
        """--rungs N needs and judges only rungs 1..N: an engine with only the layer-0 seams passes rungs 1-2."""
        runs = {tag: {k: d[k] for k in ("kda_out-0", "l_out-0")} for tag, d in self.runs32.items()}
        res = self.check(runs, rungs=2)
        self.assertEqual(([r.n for r, _ in res], failing(res)), ([1, 2], []))
        self.refused(runs, r"missing seams the rungs need: ffn_moe_out-1, ffn_moe_topk-1", rungs=3)


class CheckDumpsFail(_Fixture):
    # seam -> the rungs that judge it
    VALUE_SEAMS = {"kda_out-0": [1], "l_out-0": [2], "ffn_moe_out-1": [3], "mla_out-3": [4],
                   "res_probs_ffn-3": [5], "res_mix_ffn-3": [5], "result_output": [6],
                   "indexer_score-3": [4, 6], "indexer_score-8": [6]}

    def test_a_1e3_change_of_one_value_fails_the_rungs_that_judge_it(self):
        tag, t = "seq2.decode", 70                      # a decode step: the per-call concatenation is judged too
        for name, want in self.VALUE_SEAMS.items():
            with self.subTest(seam=name):
                runs = copy_runs(self.runs32)
                a = runs[tag][name]
                j = t - 3 if name.startswith("indexer_score") else min(5, a.shape[1] - 1)   # visible: finite
                self.assertTrue(np.isfinite(a[t, j]))
                a[t, j] += 1e-3
                self.assertEqual(failing(self.check(runs)), want)
        tag = "seq0.prefill"                            # the whole seam, 1e-3 relative
        for name, want in self.VALUE_SEAMS.items():
            with self.subTest(seam=name, whole=True):
                runs = copy_runs(self.runs32)
                runs[tag][name] = runs[tag][name] * np.float32(1 + 1e-3)
                self.assertEqual(failing(self.check(runs)), want)

    def test_one_changed_selection_fails(self):
        cases = {"ffn_moe_topk-1": [3, 6], "ffn_moe_topk-5": [6], "top_k-3": [4, 6], "top_k-7": [6]}
        tag, t = "seq2.prefill", 40
        for name, want in cases.items():
            with self.subTest(seam=name):
                runs = copy_runs(self.runs32)
                row = runs[tag][name][t]
                pool = range(self.m64.cfg.n_expert) if name.startswith("ffn") else range(t + 1)
                row[0] = next(i for i in pool if i not in set(row.tolist()))
                self.assertEqual(failing(self.check(runs)), want)
        runs = copy_runs(self.runs32)                       # a set of the wrong size: experts truncated
        runs[tag]["ffn_moe_topk-2"] = runs[tag]["ffn_moe_topk-2"][:, :8].copy()
        self.assertEqual(failing(self.check(runs)), [6])

    def test_non_finite_fails(self):
        runs = copy_runs(self.runs32)
        runs["seq1.prefill"]["mla_out-3"][2, 7] = np.nan
        res = self.check(runs)
        self.assertEqual(failing(res), [4])
        self.assertTrue(any("non-finite" in f for f in res[3][1].failures))

    def test_masked_indexer_cells_must_be_minus_inf(self):
        """-inf exactly where the scores are masked (future and padded keys): NaN, +inf or 0 there fails."""
        for fill, word in ((np.nan, "NaN on"), (np.inf, "+inf on"), (0.0, "not -inf on")):
            for scope in ("one", "all"):
                with self.subTest(fill=fill, scope=scope):
                    runs = copy_runs(self.runs32)
                    for tag in ([("seq2.decode")] if scope == "one" else list(runs)):
                        for L in (3, 7, 8):
                            a = runs[tag][f"indexer_score-{L}"]
                            if scope == "one":
                                a[40, 41] = fill                 # query 40's first future key
                            else:
                                a[~np.isfinite(a)] = fill
                    res = self.check(runs)
                    self.assertEqual(failing(res), [4, 6])
                    self.assertTrue(any(word in f for f in res[3][1].failures), res[3][1].failures[:3])


class CheckDumpsRefused(_Fixture):
    def renamed(self, old, new, tag=None):
        runs = copy_runs(self.runs32)
        for t in ([tag] if tag else list(runs)):
            runs[t][new] = runs[t].pop(old)
        return runs

    def test_a_seam_on_the_wrong_layer_is_refused(self):
        cases = [("kda_out-0", "kda_out-3", r"kda_out-3: wrong layer: layer 3 is an MLA layer with a MoE; "
                                            r"kda_out exists on layers 0,1,2,4,5,6"),
                 ("mla_out-3", "mla_out-0", r"mla_out-0: wrong layer: layer 0 is a KDA layer with a dense FFN; "
                                            r"mla_out exists on layers 3,7,8"),
                 ("ffn_moe_out-1", "ffn_moe_out-0", r"ffn_moe_out-0: wrong layer: layer 0 is a KDA layer with a "
                                                    r"dense FFN"),
                 ("l_out-0", "l_out-9", r"l_out-9: wrong layer: there is no layer 9 \(the model has 9\)"),
                 ("res_mix_attn-1", "res_mix_attn-0", r"res_mix_attn-0: wrong layer"),
                 ("result_output", "result_output-8", r"result_output is a whole-model seam, it has no layer"),
                 ("l_out-0", "l_out", r"l_out is a per-layer seam")]
        for old, new, pattern in cases:
            with self.subTest(new=new):
                self.refused(self.renamed(old, new, "seq2.decode"), pattern)

    def test_a_wrong_seam_name_is_refused(self):
        for old, new in (("kda_out-0", "kda_outt-0"), ("result_output", "logits"),
                         ("kda_out-0", "layers.0.attn_out"), ("ffn_moe_topk-1", "moe.1.expert_ids"),
                         ("l_out-3", "l_out-03"), ("l_out-3", "l_out-+3"), ("l_out-3", "l_out-\u00b3")):
            with self.subTest(new=new):
                self.refused(self.renamed(old, new, "seq0.prefill"), re.escape(new) + r": not a seam name of ref/")

    def test_wrong_or_missing_runs_are_refused(self):
        runs = copy_runs(self.runs32)
        runs["seq3.prefill"] = runs["seq0.prefill"]
        self.refused(runs, r"seq3.prefill: not a run of the golden")
        runs = copy_runs(self.runs32)
        runs["seq0.verify"] = runs.pop("seq0.decode")
        self.refused(runs, r"seq0.verify: not a run of the golden")
        runs = copy_runs(self.runs32)
        del runs["seq1.decode"]
        self.refused(runs, r"runs missing: seq1.decode")

    def test_missing_seams_and_wrong_token_rows_are_refused(self):
        runs = copy_runs(self.runs32)
        del runs["seq2.decode"]["top_k-8"]
        self.refused(runs, r"seq2.decode: missing seams the rungs need: top_k-8")
        runs = copy_runs(self.runs32)
        runs["seq2.decode"]["l_out-0"] = runs["seq2.decode"]["l_out-0"][:66]     # the prefill part only
        self.refused(runs, r"seq2.decode/l_out-0: shape \(66, 256\), the reference's is \(80, 256\)")
        runs = copy_runs(self.runs32)
        runs["seq0.prefill"]["kda_g1-0"] = runs["seq0.prefill"]["kda_g1-0"].reshape(32, -1)
        self.refused(runs, r"seq0.prefill/kda_g1-0: shape \(32, 192\)")
        # an unjudged seam is validated too (name, layer, its whole shape)
        self.refused(self.renamed("kda_beta-1", "kda_beta-7", "seq1.prefill"), r"kda_beta-7: wrong layer")
        runs = copy_runs(self.runs32)
        runs["seq0.prefill"]["kda_g1-0"] = runs["seq0.prefill"]["kda_g1-0"][:, :1].copy()
        self.refused(runs, r"seq0.prefill/kda_g1-0: shape \(32, 1, 32\), the reference's is \(32, 6, 32\)")
        runs = copy_runs(self.runs32)
        runs["seq1.decode"]["indexer_score-7"] = runs["seq1.decode"]["indexer_score-7"][:, :4].copy()
        self.refused(runs, r"seq1.decode/indexer_score-7: shape \(5, 4\), the reference's is \(5, 5\)")
        # only a count the model decides may differ (judged on the rung): experts truncated to 8 is a FAIL
        runs = copy_runs(self.runs32)
        for name in ("ffn_moe_topk-4", "ffn_moe_weights-4", "ffn_moe_weights_norm-4", "ffn_moe_weights_scaled-4"):
            runs["seq0.prefill"][name] = runs["seq0.prefill"][name][:, :8].copy()
        self.assertEqual(failing(self.check(runs)), [6])
        runs["seq0.prefill"]["ffn_moe_topk-4"] = runs["seq0.prefill"]["ffn_moe_topk-4"][:5].copy()
        self.refused(runs, r"seq0.prefill/ffn_moe_topk-4: shape \(5, 8\).*only the last axis, a count, may differ")

    def test_malformed_files_are_refused(self):
        def fresh(name, fmt="raw"):
            d = self.root / name
            G.write_dumps(self.runs32, d, fmt)
            return d
        d = fresh("short")
        p = d / "seq0.prefill" / "l_out-0.f32"
        p.write_bytes(p.read_bytes()[:-4])
        with self.assertRaisesRegex(G.DumpsRefused, r"seq0.prefill/l_out-0.f32: \d+ bytes, the manifest says"):
            G.load_dumps(d)
        d = fresh("stray")
        (d / "seq0.prefill" / "extra-3.f32").write_bytes(b"\0" * 4)
        with self.assertRaisesRegex(G.DumpsRefused, r"seq0.prefill/: extra-3 not in manifest.txt"):
            G.load_dumps(d)
        d = fresh("badline")
        mf = d / "seq1.decode" / "manifest.txt"
        mf.write_text(mf.read_text() + "l_out-0 256 x\n")
        with self.assertRaisesRegex(G.DumpsRefused, r"is not `name d0 d1 \.\.\.`"):
            G.load_dumps(d)
        d = fresh("twice")
        mf = d / "seq1.decode" / "manifest.txt"
        mf.write_text(mf.read_text() + mf.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(G.DumpsRefused, r"listed twice"):
            G.load_dumps(d)
        d = fresh("nomanifest")
        (d / "seq1.decode" / "manifest.txt").unlink()
        with self.assertRaisesRegex(G.DumpsRefused, r"seq1.decode/: no manifest.txt"):
            G.load_dumps(d)
        d = fresh("both")
        np.savez(d / "seq0.prefill.npz", **self.runs32["seq0.prefill"])
        with self.assertRaisesRegex(G.DumpsRefused, r"seq0.prefill: dumped both as a directory and as an .npz"):
            G.load_dumps(d)
        d = fresh("floatids", "npz")
        arrs = dict(self.runs32["seq1.prefill"])
        arrs["ffn_moe_topk-1"] = arrs["ffn_moe_topk-1"].astype(np.float32)
        np.savez(d / "seq1.prefill.npz", **arrs)
        with self.assertRaisesRegex(G.DumpsRefused, r"ffn_moe_topk-1 is float32; integer expected"):
            G.load_dumps(d)
        for junk, pattern in ((b"not an npz", r"UnpicklingError|ValueError"), (b"PK\x03\x04 truncated", r"BadZipFile"),
                              (None, r"ValueError")):
            with self.subTest(junk=junk):
                d = fresh(f"junk{len(junk) if junk else 0}", "npz")
                p = d / "seq1.prefill.npz"
                if junk is None:                          # a pickled object array in a valid .npz
                    np.savez(p, **{"l_out-0": np.array([{"a": 1}], dtype=object)})
                else:
                    p.write_bytes(junk)
                with self.assertRaisesRegex(G.DumpsRefused, rf"seq1.prefill.npz: cannot be read \(({pattern})"):
                    G.load_dumps(d)
        (self.root / "empty").mkdir()
        (self.root / "empty" / "engine.log").write_text("not a run\n")
        with self.assertRaisesRegex(G.DumpsRefused, r"no <seq>.<mode>/ or <seq>.<mode>.npz dumps"):
            G.load_dumps(self.root / "empty")


class CheckDumpsMutations(_Fixture):
    """Every M1a mutation switch, applied to the float32 reference, fails first on its rung."""

    table: list = []

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.table:
            w = max(len(r[0]) for r in cls.table)
            lines = [f"{'mutation':{w}s}  rungs failed     first rung: its seams' max / bound"]
            lines += [f"{n:{w}s}  {','.join(map(str, f)):15s}  {moved}" for n, f, moved in sorted(cls.table)]
            print("\nKimi K3.1 check-dumps, M1a mutations (float32 reference dumps):\n" + "\n".join(lines),
                  file=sys.stderr)

    def run_mutation(self, name, first):
        """A "ref" mutation edits the reference; a "synth" one the fixture's converter, whose GGUF the
        unmodified reference then reads. Either way the dumps are judged against the correct fixture."""
        what, target, edits, eps = O.MUTATIONS[name]
        path = self.gguf
        if target == "synth":
            smod = O.load_mutant(O.SYNTH_SRC, edits, name)
            vals, _, _ = smod.make_weights(smod.TINY, smod.SEED)
            md, ts = smod.to_gguf(vals, smod.TINY)
            path = self.root / f"{name}.gguf"
            smod.write_gguf(path, md, ts)
            edits = []
        else:
            self.assertEqual(target, "ref")
        m = O.load_mutant(O.REF_SRC, edits, name).KimiK31(path, dtype="float32", eps_overrides=eps)
        try:
            runs = self.dumps(m)
        finally:
            m.close()
        res = self.check(runs)
        bad = failing(res)
        rep = res[first - 1][1]
        moved = {label: s["max"] / s["atol"] for label, s in rep.stats.get("seams", {}).items()}
        self.table.append((name, bad, ", ".join(f"{k} {v:.0f}x" for k, v in moved.items())))
        self.assertTrue(bad, f"mutation {name} ({what}) passed check-dumps")
        self.assertEqual(bad[0], first, f"mutation {name} ({what}): first failing rung\n"
                         + "\n".join(G.rung_summary(r, p) for r, p in res))
        self.assertTrue(moved and max(moved.values()) >= 10.0, f"{name}: rung {first} moved only {moved}")


def _add_mutation_tests():
    for name, first in M1A_MUTATIONS.items():
        def t(self, name=name, first=first):
            self.run_mutation(name, first)
        t.__doc__ = f"{O.MUTATIONS[name][0]} (fails first on rung {first})"
        setattr(CheckDumpsMutations, f"test_mutation_{name.replace('-', '_')}", t)


_add_mutation_tests()


class CheckDumpsCli(_Fixture):
    def cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(GATE), *args, "--gguf", str(self.gguf)],
                              capture_output=True, text=True, timeout=600)

    def test_dump_ref_then_check_dumps(self):
        d = self.root / "cli"
        r = self.cli("dump-ref", "--out", str(d))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.cli("check-dumps", "--dumps", str(d), "--dtype", "float32")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.splitlines()[-1], "check-dumps PASS")
        self.assertEqual(r.stdout.count(" PASS: "), 6, r.stdout)
        r = self.cli("check-engine", "--dumps", str(d), "--rungs", "2")        # the plan's name; rungs 1-2 only
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("check-dumps PASS", r.stdout)
        self.assertEqual(r.stdout.splitlines()[-1], "check-dumps rungs 1-2 of 6 pass (rungs 3-6 not judged)")
        p = d / "seq0.decode" / "kda_out-0.f32"                     # one value, +1e-3
        v = np.fromfile(p, dtype=np.float32)
        v[300] += np.float32(1e-3)
        v.tofile(p)
        r = self.cli("check-dumps", "--dumps", str(d))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("check-dumps FAIL (failing rungs [1], first 1)", r.stdout)
        mf = d / "seq0.decode" / "manifest.txt"                     # kda_out-0 renamed onto an MLA layer
        mf.write_text(mf.read_text().replace("kda_out-0 ", "kda_out-3 "))
        p.rename(p.with_name("kda_out-3.f32"))
        r = self.cli("check-dumps", "--dumps", str(d))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("check-dumps: refused: seq0.decode/kda_out-3: wrong layer", r.stdout)
        j = self.root / "cli-junk"                                    # an unreadable .npz: refused, not a FAIL
        j.mkdir()
        (j / "seq1.prefill.npz").write_bytes(b"junk")
        r = self.cli("check-dumps", "--dumps", str(j))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("check-dumps: refused: seq1.prefill.npz: cannot be read", r.stdout)

    def test_a_gguf_of_another_geometry_is_refused(self):
        self.refused(self.runs32, r"not the big fixture geometry \(--geometry big\)", geometry="big")
        for cmd in (["check-dumps", "--dumps", str(self.root)], ["dump-ref", "--out", str(self.root / "no")]):
            r = self.cli(*cmd, "--geometry", "big")
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn(f"{cmd[0]}: refused: the GGUF is not the big fixture geometry", r.stdout)
            self.assertIn("expert_count 24 != 64", r.stdout)
        self.assertFalse((self.root / "no").exists())

    def test_another_gguf_is_refused(self):
        other = self.root / "other.gguf"
        b = bytearray(self.gguf.read_bytes())
        b[-1] ^= 1
        other.write_bytes(bytes(b))
        r = subprocess.run([sys.executable, str(GATE), "check-dumps", "--dumps", str(self.root), "--gguf", str(other)],
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("is not the GGUF the golden", r.stdout)

    def test_a_gguf_of_another_size_is_refused_unread(self):
        """A real K3.1 GGUF (100+ GB) is refused by its size, before any hashing."""
        other = self.root / "bigger.gguf"
        other.write_bytes(self.gguf.read_bytes() + b"\0")
        meta = self.golden[0]
        real, calls = G.sha256, []
        G.sha256 = lambda p: calls.append(p) or real(p)
        try:
            self.assertFalse(G.is_fixture_gguf(other, meta))
            self.assertEqual(calls, [])
            self.assertTrue(G.is_fixture_gguf(self.gguf, meta))
            self.assertEqual(calls, [self.gguf])
        finally:
            G.sha256 = real

    def test_a_file_that_is_no_gguf_is_refused(self):
        """Not a GGUF - missing, empty, junk, cut inside its header: a refusal (exit 2, no traceback) from both
        commands on both geometries, before any tensor is opened."""
        bad = self.root / "bad"
        bad.mkdir(exist_ok=True)
        (bad / "empty.gguf").write_bytes(b"")
        (bad / "junk.gguf").write_bytes(b"not a gguf " * 64)
        (bad / "cut.gguf").write_bytes(self.gguf.read_bytes()[:200])
        cases = {"missing.gguf": "no such file", "empty.gguf": "cannot be read as a GGUF",
                 "junk.gguf": "cannot be read as a GGUF (ValueError: junk.gguf: not a GGUF file",
                 "cut.gguf": "cannot be read as a GGUF"}
        for name, why in cases.items():
            for geo in ("tiny", "big"):
                for cmd in (["check-dumps", "--dumps", str(self.root)], ["dump-ref", "--out", str(bad / "out")]):
                    with self.subTest(gguf=name, geometry=geo, cmd=cmd[0]):
                        r = subprocess.run([sys.executable, str(GATE), *cmd, "--geometry", geo,
                                            "--gguf", str(bad / name)], capture_output=True, text=True, timeout=600)
                        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                        self.assertNotIn("Traceback", r.stderr)
                        self.assertIn(f"{cmd[0]}: refused: {bad / name}: {why}", r.stdout)
        self.assertFalse((bad / "out").exists())

    def main(self, *argv) -> tuple[int, str]:
        """G.main in this process: an exception (what would be a traceback) fails the test."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = G.main([str(a) for a in argv])
        return rc, out.getvalue()

    def test_a_corrupt_header_or_cut_data_is_refused(self):
        """GGUF magic and v3, then a header no reader can hold (a 2^62 or 2^64-1 string length, a 2^61-element
        array, a 2^32-1-dimension tensor, random bytes): a refusal on both geometries, never a MemoryError or
        OverflowError. Past the header: a tensor type ref/load.py does not decode, or tensor data cut short
        (the fixture's first 90%), is a refusal from both commands too - dump-ref's before any tensor is read."""
        import struct
        from gguf_reader import GGUFFile
        bad = self.root / "corrupt"
        bad.mkdir(exist_ok=True)
        head = struct.pack("<IIQQ", 0x46554747, 3, 0, 1)              # magic, v3, 0 tensors, 1 key
        files = {"hugelen.gguf": head + struct.pack("<Q", 1 << 62),
                 "overflow.gguf": head + struct.pack("<Q", (1 << 64) - 1),
                 "hugearray.gguf": head + struct.pack("<Q", 1) + b"k" + struct.pack("<IIQ", 9, 4, 1 << 61),
                 "hugedims.gguf": struct.pack("<IIQQ", 0x46554747, 3, 1, 0) + struct.pack("<Q", 1) + b"t"
                                  + struct.pack("<I", (1 << 32) - 1),
                 "randtail.gguf": struct.pack("<II", 0x46554747, 3) + np.random.default_rng(0).bytes(4096)}
        for name, b in files.items():
            (bad / name).write_bytes(b)
        src = self.gguf.read_bytes()
        (bad / "cutdata.gguf").write_bytes(src[:len(src) * 9 // 10])
        t = GGUFFile(self.gguf).tensors[0]
        key = struct.pack("<Q", len(t.name.encode())) + t.name.encode()
        at = src.index(key) + len(key) + 4 + 8 * len(t.shape)        # its type id, after n_dims and the dims
        b = bytearray(src)
        b[at:at + 4] = struct.pack("<I", 99)
        (bad / "badtype.gguf").write_bytes(bytes(b))
        self.assertEqual(GGUFFile(bad / "badtype.gguf").tensors[0].type_name, "type99")
        cases = {name: (("tiny", "big"), "cannot be read as a GGUF (ValueError: ") for name in files}
        cases["cutdata.gguf"] = (("tiny",), "tensor ")                 # (on big: its geometry, from the header)
        cases["badtype.gguf"] = (("tiny",), f"tensor {t.name} is type99, a type ref/load.py does not decode")
        for name, (geos, why) in cases.items():
            for geo in geos:
                for cmd in (["check-dumps", "--dumps", self.root], ["dump-ref", "--out", bad / "out"]):
                    with self.subTest(gguf=name, geometry=geo, cmd=cmd[0]):
                        rc, out = self.main(*cmd, "--geometry", geo, "--gguf", bad / name)
                        self.assertEqual(rc, 2, out)
                        self.assertIn(f"{cmd[0]}: refused: {bad / name}: {why}", out)
                        if name == "cutdata.gguf":
                            self.assertRegex(out, r"past the end of the file \(\d+ bytes\): its data is cut")
        self.assertFalse((bad / "out").exists())

    def test_golden_is_refused_on_big(self):
        """The big fixture has no golden: an explicit --golden is refused, not silently ignored."""
        for cmd in (["check-dumps", "--dumps", str(self.root)], ["dump-ref", "--out", str(self.root / "no-golden")]):
            with self.subTest(cmd=cmd[0]):
                r = self.cli(*cmd, "--geometry", "big", "--golden", str(G.GOLDEN_DIR))
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn(f"{cmd[0]}: refused: --golden: the big fixture has no golden", r.stdout)
        self.assertFalse((self.root / "no-golden").exists())
        d = self.root / "cli-golden"                                   # tiny: an explicit --golden is its own
        G.write_dumps(self.runs32, d, "npz")
        rc, out = self.main("check-dumps", "--dumps", d, "--golden", G.GOLDEN_DIR, "--gguf", self.gguf)
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.splitlines()[-1], "check-dumps PASS")

    def test_big_refuses_another_geometry_unhashed(self):
        """--geometry big reads the header first: a GGUF of another geometry (a real K3.1 one) is never hashed,
        and no tensor of it is opened - neither the reference model nor ref/load.py's reader is built."""
        import load

        def opened(*args, **kw):
            raise AssertionError("a tensor reader was built before the geometry was checked")
        real, calls = G.sha256, []
        patches = [(G, "sha256", lambda p: calls.append(p) or real(p)), (kimi_k31, "KimiK31", opened),
                   (kimi_k31, "Gguf", opened), (load, "Gguf", opened)]
        saved = [(mod, attr, getattr(mod, attr)) for mod, attr, _ in patches]
        for mod, attr, fn in patches:
            setattr(mod, attr, fn)
        try:
            for cmd in (["check-dumps", "--dumps", self.root], ["dump-ref", "--out", self.root / "no-big"]):
                with self.subTest(cmd=cmd[0]):
                    rc, out = self.main(*cmd, "--geometry", "big", "--gguf", self.gguf)
                    self.assertEqual(rc, 2, out)
                    self.assertIn(f"{cmd[0]}: refused: the GGUF is not the big fixture geometry", out)
        finally:
            for mod, attr, fn in saved:
                setattr(mod, attr, fn)
        self.assertEqual(calls, [])
        self.assertFalse((self.root / "no-big").exists())

    def test_a_real_error_is_raised_not_refused(self):
        """Once the refusals have passed, an error in the reference or the judging is a real one: it is raised
        (a traceback, exit 1 from the shell), never printed as `refused`."""
        def boom(*args, **kw):
            raise ValueError("boom")
        cases = (("reference_dumps", ["dump-ref", "--out", str(self.root / "boom")]),
                 ("check_dumps", ["check-dumps", "--dumps", str(self.root / "cli-real")]))
        G.write_dumps(self.runs32, self.root / "cli-real", "npz")
        for fn, cmd in cases:
            with self.subTest(fn=fn):
                real = getattr(G, fn)
                setattr(G, fn, boom)
                out = io.StringIO()
                try:
                    with contextlib.redirect_stdout(out), self.assertRaisesRegex(ValueError, "^boom$"):
                        G.main([*cmd, "--gguf", str(self.gguf)])
                finally:
                    setattr(G, fn, real)
                self.assertNotIn("refused", out.getvalue())
        self.assertFalse((self.root / "boom").exists())


BIG_DIR = os.environ.get("KIMI_K31_BIG_FIXTURE")


@unittest.skipUnless(BIG_DIR, "KIMI_K31_BIG_FIXTURE not set (the directory of a kimi_k31_synth_gguf.py --geometry "
                              "big build)")
class CheckDumpsBig(unittest.TestCase):
    """The big fixture has no golden: every seam is judged against the float64 reference, on its own sequences,
    with DUMP_TOL / DUMP_REF64_TOL (2x the float32 reference's gap) and its tie bands."""

    @classmethod
    def setUpClass(cls):
        cls.gguf = pathlib.Path(BIG_DIR).expanduser() / "big.gguf"
        cls.tmp = tempfile.TemporaryDirectory(prefix="strata-kimi-k31-big-dumps-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = pathlib.Path(cls.tmp.name)
        cls.m64, cls.expected, cls.runs32 = {}, {}, {}
        for eps in kimi_k31.EPS_PRESETS:
            cls.m64[eps] = kimi_k31.KimiK31(cls.gguf, dtype="float64", eps=eps, cache_weights=True)
            cls.addClassCleanup(cls.m64[eps].close)
            cls.expected[eps] = cls.dumps(cls.m64[eps])
            m = kimi_k31.KimiK31(cls.gguf, dtype="float32", eps=eps, cache_weights=True)
            try:
                cls.runs32[eps] = cls.dumps(m)
            finally:
                m.close()

    @classmethod
    def dumps(cls, m) -> dict:
        return {tag: G.reference_dumps(m, toks, df) for tag, toks, df in G.dump_runs("big")}

    def check(self, runs, eps="moonshot", **kw):
        return G.check_dumps(runs, self.m64[eps], dtype="float32", expected=self.expected[eps], geometry="big", **kw)

    def test_reference_float32_dumps_pass_every_rung(self):
        self.assertEqual(sorted(self.runs32["moonshot"]), sorted(f"{s}.{m}" for s in ("seq0", "seq1", "seq2")
                                                                for m in G.MODES))
        for eps in kimi_k31.EPS_PRESETS:
            for reference in ("golden", "ref64"):        # no golden here: the same verdicts either way
                with self.subTest(eps=eps, reference=reference):
                    res = self.check(self.runs32[eps], eps, reference=reference)
                    self.assertEqual(failing(res), [], "\n".join(G.rung_summary(r, p) for r, p in res))
                    self.assertIsNone(res[0][1].stats["golden_run"])
                    self.assertEqual({v for _, p in res for s in p.stats["seams"].values() for v in s["vs"]},
                                     {G.REF64})
                    last = res[-1][1].stats
                    self.assertEqual((last.get("moe_rows_differ", 0), last.get("idx_rows_differ", 0)), (0, 0))
                    self.assertEqual((last["moe_rows"], last["idx_rows"]), (7704, 2568))     # 6 runs, 428 tokens

    def test_bounds_are_twice_the_measured_gap(self):
        """Each bound is 1.9-2.5x the float32 reference's gap, the larger of the two eps presets (DUMP_TOL). Rounded
        up, they were >= 2.00x where measured; the BLAS thread count moves the gap a little, hence 1.9x here."""
        gap, bound = {}, {}
        for eps in kimi_k31.EPS_PRESETS:
            for _, rep in self.check(self.runs32[eps], eps):
                for label, s in rep.stats["seams"].items():
                    g = gap.setdefault(label, [0.0, 0.0])
                    g[0], g[1] = max(g[0], s["max"]), max(g[1], s["rms"])
                    bound[label] = (s["atol"], s["rms_bound"])
                g = gap.setdefault("indexer scores", [0.0, None])
                g[0] = max(g[0], rep.stats.get("idx_score_max", 0.0))
                bound["indexer scores"] = (G.DUMP_TOL["big"]["idx_score_atol"], None)
        self.assertEqual(len(gap), 8, sorted(gap))
        for label, (mx, rms) in gap.items():
            atol, rms_bound = bound[label]
            print(f"big {label}: f32 vs f64 max {mx:.2e} (bound {atol:.1e}, {atol / mx:.2f}x)"
                  + (f" rms {rms:.2e} (bound {rms_bound:.1e}, {rms_bound / rms:.2f}x)" if rms else ""), file=sys.stderr)
            self.assertGreaterEqual(atol, 1.9 * mx, label)
            self.assertLessEqual(atol, 2.5 * mx, label)
            if rms is not None:
                self.assertGreaterEqual(rms_bound, 1.9 * rms, label)
                self.assertLessEqual(rms_bound, 2.5 * rms, label)
        # tiny's float32 bounds would not hold here (the logits and indexer-score gaps exceed them)
        self.assertGreater(gap["logits"][0], G.TOL["float32"]["logits_atol"])
        self.assertGreater(gap["indexer scores"][0], G.IDX_SCORE_ATOL)

    def test_a_1e3_change_of_one_value_fails_its_rung(self):
        for seam, rung in (("kda_out-0", 1), ("res_mix_ffn-3", 5), ("result_output", 6)):
            with self.subTest(seam=seam):
                runs = copy_runs(self.runs32["moonshot"])
                runs["seq2.decode"][seam][270, 5] += np.float32(1e-3) * (100 if seam == "result_output" else 1)
                self.assertEqual(failing(self.check(runs)), [rung])
        runs = copy_runs(self.runs32["moonshot"])
        ids = runs["seq0.prefill"]["ffn_moe_topk-7"]
        ids[50, 0] = next(e for e in range(64) if e not in ids[50])
        self.assertEqual(failing(self.check(runs)), [6])
        # one preset's dumps judged under the other fail too: on big first on rung 4 (the q_a/kv_a eps); the KDA l2
        # eps stays inside rung 1's bound here (on tiny it fails rung 1)
        for dumps_eps, ref_eps in (("llama.cpp", "moonshot"), ("moonshot", "llama.cpp")):
            with self.subTest(dumps=dumps_eps, reference=ref_eps):
                bad = failing(self.check(self.runs32[dumps_eps], ref_eps))
                self.assertTrue(bad)
                self.assertEqual(bad[0], 4, bad)

    def test_cli(self):
        d = self.root / "cli"
        r = subprocess.run([sys.executable, str(GATE), "dump-ref", "--geometry", "big", "--gguf", str(self.gguf),
                            "--out", str(d), "--format", "npz"], capture_output=True, text=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = subprocess.run([sys.executable, str(GATE), "check-engine", "--geometry", "big", "--gguf", str(self.gguf),
                            "--dumps", str(d)], capture_output=True, text=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("big fixture", r.stdout.splitlines()[0])
        self.assertIn("golden run none", r.stdout.splitlines()[0])
        self.assertEqual(r.stdout.splitlines()[-1], "check-dumps PASS")
        r = subprocess.run([sys.executable, str(GATE), "check-dumps", "--gguf", str(self.gguf), "--dumps", str(d)],
                           capture_output=True, text=True, timeout=600)    # tiny (the golden's) on the big GGUF:
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)                  # refused on its header, unhashed
        self.assertIn("check-dumps: refused: the GGUF is not the tiny fixture geometry (--geometry tiny)", r.stdout)


if __name__ == "__main__":
    unittest.main()
