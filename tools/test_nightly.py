"""tools/test_nightly.py - tools/maya_nightly.py: the plan, the budget, the verdicts and a whole night with a
stand-in for the engine run (no GPU, no git).

    python -m unittest tools.test_nightly -v
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import maya_golden as G  # noqa: E402
import maya_nightly as N  # noqa: E402
from test_golden import CFG, changed, tokens  # noqa: E402

LINE = ("compare 8c618a59 (v1.1.76) vs 8c618a59 (v1.1.76) | exact: yes | decode 61.2 vs 61.0 +0.3% | "
        "2K 456 vs 458 -0.4% | 8K 640 vs 700 -8.6%")


class Parts(unittest.TestCase):
    def test_compare_parts(self):
        self.assertEqual(N.compare_parts(LINE), ("yes", {"decode": 0.3, "2K": -0.4, "8K": -8.6}))
        self.assertEqual(N.compare_parts("")[0], "?")

    def test_verdicts(self):
        ok = {"status": "PASS", "diffs": [], "notes": []}
        bad = {"status": "DIFF", "diffs": ["decode 2: differs from token 17 (...)", "x"], "notes": ["n"]}
        self.assertEqual(N.verdict("yes", {"decode": 0.3}, ok, 5), ("PASS", []))
        self.assertEqual(N.verdict("yes", {"decode": -4.9}, ok, 5)[0], "PASS")
        st, why = N.verdict("yes", {"decode": 0.3, "8K": -8.6}, ok, 5)
        self.assertEqual((st, why), ("SLOW", ["slower than the previous release: 8K -8.6%"]))
        st, why = N.verdict("NO", {}, ok, 5)
        self.assertEqual((st, why[0]), ("DIFF", "answers differ from the previous release"))
        st, why = N.verdict("yes", {"decode": -9.0}, bad, 5)
        self.assertEqual(st, "DIFF")
        self.assertIn("saved answers: decode 2: differs from token 17 (...) (+1 more)", why)
        st, why = N.verdict("yes", {}, None, 5)                       # (nothing saved yet: said, not a failure)
        self.assertEqual(st, "PASS")
        self.assertIn("no saved answers yet", why[0])
        self.assertEqual(N.verdict("yes in 2 of 4 pairings (the answers vary between starts)", {}, ok, 5)[0], "PASS")
        self.assertEqual(N.verdict("?", {}, ok, 5)[0], "DIFF")

    def test_budget_uses_last_times_and_skips_what_does_not_fit(self):
        m = lambda n: {"name": n, "config": n + ".json"}  # noqa: E731
        st = dict(N.DEFAULTS, budget_min=10, est_min=6)
        old = [N.result_line("2026-11-23", "box", m("A"), "c", "v1", "yes", {}, "PASS 5/5", 200, "PASS", []),
               N.result_line("2026-11-23", "box", m("B"), "c", "v1", "-", {}, "-", 3, "BUSY", ["gpu"]),
               N.result_line("2026-11-23", "other", m("C"), "c", "v1", "yes", {}, "-", 30, "PASS", [])]
        self.assertEqual(N.last_seconds(old, "box", "A"), 200)
        self.assertIsNone(N.last_seconds(old, "box", "B"))            # (a BUSY line is not a run time)
        self.assertIsNone(N.last_seconds(old, "box", "C"))            # (another machine's)
        rows, total = N.budget([m("A"), m("B"), m("C")], st, old, "box")
        self.assertEqual([(r[0]["name"], r[1], r[2]) for r in rows], [("A", 200, True), ("B", 360, True),
                                                                      ("C", 360, False)])
        self.assertEqual(total, 560)
        text = "\n".join(N.budget_lines(rows, total, st))
        self.assertIn("time budget: 9 of 10 min for 2 model(s)", text)
        self.assertIn("C: about 6.0 min - does not fit tonight: SKIPPED", text)

    def test_status_text_and_exit_code(self):
        rows = [("A", "one", "PASS", [], 200.0), ("B", "", "DIFF", ["answers differ"], 100.0)]
        t = N.status_text("2026-11-24", "box", "abc", "v1.1.76", rows, 300)
        self.assertTrue(t.startswith("FAIL | 2026-11-24 | box | abc vs v1.1.76 | 2 of 2 model(s) ran | 5.0 min"))
        self.assertIn("  DIFF    B (-) 100 s: answers differ", t)
        self.assertEqual(N.exit_code(rows), 1)
        self.assertTrue(N.status_text("d", "b", "c", "v", rows[:1], 1).startswith("PASS"))
        busy = [("A", "", "BUSY", ["GPU 0 is in use"], 1.0)]
        self.assertTrue(N.status_text("d", "b", "c", "v", busy, 1).startswith("NOT RUN"))
        self.assertEqual(N.exit_code(busy), 3)
        self.assertTrue(N.status_text("d", "b", "c", "v", rows[:1] + busy, 1).startswith("PARTIAL"))
        self.assertEqual(N.exit_code(rows[:1] + busy), 0)

    def test_plan_needs_names_and_configs(self):
        with self.assertRaises(ValueError):
            N.load_plan({"models": [{"name": "A"}]})
        st, models = N.load_plan({"budget_min": 7, "models": [{"name": "A", "config": "a"},
                                                              {"name": "B", "config": "b"}]}, only=["b"])
        self.assertEqual((st["budget_min"], st["rounds"], [m["name"] for m in models]), (7, 2, ["B"]))

    def test_config_gpus(self):
        self.assertEqual([N.config_gpus({"gpu": g}) for g in (2, [0, 2], "0,2", "", None)],
                         [[2], [0, 2], [0, 2], [], []])


class Night(unittest.TestCase):
    """A whole night in a temporary folder: the engine run replaced by one that writes a tokens file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "maya-a.json").write_text(json.dumps(CFG))
        self.doc = {"base": "v1.1.76", "wait_min": 0, "models": [{"name": "A", "config": "maya-a.json", "layout": "one",
                                                   "locks": [str(self.root / "lock-v100")]}]}
        self.said = []
        self.tok = tokens(config=str(self.root / "maya-a.json"), records=[LINE.replace("-8.6%", "-0.6%")])
        self.lock_seen = None

    def tearDown(self):
        self.tmp.cleanup()

    def fake_bench(self, tok):
        def bench(root, m, cfg_path, base, st, tokens_file, log):
            self.lock_seen = (root / "lock-v100").exists()
            Path(log).write_text("bench output\n")
            tokens_file.write_text(json.dumps(tok))
            return 0
        return bench

    def night(self, tok=None, busy=lambda cfg, m: [], **kw):
        run = lambda root, m, base, st, say: N.run_model(root, m, base, st, say,  # noqa: E731
                                                         bench=self.fake_bench(tok or self.tok), busy=busy)
        return N.night(self.root, self.doc, self.said.append, run=run, **kw)

    def capture(self):
        G.capture(self.tok, CFG, G.golden_dir(self.root), "one")

    def test_pass_with_saved_answers(self):
        self.capture()
        self.assertEqual(self.night(), 0)
        line = (self.root / N.RESULTS).read_text().strip()
        self.assertIn("| A | one |", line)
        self.assertIn("exact: yes | speed: decode +0.3% 2K -0.4% 8K -0.6% | saved answers: PASS 5/5 |", line)
        self.assertTrue(line.endswith("s | PASS"))
        self.assertTrue((self.root / N.STATUS).read_text().startswith("PASS |"))
        self.assertTrue(self.lock_seen)                               # (its extra lock held during the run ...)
        self.assertFalse((self.root / "lock-v100").exists())          # (... and released after)
        self.assertFalse((self.root / N.RUN_LOCK).exists())

    def test_a_changed_answer_fails_the_night_and_says_where(self):
        self.capture()
        self.assertEqual(self.night(tok=changed(self.tok, "decode 2", 17, 999)), 1)
        status = (self.root / N.STATUS).read_text()
        self.assertTrue(status.startswith("FAIL |"))
        self.assertIn("saved answers: decode 2: differs from token 17 (saved 217, now 999", status)

    def test_no_saved_answers_is_said_not_failed(self):
        self.assertEqual(self.night(), 0)
        self.assertIn("no saved answers yet", (self.root / N.STATUS).read_text())

    def test_busy_machine_starts_nothing(self):
        self.assertEqual(self.night(busy=lambda cfg, m: ["GPU 1 is in use (pid 7 python)"]), 3)
        self.assertIsNone(self.lock_seen)                             # (the engine run never started)
        status = (self.root / N.STATUS).read_text()
        self.assertTrue(status.startswith("NOT RUN |"))
        self.assertIn("BUSY    A (one)", status)

    def test_a_held_extra_lock_is_busy_and_stays(self):
        (self.root / "lock-v100").write_text("another helper\n")
        cfg = json.loads((self.root / "maya-a.json").read_text())
        self.assertEqual(N.extra_locks_held(N.extra_locks(cfg, self.doc["models"][0])),
                         [f"lock {self.root / 'lock-v100'} is held: another helper"])
        self.assertEqual(self.night(busy=lambda cfg, m: N.extra_locks_held(N.extra_locks(cfg, m))), 3)
        self.assertEqual((self.root / "lock-v100").read_text(), "another helper\n")

    def slow_then(self, second_line):
        """A night whose first compare run reads slow (answers equal) and whose second reads `second_line`."""
        self.capture()
        lines, calls = [LINE.replace("+0.3%", "-28.0%").replace("-8.6%", "-0.6%"), second_line], []

        def bench(root, m, cfg_path, base, st, tokens_file, log):
            calls.append(1)
            Path(log).write_text("bench output\n")
            tokens_file.write_text(json.dumps(dict(self.tok, records=[lines[len(calls) - 1]])))
            return 0
        run = lambda root, m, base, st, say: N.run_model(root, m, base, st, say, bench=bench,  # noqa: E731
                                                         busy=lambda c, m: [])
        code = N.night(self.root, self.doc, self.said.append, run=run)
        return code, len(calls), (self.root / N.STATUS).read_text()

    def test_a_slow_run_is_run_again_before_it_is_called_slow(self):
        code, calls, status = self.slow_then(LINE.replace("-8.6%", "-0.6%"))
        self.assertEqual((code, calls), (0, 2))
        self.assertIn("PASS    A (one)", status)
        self.assertIn("first try decode -28.0%, not on the second try (a disturbed run)", status)

    def test_slow_twice_is_slow(self):
        code, calls, status = self.slow_then(LINE.replace("+0.3%", "-27.0%").replace("-8.6%", "-0.6%"))
        self.assertEqual((code, calls), (1, 2))
        self.assertIn("SLOW    A (one)", status)
        self.assertIn("first try decode -28.0% - run again; slower than the previous release: decode -27.0%", status)

    def test_a_failed_bench_is_an_error_line(self):
        def bench(root, m, cfg_path, base, st, tokens_file, log):
            Path(log).write_text("the engine stopped while loading\n")
            return 1
        run = lambda root, m, base, st, say: N.run_model(root, m, base, st, say, bench=bench,  # noqa: E731
                                                         busy=lambda c, m: [])
        self.assertEqual(N.night(self.root, self.doc, self.said.append, run=run), 1)
        self.assertIn("ERROR   A (one)", (self.root / N.STATUS).read_text())
        self.assertIn("the engine stopped while loading", (self.root / N.STATUS).read_text())

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self.night(dry=True), 0)
        self.assertFalse((self.root / N.RESULTS).exists() or (self.root / N.STATUS).exists())
        text = "\n".join(self.said)
        self.assertIn("--bench --bench-tokens <tokens> --bench-wait 0 --bench-text-ref v1.1.76 --compare v1.1.76", text)
        self.assertIn("time budget: 6 of 20 min for 1 model(s)", text)
        self.assertIn("dry run: nothing started", text)

    def test_smoke_model_checks_the_saved_answers_alone(self):
        # (a model too large for its box: the short run, no other build - PASS on equal answers, DIFF on others,
        #  and a run that cannot answer is the bench's own failure = ERROR, as #1200 would have been)
        self.doc["models"][0]["smoke"] = True
        tok = dict(self.tok, records=[], smoke=True)
        self.assertEqual(self.night(tok=tok, dry=True), 0)
        self.assertIn("--bench --bench-tokens <tokens> --bench-wait 0 --bench-smoke", "\n".join(self.said))
        self.assertNotIn("--compare", "\n".join(self.said))
        self.assertEqual(self.night(tok=tok), 0)                    # nothing saved yet: a note, it answered
        self.assertIn("smoke: it loads and answers", (self.root / N.STATUS).read_text())
        G.capture(tok, CFG, G.golden_dir(self.root), "one")
        self.assertEqual(self.night(tok=tok), 0)
        line = (self.root / N.RESULTS).read_text().strip().splitlines()[-1]
        self.assertIn("exact: - |", line)
        self.assertIn("s | PASS: smoke: it loads and answers, against the saved answers", line)
        other =json.loads(json.dumps(tok))
        name = sorted(other["runs"]["this"][0])[0]
        other["runs"]["this"][0][name] = other["runs"]["this"][0][name][:-1] + [7]
        self.assertEqual(self.night(tok=other), 1)
        self.assertIn("DIFF    A (one)", (self.root / N.STATUS).read_text())
        # (a layout whose answers vary between starts: said, not a DIFF)
        self.doc["models"][0]["answers_vary"] = True
        self.assertEqual(self.night(tok=other), 0)
        self.assertIn("saved answers: 4 of 5 equal - this layout's answers vary between starts",
                      (self.root / N.STATUS).read_text())

    def test_over_budget_is_skipped(self):
        self.doc["budget_min"] = 1
        self.assertEqual(self.night(), 3)
        self.assertIn("SKIPPED A (one)", (self.root / N.STATUS).read_text())

    def test_one_nightly_at_a_time(self):
        import os
        (self.root / N.RUN_LOCK).write_text(f"{os.getppid()} now\n")   # (a live process that is not us)
        self.assertEqual(self.night(), 3)
        self.assertIn("another nightly is running", "\n".join(self.said))
        (self.root / N.RUN_LOCK).write_text("999999999 old\n")         # (a dead one: taken over)
        self.assertEqual(self.night(), 0)


if __name__ == "__main__":
    unittest.main()
