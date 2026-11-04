"""Tests for the speed calibration (tools/calibrate_draft.py, and maya.py's speed_calibration around it) without a GPU:
stand-in engines whose decode speed depends on whether their config drafts with DFlash2.

    python -m unittest tools.test_calibrate_draft
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_draft as CD  # noqa: E402
import maya  # noqa: E402


class FakeEngine:
    """Writes every answer at `tok_s` tokens/s; `answer_len`: answers end after that many tokens (else max_new)."""

    def __init__(self, tok_s, answer_len=None, drafts=(0, 0)):
        self.tok_s, self.answer_len, self.drafts = tok_s, answer_len, drafts
        self.last = {}
        self.requests = []

    def generate(self, ids, max_new, sampling, cancel):
        self.requests.append((tuple(ids), max_new, dict(sampling)))
        n = max_new if self.answer_len is None else min(max_new, self.answer_len)
        self.last = {"decode_ms": n / self.tok_s * 1000.0, "drafts_accepted": self.drafts[0],
                     "drafts_offered": self.drafts[1]}
        for i in range(n):
            yield i


def fake_requests(cfg):
    return [7, 7, 7], [[i, i] for i in range(len(CD.QUESTIONS))], 8000


BASE = {"args": ["--glm-pack", "pack", "--max-context", "32768"], "env": {"STRATA_GLM_USAGE": "0"},
        "sampling": {"temperature": 1.0, "top_p": 0.95}}


def calibrate(other_tok_s, dflash_tok_s, **kw):
    engines = {}

    def start(cfg):
        side = "dflash2" if "--dflash" in cfg["args"] else "other"
        engines[side] = FakeEngine(dflash_tok_s if side == "dflash2" else other_tok_s, **kw)
        engines[side + "_env"] = dict(cfg.get("env") or {})
        return engines[side]
    with patch.object(CD, "requests", fake_requests):
        res = CD.run(CD.without_dflash(BASE), CD.with_dflash(BASE, "drafter.gguf"), say=lambda *a: None,
                     start_engine=start)
    return res, engines


class Measure(unittest.TestCase):
    def test_dflash2_is_kept_only_when_clearly_faster(self):
        res, _ = calibrate(20.0, 25.0)
        self.assertEqual(res["keep"], "dflash2")
        self.assertAlmostEqual(res["gain"], 0.25)
        self.assertEqual(calibrate(20.0, 20.8)[0]["keep"], "other")     # +4%: within the noise, the MTP block stays
        self.assertEqual(calibrate(24.0, 12.0)[0]["keep"], "other")     # half the speed (#127)

    def test_the_warm_up_is_not_counted_and_the_speed_is_sustained(self):
        res, eng = calibrate(20.0, 30.0)
        for side in ("other", "dflash2"):
            first = eng[side].requests[0]
            self.assertEqual(first[0], (7, 7, 7))                       # the warm-up: the document, once
            self.assertEqual(first[1], CD.WARM_ANSWER)
            self.assertGreaterEqual(res[side]["tokens"], CD.MIN_TOKENS)
            self.assertEqual(res[side]["tokens"], sum(r[1] for r in eng[side].requests[1:]))
            self.assertGreaterEqual(res[side]["answers"], 2)             # several long answers, not one burst
        self.assertEqual(res["other"]["tok_s"], 20.0)

    def test_both_sides_answer_the_same_questions_with_the_same_seeds_sampled(self):
        _, eng = calibrate(20.0, 30.0)
        mine = lambda side: [(ids, n, s.get("seed"), s.get("temperature"), s.get("top_p"))
                             for ids, n, s in eng[side].requests]
        self.assertEqual(mine("other"), mine("dflash2"))
        seeds = [r[2] for r in mine("other")]
        self.assertEqual(len(set(seeds)), len(seeds))                   # a seed per answer
        self.assertTrue(all(r[3] == 1.0 and r[4] == 0.95 for r in mine("other")))   # as the dashboard samples
        self.assertEqual(len({r[0] for r in mine("other")[1:]}), len(mine("other")) - 1)   # each question fresh

    def test_short_answers_take_more_questions(self):
        res, eng = calibrate(20.0, 30.0, answer_len=200)
        self.assertEqual(res["other"]["answers"], 8)
        self.assertGreaterEqual(res["other"]["tokens"], CD.MIN_TOKENS)

    def test_too_few_tokens_are_no_measurement(self):
        with self.assertRaises(RuntimeError):
            calibrate(20.0, 30.0, answer_len=40)

    def test_each_side_keeps_this_pcs_usage_untouched(self):
        _, eng = calibrate(20.0, 30.0)
        for side in ("other", "dflash2"):
            self.assertEqual(eng[side + "_env"]["STRATA_GLM_USAGE"], "0")   # (usage_copy: "0" stays "0")
            self.assertEqual(eng[side + "_env"]["STRATA_GLM_SLOTS"], "0")   # no conversations kept aside

    def test_no_learned_profile_or_trace_is_written(self):
        base = {**BASE, "args": BASE["args"] + ["--expert-profile", "p.bin", "--expert-profile-save", "learned.bin",
                                                "--dump-routing", "trace.bin"]}
        seen = []

        def start(cfg):
            seen.append(list(cfg["args"]))
            return FakeEngine(20.0)
        with patch.object(CD, "requests", fake_requests):
            CD.run(CD.without_dflash(base), CD.with_dflash(base, "d.gguf"), say=lambda *a: None, start_engine=start)
        for args in seen:
            self.assertIn("--expert-profile", args)                     # the profile is read, as at a start
            self.assertNotIn("--expert-profile-save", args)
            self.assertNotIn("--dump-routing", args)

    def test_with_and_without_dflash(self):
        d = CD.with_dflash(BASE, "x.gguf")
        self.assertEqual(d["args"][-2:], ["--dflash", "x.gguf"])
        self.assertEqual(CD.without_dflash(d)["args"], BASE["args"])
        self.assertEqual(CD.with_dflash(d, "y.gguf")["args"].count("--dflash"), 1)
        self.assertNotIn("--dflash", BASE["args"])                       # (copies: the config itself is not changed)


def result(keep, o=20.0, d=25.0):
    return {"other": {"tok_s": o, "tokens": 1536}, "dflash2": {"tok_s": d, "tokens": 1536}, "keep": keep,
            "gain": d / o - 1.0}


class InTheSetup(unittest.TestCase):
    """maya.speed_calibration: the config it leaves behind."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.drafter = self.dir / "drafter.gguf"
        self.drafter.write_text("x")
        self.cfg_path = self.dir / "maya-test.json"
        self.write(gpu=[0, 1])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, gpu, extra=()):
        cfg = {"exe": "strata", "args": ["--glm-pack", "pack", "--max-context", "32768", "--dflash", str(self.drafter),
                                         *extra],
               "gpu": gpu, "installer": {"draft": "dflash2", "quant": "Maya-S"}}
        self.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    def run_it(self, res=None, raises=None, busy=()):
        calls = []

        def fake_run(other, dflash, say=None, other_label=None):
            calls.append((other, dflash, other_label))
            if raises:
                raise raises
            return res
        with patch.object(maya.CD, "run", side_effect=fake_run), patch.object(maya, "busy_gpus", return_value=list(busy)), \
                patch.object(maya, "saved_calibration", return_value=None), patch.object(maya, "say"), \
                patch.object(maya, "ok"), patch.object(maya, "warn"):
            out = maya.speed_calibration(self.cfg_path)
        return out, calls, json.loads(self.cfg_path.read_text(encoding="utf-8"))

    def test_slower_dflash2_switches_the_config_to_the_mtp_block(self):
        out, calls, cfg = self.run_it(result("other", o=23.9, d=12.0))
        self.assertTrue(out)
        self.assertNotIn("--dflash", cfg["args"])
        self.assertEqual(cfg["installer"]["draft"], "mtp")
        rec = cfg["installer"]["speed_calibration"]
        self.assertEqual((rec["kept"], rec["mtp_tok_s"], rec["dflash2_tok_s"]), ("mtp", 23.9, 12.0))
        self.assertEqual(calls[0][2], "the MTP block")
        self.assertNotIn("--dflash", calls[0][0]["args"])               # the other side really has no drafter
        self.assertIn("--dflash", calls[0][1]["args"])

    def test_faster_dflash2_is_kept(self):
        out, _, cfg = self.run_it(result("dflash2", o=42.6, d=61.4))
        self.assertTrue(out)
        self.assertEqual(cfg["args"].count("--dflash"), 1)
        self.assertEqual(cfg["installer"]["draft"], "dflash2")
        self.assertEqual(cfg["installer"]["speed_calibration"]["kept"], "dflash2")

    def test_one_gpu_compares_with_no_drafting(self):
        self.write(gpu=[0])
        _, calls, cfg = self.run_it(result("other", o=19.0, d=14.7))
        self.assertEqual(calls[0][2], "no drafting")
        self.assertEqual(cfg["installer"]["speed_calibration"]["no_drafting_tok_s"], 19.0)

    def test_a_failure_changes_nothing_and_is_not_retried_by_every_start(self):
        out, _, cfg = self.run_it(raises=RuntimeError("the engine exited before it was ready"))
        self.assertFalse(out)
        self.assertIn("--dflash", cfg["args"])                           # DFlash2 stays, as chosen
        self.assertIn("failed", cfg["installer"]["speed_calibration"])
        self.assertFalse(maya.needs_speed_calibration(cfg))

    def test_busy_gpus_postpone_it_to_the_next_start(self):
        out, calls, cfg = self.run_it(result("other"), busy=[(0, 20000)])
        self.assertFalse(out)
        self.assertEqual(calls, [])
        self.assertTrue(maya.needs_speed_calibration(cfg))

    def test_which_configs_need_it(self):
        cfg = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.assertTrue(maya.needs_speed_calibration(cfg))                # DFlash2, never measured (set up before the calibration)
        self.assertFalse(maya.needs_speed_calibration({**cfg, "installer": {"speed_calibration": {"kept": "dflash2"}}}))
        self.assertFalse(maya.needs_speed_calibration(maya.CD.without_dflash(cfg)))
        self.assertFalse(maya.needs_speed_calibration({**cfg, "args": cfg["args"] + ["--parallel", "2"]}))

    def test_a_start_measures_an_unmeasured_dflash2_config_once(self):
        for f in ("strata", "pack/index.txt", "tok/vocab.json"):
            (self.dir / f).parent.mkdir(parents=True, exist_ok=True)
            (self.dir / f).write_text("x")
        cfg = {"exe": str(self.dir / "strata"), "tokenizer": str(self.dir / "tok"), "gpu": [0, 1],
               "args": ["--glm-pack", str(self.dir / "pack"), "--dflash", str(self.drafter)]}
        self.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        a = SimpleNamespace(port=None, host=None, api_key=None, gpu=None, gpus=None, parallel=None)

        def measured(path, why=""):
            c = json.loads(path.read_text(encoding="utf-8"))
            c.setdefault("installer", {})["speed_calibration"] = {"kept": "dflash2"}
            path.write_text(json.dumps(c), encoding="utf-8")
            return True
        with patch.object(maya, "refresh_engine"), patch.object(maya, "say"), \
                patch.object(maya, "unattended_start", return_value=None), \
                patch.object(maya, "speed_calibration", side_effect=measured) as sc:   # (someone is at the terminal)
            maya.server_command(self.cfg_path, a)
            maya.server_command(self.cfg_path, a)
        sc.assert_called_once()

    def test_the_setup_text_says_faster_and_that_it_is_measured(self):
        args = SimpleNamespace(draft=None, yes=True)
        shown = []
        with patch.object(maya, "choose", side_effect=lambda q, intro, rows, d, y: shown.extend(rows) or 0):
            maya.choose_draft(args, {"backend": "cuda", "archs": [86]}, {})
        text = shown[1][2]
        self.assertIn("the same answers, faster", text)
        self.assertIn("speed calibration", text)
        self.assertNotIn("measure yours", text)


if __name__ == "__main__":
    unittest.main()
