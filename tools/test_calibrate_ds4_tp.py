"""Tests for tools/calibrate_ds4_tp.py without a GPU: stand-in engines whose decode and prompt speeds are functions of the
--tp and --cpu-threads they were started with (each setting an engine start of its own, as strata-ds4's are).

    python -m unittest tools.test_calibrate_ds4_tp
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_ds4_tp as C4  # noqa: E402

C4.usable_cpus = lambda: 88   # (a host with more CPUs than the 44 threads below: the lane leaves some free)

NINE = [24.0] + [16.0] * 8               # a 24 GB GPU and eight 16 GB ones, in the engine's order


class FakeEngine:
    """Started with cfg["args"]: decode tok/s = speed(tp, threads), a long prompt's read tok/s = read(tp)."""

    def __init__(self, cfg, speed, read, cpu_experts=100, threads=44):
        a = cfg["args"]
        self.tp = int(C4.arg(a, "--tp") or 2)   # (the engine's default)
        w = C4.arg(a, "--cpu-threads")
        self.threads = int(w) if w else threads
        self.args = list(a)
        self.speed, self.read = speed, read
        self.info = {"cpu_threads": self.threads, "cpu_experts": cpu_experts}
        self.last = {}

    def generate(self, ids, max_new, sampling, cancel):
        if max_new == 1:                              # (the long prompt: its reading)
            self.last = {"prompt_tokens": len(ids), "prompt_ms": len(ids) / self.read(self.tp) * 1000.0, "reused": 0,
                         "decode_ms": 0.0}
        else:
            self.last = {"prompt_tokens": len(ids), "prompt_ms": 1.0, "reused": 0,
                         "decode_ms": max_new / self.speed(self.tp, self.threads) * 1000.0}
        for i in range(max_new):
            yield i


class Run:
    """measure() over FakeEngines: the starts it made, and its result."""

    def __init__(self, speed, read=lambda tp: 900.0, vram=NINE, args=None, fail=lambda tp, n: False, **kw):
        self.starts = []

        def start(cfg):
            tp = int(C4.arg(cfg["args"], "--tp") or 2)
            if fail(tp, sum(s[0] == tp for s in self.starts)):    # (n: that --tp's starts so far)
                raise RuntimeError("the engine exited before it was ready")
            e = FakeEngine(cfg, speed, read, **kw)
            self.starts.append((e.tp, C4.arg(cfg["args"], "--cpu-threads")))
            return e
        cfg = {"args": args or ["--pack", "/m/maya-ds4", "--max-context", "65536", "--tp", "2"]}
        self.res = C4.measure(cfg, vram, [[1, 2, 3]] * 12, [0] * 16000, start, say=lambda *a: None)


# decode as measured on 9 GPUs at 2.9 bits (36c7c091), and the long prompt's reading
DECODE = {1: 80.0, 2: 103.0, 4: 139.0, 8: 151.0, 9: 120.0}
READ = {1: 950.0, 2: 908.0, 4: 760.0, 8: 607.0, 9: 500.0}


class Candidates(unittest.TestCase):
    def test_tp(self):
        self.assertEqual(C4.tp_candidates(NINE), [2, 4, 8, 9])
        self.assertEqual(C4.tp_candidates([16.0, 16.0]), [2])
        self.assertEqual(C4.tp_candidates([24.0]), [])
        self.assertEqual(C4.tp_candidates([24.0, 16.0]), [1, 2])   # (no two alike: each alone, or one group)
        self.assertEqual(C4.tp_candidates([16.0] * 3), [2, 3])
        self.assertEqual(C4.tp_candidates([16.0] * 6), [2, 4, 6])
        self.assertEqual(C4.tp_candidates([24.0, 24.0] + [16.0] * 4), [2, 4, 6])

    def test_threads(self):
        self.assertEqual(C4.thread_candidates(44), [33, 22])
        self.assertEqual(C4.thread_candidates(3), [2])
        self.assertEqual(C4.thread_candidates(2), [])
        # a lane that takes every CPU (#433): 2 and 4 fewer first
        self.assertEqual(C4.thread_candidates(32, every_cpu=True), [30, 28, 24, 16])
        self.assertEqual(C4.thread_candidates(4, every_cpu=True), [3, 2])

    def test_apply(self):
        a = ["--pack", "/p", "--tp", "2"]
        self.assertEqual(C4.apply(a, {"--tp": 8, "--cpu-threads": None}), ["--pack", "/p", "--tp", "8"])
        self.assertEqual(C4.apply(a, {"--tp": 8, "--cpu-threads": 33}), ["--pack", "/p", "--tp", "8", "--cpu-threads",
                                                                          "33"])
        b = ["--pack", "/p", "--tp", "4", "--cpu-threads", "20"]
        self.assertEqual(C4.apply(b, {"--tp": 8, "--cpu-threads": None}), ["--pack", "/p", "--tp", "8"])
        self.assertEqual(C4.apply(b, {"--tp": 8, "--cpu-threads": None}, keep=("--tp",)),
                         ["--pack", "/p", "--tp", "4"])
        self.assertEqual(a, ["--pack", "/p", "--tp", "2"])   # (a copy)


class Measure(unittest.TestCase):
    def test_fastest_decode_whose_prompts_still_read(self):
        r = Run(lambda tp, w: DECODE[tp], lambda tp: READ[tp])
        self.assertEqual(r.res["args"], {"--tp": 8, "--cpu-threads": None})
        self.assertEqual([s[0] for s in r.starts[:4]], [2, 4, 8, 9])   # (the config's own first)
        self.assertEqual(r.starts[4:], [(8, "33")])   # (33 threads no faster: a half is not tried)
        rep = r.res["report"]
        self.assertEqual((rep["tok_s"], rep["own_tok_s"], rep["prompt_tok_s"]), (151.0, 103.0, 607))

    def test_too_slow_a_prompt_is_not_taken(self):
        read = {**READ, 8: 500.0}                      # (below 60% of --tp 2's 908)
        r = Run(lambda tp, w: DECODE[tp], lambda tp: read[tp])
        self.assertEqual(r.res["args"]["--tp"], 4)

    def test_a_small_gain_is_not_kept(self):
        r = Run(lambda tp, w: 103.0 * (1.03 if tp == 4 else 1.0))
        self.assertEqual(r.res["args"]["--tp"], 2)
        self.assertEqual(r.res["report"]["confirm"], {})

    def test_a_gain_under_confirm_below_is_started_again(self):
        r = Run(lambda tp, w: 103.0 * (1.10 if tp == 4 else 1.0))
        self.assertEqual(r.res["args"]["--tp"], 4)
        self.assertEqual([s[0] for s in r.starts[:6]], [2, 4, 8, 9, 2, 4])
        self.assertEqual(set(r.res["report"]["confirm"]), {"2", "4"})

    def test_fewer_threads_kept_while_they_help(self):
        r = Run(lambda tp, w: DECODE[tp] * (1.10 if w == 33 else 1.0))
        self.assertEqual(r.res["args"], {"--tp": 8, "--cpu-threads": 33})
        self.assertEqual(r.starts[4:], [(8, "33"), (8, "22")])

    def test_no_cpu_lane_reported_threads_still_tested(self):
        # (#433: a reported 0 skipped the test on a PC whose lane took ~60% of the experts - now one start says)
        r = Run(lambda tp, w: DECODE[tp], cpu_experts=0)
        self.assertEqual(r.starts[4:], [(8, "33")])
        self.assertEqual(r.res["args"]["--cpu-threads"], None)

    def test_a_lane_on_every_cpu_tries_two_fewer_first(self):
        saved = C4.usable_cpus
        C4.usable_cpus = lambda: 44
        try:
            r = Run(lambda tp, w: DECODE[tp] * (1.9 if w == 42 else 1.0))
        finally:
            C4.usable_cpus = saved
        self.assertEqual(r.res["args"], {"--tp": 8, "--cpu-threads": 42})
        self.assertEqual(r.starts[4:6], [(8, "42"), (8, "40")])

    def test_one_gpu_only_threads(self):
        r = Run(lambda tp, w: 5.0 * (1.2 if w == 33 else 1.0), vram=[24.0], args=["--pack", "/p", "--tp", "2"])
        self.assertEqual([s[0] for s in r.starts], [2, 2, 2])
        self.assertEqual(r.res["args"], {"--tp": 2, "--cpu-threads": 33})

    def test_a_setting_that_does_not_start_is_left_out(self):
        r = Run(lambda tp, w: {2: 103.0, 4: 139.0, 7: 200.0}[tp], vram=[24.0] * 7, fail=lambda tp, n: tp == 7)
        self.assertEqual([s[0] for s in r.starts[:2]], [2, 4])
        self.assertEqual(r.res["args"]["--tp"], 4)
        self.assertEqual(set(r.res["report"]["tp_sweep"]), {"2", "4"})
        self.assertIn("--tp 7", r.res["report"]["failed"])

    def test_a_config_without_tp_is_the_engines_default(self):
        # (no --tp: the engine's 2 is the baseline - not a --tp 1 start, which reads prompts in windows of 16)
        r = Run(lambda tp, w: DECODE[tp], cpu_experts=0, args=["--pack", "/p"])
        self.assertEqual([s[0] for s in r.starts[:4]], [2, 4, 8, 9])
        self.assertEqual(r.res["report"]["own"]["--tp"], 2)

    def test_the_baseline_must_start(self):
        with self.assertRaises(RuntimeError):
            Run(lambda tp, w: DECODE[tp], fail=lambda tp, n: tp == 2)

    def test_a_winner_that_does_not_start_again_is_not_kept(self):
        r = Run(lambda tp, w: 103.0 * (1.10 if tp == 4 else 1.0), fail=lambda tp, n: tp == 4 and n == 1)
        self.assertEqual(r.res["args"]["--tp"], 2)
        self.assertIn("--tp 4 again", r.res["report"]["failed"])

    def test_threads_that_do_not_start_stop_the_sweep(self):
        r = Run(lambda tp, w: DECODE[tp] * (1.10 if w == 33 else 1.0), fail=lambda tp, n: tp == 8 and n == 1)
        self.assertEqual(r.res["args"], {"--tp": 8, "--cpu-threads": None})
        self.assertIn("33 CPU threads", r.res["report"]["failed"])

    def test_the_configs_own_threads_are_the_baseline(self):
        r = Run(lambda tp, w: DECODE[tp], args=["--pack", "/p", "--tp", "2", "--cpu-threads", "40"])
        self.assertTrue(all(s[1] == "40" for s in r.starts[:4]))
        self.assertEqual(r.starts[4:], [(8, "30")])
        self.assertEqual(r.res["args"], {"--tp": 8, "--cpu-threads": 40})


class Profile(unittest.TestCase):
    def test_every_start_a_fresh_copy(self):
        with tempfile.TemporaryDirectory() as d:
            pack, tmp = Path(d) / "maya-ds4", Path(d) / "tmp"
            pack.mkdir()
            tmp.mkdir()
            (pack / "expert-profile.bin").write_bytes(b"own")
            fresh = C4.profile_copy(["--pack", str(pack), "--tp", "2"], tmp)
            a = fresh(["--pack", str(pack), "--tp", "4"])
            self.assertEqual(a, ["--pack", str(pack), "--tp", "4", "--expert-profile", str(tmp / "expert-profile.bin")])
            (tmp / "expert-profile.bin").write_bytes(b"the tuning's answers")   # (the engine saved its own)
            fresh(a)
            self.assertEqual((tmp / "expert-profile.bin").read_bytes(), b"own")
            self.assertEqual((pack / "expert-profile.bin").read_bytes(), b"own")

    def test_a_named_profile_and_none(self):
        with tempfile.TemporaryDirectory() as d:
            own, tmp = Path(d) / "prof.bin", Path(d)
            fresh = C4.profile_copy(["--pack", "/p", "--expert-profile", str(own)], tmp / "t")
            (tmp / "t").mkdir()
            a = fresh(["--pack", "/p", "--expert-profile", str(own), "--tp", "8"])
            self.assertEqual(a, ["--pack", "/p", "--tp", "8", "--expert-profile", str(tmp / "t" / "expert-profile.bin")])
            self.assertFalse((tmp / "t" / "expert-profile.bin").exists())   # (none yet: the engine starts without)
            same = ["--pack", "/p", "--expert-profile", "none"]
            self.assertEqual(C4.profile_copy(same, tmp)(same), same)


if __name__ == "__main__":
    unittest.main()
