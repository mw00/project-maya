"""Tests for tools/calibrate_glm.py without a GPU: a stand-in engine whose decode speed is a function of the CPU lane
settings a request names (`strata_tune`: pcie_frac, cpu_threads).

    python -m unittest tools.test_calibrate_glm
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_glm as CAL  # noqa: E402


class FakeEngine:
    """Decode tok/s = speed(share, threads); the engine started with share `own_share` and `threads` threads."""

    def __init__(self, speed, threads=40, own_share=0.0):
        self.speed = speed
        self.own = (own_share, threads)
        self.info = {"cpu_threads": threads, "pcie_share": f"{own_share:.2f}"}
        self.last = {}
        self.requests = []
        self.prompts = []
        self.closed = False

    def generate(self, ids, max_new, sampling, cancel):
        self.prompts.append(tuple(ids))
        tune = sampling.get("strata_tune") or {}
        share = tune.get("pcie_frac", self.own[0])
        threads = tune.get("cpu_threads", self.own[1])
        self.requests.append((share, threads))
        self.last = {"decode_ms": max_new / self.speed(share, threads) * 1000.0}
        for i in range(max_new):
            yield i


def run_measure(engine):
    started = []

    def start(cfg):
        started.append(cfg)
        return engine
    res = CAL.measure({"env": {}}, [[1, 2, 3]] * 3, start, say=lambda *a: None, extra_threads=())
    return res, started


class Measure(unittest.TestCase):
    def test_faster_share_and_threads_are_kept(self):
        # more PCIe share and fewer threads are faster here: 0.25 and 20 threads win by far more than MIN_GAIN
        eng = FakeEngine(lambda s, t: 15.0 + (5.0 if abs(s - 0.25) < 1e-9 else 0.0) + (3.0 if t == 20 else 0.0))
        res, started = run_measure(eng)
        self.assertEqual(len(started), 1)                       # one engine start: no restarts
        self.assertEqual(res["settings"], {CAL.SHARE_ENV: "0.25", CAL.THREADS_ENV: "20"})
        self.assertEqual(res["report"]["tok_s"], 23.0)

    def test_small_gain_keeps_the_engines_own(self):
        # 2% better is within the noise: nothing changes
        eng = FakeEngine(lambda s, t: 20.0 * (1.02 if t == 30 else 1.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertEqual(res["report"]["tok_s"], 20.0)

    def test_own_split_is_measured_without_a_share_key(self):
        eng = FakeEngine(lambda s, t: 18.0)
        run_measure(eng)
        self.assertIn((0.0, 40), eng.requests)                  # the engine's own: no keys at all
        shares = {r[0] for r in eng.requests}
        self.assertTrue(set(CAL.PCIE_SHARES) <= shares)

    def test_sweeps_stop_once_far_behind(self):
        # the PCIe share costs a lot from 0.1 on, fewer threads from 20 down: neither sweep goes past its first loser
        eng = FakeEngine(lambda s, t: 20.0 * (0.4 if s >= 0.1 else 1.0) * (0.5 if t <= 20 else 1.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertNotIn(0.25, {r[0] for r in eng.requests})
        self.assertNotIn(10, {r[1] for r in eng.requests})

    def test_threads_only(self):
        eng = FakeEngine(lambda s, t: 10.0 + (2.0 if t == 10 else 0.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {CAL.THREADS_ENV: "10"})

    def test_every_measurement_answers_new_prompts(self):
        # the tiers follow the text: no prompt is answered twice while the list lasts, a measurement's three in turn
        eng = FakeEngine(lambda s, t: 18.0)
        ids = [[i] for i in range(90)]
        CAL.measure({"env": {}}, ids, lambda c: eng, say=lambda *a: None, extra_threads=())
        self.assertLessEqual(len(eng.prompts), len(ids))
        self.assertEqual(len(set(eng.prompts)), len(eng.prompts))
        self.assertEqual(eng.prompts[:6], [(0,), (1,), (2,), (3,), (4,), (5,)])

    def test_no_cpu_lane(self):
        eng = FakeEngine(lambda s, t: 30.0, threads=0, own_share=1.0)
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertEqual(eng.requests, [])


class DiskEngine(FakeEngine):
    """A FakeEngine whose model does not fit VRAM and RAM (the decode reads from the SSD): its speed also depends on
    the request's read_chunks (8 when it names none)."""

    def __init__(self, speed, experts=12384, held=9000, **kw):
        super().__init__(lambda s, t: 0.0, **kw)
        self.speed3 = speed
        self.info.update(experts=experts, vram_slots=held // 2, ram_slots=held - held // 2)
        self.chunks = []

    def generate(self, ids, max_new, sampling, cancel):
        tune = sampling.get("strata_tune") or {}
        c = tune.get("read_chunks", 8)
        self.chunks.append(c)
        self.speed = lambda s, t: self.speed3(s, t, c)
        return super().generate(ids, max_new, sampling, cancel)


class ReadChunks(unittest.TestCase):
    def test_a_faster_read_size_is_kept(self):
        # a Windows laptop on an Intel RST RAID (#67): 4 pieces 15% faster than the engine's 8
        eng = DiskEngine(lambda s, t, c: 14.5 * (1.15 if c == 4 else 1.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {CAL.CHUNKS_ENV: "4"})
        self.assertAlmostEqual(res["report"]["tok_s"], 16.7, places=1)

    def test_fewer_pieces_slower_keeps_the_engines_own_and_stops(self):
        # a Linux NVMe (Mercury): 8 > 4 > 2 - nothing kept, and 2 is not tried once 4 lost
        eng = DiskEngine(lambda s, t, c: {8: 12.6, 4: 12.2, 2: 11.6}[c])
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertIn(4, eng.chunks)
        self.assertNotIn(2, eng.chunks)

    def test_no_disk_no_read_size(self):
        # every expert in VRAM or RAM: the disk reads' size is not measured
        eng = DiskEngine(lambda s, t, c: 20.0, experts=9000, held=9000)
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertEqual(set(eng.chunks), {8})

    def test_reads_disk(self):
        self.assertTrue(CAL.reads_disk({"experts": 12384, "vram_slots": 3000, "ram_slots": 5000}))
        self.assertFalse(CAL.reads_disk({"experts": 12384, "vram_slots": 7000, "ram_slots": 6000}))
        self.assertFalse(CAL.reads_disk({"cpu_threads": 40}))
        self.assertFalse(CAL.reads_disk({"experts": "x", "vram_slots": 1}))


class Helpers(unittest.TestCase):
    def test_apply_replaces_and_clears(self):
        env = {"STRATA_GLM_RAM_HEADROOM_GB": "4", CAL.SHARE_ENV: "0.50", CAL.THREADS_ENV: "12"}
        self.assertEqual(CAL.apply(env, {CAL.THREADS_ENV: "20"}),
                         {"STRATA_GLM_RAM_HEADROOM_GB": "4", CAL.THREADS_ENV: "20"})
        self.assertEqual(CAL.apply(env, {}), {"STRATA_GLM_RAM_HEADROOM_GB": "4"})

    def test_thread_candidates(self):
        self.assertEqual(CAL.thread_candidates(40, (43, 21)), [40, 30, 27, 20, 10, 21])
        self.assertEqual(CAL.thread_candidates(3), [3, 2])

    def test_pick(self):
        self.assertEqual(CAL.pick({"a": [10.0], "b": [10.2]}, "a"), "a")
        self.assertEqual(CAL.pick({"a": [10.0], "b": [10.5]}, "a"), "b")

    def test_prompts_are_groups_of_code_explanation_list(self):
        self.assertEqual(len(set(CAL.PROMPTS)), len(CAL.PROMPTS))
        self.assertEqual(len(CAL.PROMPTS) % CAL.GROUP, 0)
        for i in range(0, len(CAL.PROMPTS), CAL.GROUP):
            self.assertTrue(CAL.PROMPTS[i].startswith("Write a Python function"))
            self.assertTrue(CAL.PROMPTS[i + 1].startswith("Explain"))
            self.assertTrue(CAL.PROMPTS[i + 2].startswith("List"))

    def test_usage_copy(self):
        # the engine reads and writes a copy: this PC's usage file stays as it was
        with tempfile.TemporaryDirectory() as d:
            pack, tmp = Path(d) / "pack", Path(d) / "tmp"
            pack.mkdir()
            tmp.mkdir()
            (pack / "expert_usage.txt").write_text("3 7:12\n")
            env = CAL.usage_copy({"args": ["--glm-pack", str(pack)]}, tmp)
            self.assertEqual(env, {"STRATA_GLM_USAGE": str(tmp / "expert_usage.txt")})
            self.assertEqual((tmp / "expert_usage.txt").read_text(), "3 7:12\n")
            own = Path(d) / "mine.txt"
            own.write_text("5 1:2\n")
            env = CAL.usage_copy({"args": ["--glm-pack", str(pack)], "env": {"STRATA_GLM_USAGE": str(own)}}, tmp)
            self.assertEqual((tmp / "expert_usage.txt").read_text(), "5 1:2\n")
            self.assertEqual(CAL.usage_copy({"env": {"STRATA_GLM_USAGE": "0"}}, tmp), {"STRATA_GLM_USAGE": "0"})
            (tmp / "expert_usage.txt").unlink()
            env = CAL.usage_copy({"args": ["--glm-pack", str(Path(d) / "none")]}, tmp)   # no file yet: a new one
            self.assertEqual(env, {"STRATA_GLM_USAGE": str(tmp / "expert_usage.txt")})
            self.assertFalse((tmp / "expert_usage.txt").exists())

    def test_run_uses_a_usage_copy(self):
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "pack"
            pack.mkdir()
            (pack / "expert_usage.txt").write_text("3 7:12\n")
            seen = []

            def start(c):
                seen.append(dict(c["env"]))
                return FakeEngine(lambda s, t: 18.0, threads=0)
            cfg = {"args": ["--glm-pack", str(pack)], "env": {CAL.THREADS_ENV: "12"}}
            orig = CAL.prompt_ids
            CAL.prompt_ids = lambda c: [[1, 2, 3]] * 3
            try:
                CAL.run(cfg, say=lambda *a: None, start_engine=start)
            finally:
                CAL.prompt_ids = orig
            self.assertNotIn(CAL.THREADS_ENV, seen[0])                              # the engine's own choice
            self.assertNotEqual(Path(seen[0]["STRATA_GLM_USAGE"]).parent, pack)    # not the pack's own file
            self.assertEqual((pack / "expert_usage.txt").read_text(), "3 7:12\n")

    def test_cpu_list(self):
        self.assertEqual(CAL.cpu_list("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11])

    def test_host_thread_extras(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cpu"
            (root).mkdir()
            (root / "online").write_text("0-7\n")
            for c in range(8):
                t = root / f"cpu{c}" / "topology"
                t.mkdir(parents=True)
                t.joinpath("physical_package_id").write_text("0\n" if c < 4 else "1\n")
            hybrid = Path(d) / "cpu_core_cpus"
            self.assertEqual(CAL.host_thread_extras(root, hybrid), [3])        # two sockets of 4: 4 - 1
            hybrid.write_text("0-5\n")
            self.assertEqual(CAL.host_thread_extras(root, hybrid), [3, 5])     # + a hybrid CPU's 6 P-core CPUs - 1


class EngineError(unittest.TestCase):
    def test_the_engine_own_last_line(self):
        # the engine's own lines say "maya ..." (or "glm ...", "ERR ..."); without one, the log's last line
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("loading\nmaya generate: pack: out of memory\nsome library noise\n")
            self.assertEqual(CAL.engine_error(str(log)), "maya generate: pack: out of memory")
            log.write_text("only noise\n")
            self.assertEqual(CAL.engine_error(str(log)), "only noise")


if __name__ == "__main__":
    unittest.main()
