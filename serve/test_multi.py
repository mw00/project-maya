"""serve/test_multi.py - several conversations at once (READY ... multi=<n>): StrataEngine's per-request routing over a
scripted engine (serve/fake_multi_engine.py), and the service's FIFO with n places.

    python -m unittest serve.test_multi -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import FairLock, StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Engine(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["3"])

    def tearDown(self):
        self.engine.close()

    def test_ready_says_how_many(self):
        self.assertEqual(self.engine.multi, 3)

    def test_concurrent_requests_each_get_their_own_tokens(self):
        out = {}

        def one(k):
            ids = list(range(k * 100, k * 100 + 40))
            toks = [t for t in self.engine.generate(ids, 40, {}, threading.Event()) if t is not None]
            out[k] = (ids, toks)

        ths = [threading.Thread(target=one, args=(k,)) for k in range(3)]
        [t.start() for t in ths]
        [t.join(30) for t in ths]
        for k, (ids, toks) in out.items():
            self.assertEqual(toks, [i + 1000 for i in ids], k)
        self.assertEqual(len(out), 3)

    def test_each_request_reads_its_own_done_figures(self):   # (#310: one shared `last` mixed them up)
        out = {}

        def one(k):
            ids = list(range(k * 100, k * 100 + 20 + 7 * k))
            for _ in self.engine.generate(ids, 20, {}, threading.Event()):
                pass
            time.sleep(0.2)                            # (the others' DONE lines land meanwhile)
            out[k] = (len(ids), self.engine.last.get("prompt_tokens"))

        ths = [threading.Thread(target=one, args=(k,)) for k in range(3)]
        [t.start() for t in ths]
        [t.join(30) for t in ths]
        self.assertEqual(len(out), 3)
        for k, (n, got) in out.items():
            self.assertEqual(got, n, k)

    def test_a_consumer_that_stops_ends_only_its_request(self):
        out = {}

        def long_one():
            ids = list(range(500, 600))
            out["long"] = [t for t in self.engine.generate(ids, 100, {}, threading.Event()) if t is not None]

        th = threading.Thread(target=long_one)
        th.start()
        gen = self.engine.generate(list(range(10)), 50, {}, threading.Event())
        got = []
        for t in gen:
            if t is not None:
                got.append(t)
            if len(got) == 5:
                break
        gen.close()                                   # STOP <rid>, drained to its DONE
        self.assertEqual(self.engine.last.get("finish"), "cancel")
        th.join(30)
        self.assertEqual(out["long"], [i + 1000 for i in range(500, 600)])
        self.assertEqual(got, [1000, 1001, 1002, 1003, 1004])
        self.assertEqual(self.engine._routes, {})

    def test_an_engine_error_reaches_its_request(self):
        # a fourth request on a 3-state engine is refused for that request alone
        evs = [threading.Event() for _ in range(3)]
        gens = [self.engine.generate(list(range(200)), 200, {}, evs[i]) for i in range(3)]
        for g in gens:
            next(g)                                   # their GEN lines are out
        with self.assertRaises(ValueError) as cm:
            list(self.engine.generate([1, 2], 2, {}, threading.Event()))
        self.assertIn("busy", str(cm.exception))
        for g in gens:
            g.close()


class Interleaved(unittest.TestCase):
    """STRATA_GLM_PREFILL_INTERLEAVE's lines: a new prompt's PP lines come between the running requests' T lines."""

    def setUp(self):
        import os
        os.environ["FAKE_MULTI_INTERLEAVE"] = "20"
        try:
            self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["3"])
        finally:
            del os.environ["FAKE_MULTI_INTERLEAVE"]

    def tearDown(self):
        self.engine.close()

    def test_running_requests_decode_while_a_prompt_is_read(self):
        times, out = [], {}

        def long_one():
            ids = list(range(500, 700))
            toks = []
            for t in self.engine.generate(ids, 200, {}, threading.Event()):
                if t is not None:
                    toks.append(t)
                    times.append(time.monotonic())
            out["long"] = toks

        th = threading.Thread(target=long_one)
        th.start()
        while len(times) < 3:
            time.sleep(0.001)
        sent = time.monotonic()
        first = None
        got = []
        for t in self.engine.generate(list(range(300)), 5, {}, threading.Event()):   # 15 pieces of 20
            if t is not None:
                first = first or time.monotonic()
                got.append(t)
        th.join(30)
        self.assertEqual(got, [1000, 1001, 1002, 1003, 1004])
        self.assertEqual(out["long"], [i + 1000 for i in range(500, 700)])
        during = [x for x in times if sent < x < first]
        self.assertGreaterEqual(len(during), 10)      # a token of the running one between the pieces
        self.assertEqual(self.engine._routes, {})

    def test_a_stop_during_its_read_ends_that_request_alone(self):
        out = {}

        def long_one():
            out["long"] = [t for t in self.engine.generate(list(range(500, 600)), 100, {}, threading.Event())
                           if t is not None]

        th = threading.Thread(target=long_one)
        th.start()
        time.sleep(0.05)
        gen = self.engine.generate(list(range(2000)), 5, {}, threading.Event())     # 100 pieces
        for _ in range(3):
            next(gen)                                 # its GEN line is out, its read has begun
        gen.close()                                   # STOP <rid> mid-read: drained to its DONE
        self.assertEqual(self.engine.last.get("finish"), "cancel")
        self.assertEqual(self.engine.last.get("generated", 0), 0)
        th.join(30)
        self.assertEqual(out["long"], [i + 1000 for i in range(500, 600)])
        self.assertEqual(self.engine._routes, {})

    def test_prompts_arriving_during_a_read_queue_behind_it(self):
        out = {}

        def one(k, n_ids):
            ids = list(range(k * 1000, k * 1000 + n_ids))
            out[k] = [t for t in self.engine.generate(ids, 4, {}, threading.Event()) if t is not None]

        ths = [threading.Thread(target=one, args=(k, 200 if k == 0 else 40)) for k in range(3)]
        for t in ths:
            t.start()
            time.sleep(0.005)
        [t.join(30) for t in ths]
        for k in range(3):
            self.assertEqual(out[k], [k * 1000 + 1000 + i for i in range(4)], k)


class Places(unittest.TestCase):
    def test_n_requests_hold_n_places_and_a_whole_waits_for_all(self):
        lock = FairLock(3)
        for _ in range(3):
            self.assertTrue(lock.acquire(units=1, timeout=1))
        self.assertFalse(lock.acquire(units=1, blocking=False))
        got = []
        th = threading.Thread(target=lambda: (lock.acquire(), got.append("all"), lock.release()))
        th.start()
        while lock.waiting() < 1:
            time.sleep(0.001)
        lock.release(units=1)
        time.sleep(0.05)
        self.assertEqual(got, [])                     # the exclusive waiter needs every place
        lock.release(units=1)
        lock.release(units=1)
        th.join(5)
        self.assertEqual(got, ["all"])
        self.assertFalse(lock.locked())

    def test_a_waiter_for_all_holds_back_later_ones(self):
        lock = FairLock(2)
        lock.acquire(units=1)
        order = []
        a = threading.Thread(target=lambda: (lock.acquire(), order.append("all"), lock.release()))
        a.start()
        while lock.waiting() < 1:
            time.sleep(0.001)
        b = threading.Thread(target=lambda: (lock.acquire(units=1), order.append("one"), lock.release(units=1)))
        b.start()
        while lock.waiting() < 2:
            time.sleep(0.001)
        time.sleep(0.05)
        self.assertEqual(order, [])                   # a free place, but the first in line wants both
        lock.release(units=1)
        a.join(5)
        b.join(5)
        self.assertEqual(order, ["all", "one"])


if __name__ == "__main__":
    unittest.main()
