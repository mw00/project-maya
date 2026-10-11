"""serve/test_fair.py - the FIFO's order and the fair slices (STRATA_FAIR_SLICE_S), against a scripted engine that
continues an answer from wherever the prompt says it got to (as the real engine does with its conversation slots).

    python -m unittest serve.test_fair -v
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, FairLock, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ResumingEngine:
    """Each conversation's answer is the script whose key its first prompt contains; a later prompt that extends that
    first prompt (prompt + what was already written) gets the rest of the same answer."""

    def __init__(self, tok, answers: dict[str, str], delay_s: float = 0.002):
        self.tok, self.answers, self.delay, self.max_context = tok, answers, delay_s, 32768
        self.end = tok.encode(IM_END, parse_special=True)
        self.bases: list[list[int]] = []
        self.calls: list[tuple[list[int], int]] = []
        self.lock = threading.Lock()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.lock:
            self.calls.append((list(ids), max_new))
            base = next((b for b in self.bases if ids[:len(b)] == b), None)
            if base is None:
                base = list(ids)
                self.bases.append(base)
        text = self.tok.decode(base)
        answer = next(v for k, v in self.answers.items() if k in text)
        script = self.tok.encode(answer) + self.end
        for t in script[len(ids) - len(base):][:max_new]:
            if cancel.is_set():
                return
            time.sleep(self.delay)
            yield t


class PausingEngine(ResumingEngine):
    """As the real engine: PAUSE ends the request a few tokens later (the ones already on their way still come),
    then its DONE says it was cut ("cancel")."""
    can_pause = True
    LATE = 3

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.paused, self.pauses, self.last = False, 0, {}

    def pause(self, keep=False):
        self.paused = True
        self.pauses += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.paused, late, finish = False, None, "stop"
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            if self.paused:
                late = self.LATE if late is None else late - 1
                if late == 0:
                    finish = "cancel"
                    break
            yield t
        self.last = {"finish": finish, "generated": 0, "prompt_ms": 0.0, "decode_ms": 0.0}


def collect(svc, user: str, max_new: int, out: dict, cancel=None, stop_after=None):
    ids, thinking, max_new = svc.prepare([{"role": "user", "content": user}], None,
                                         {"enable_thinking": False}, max_new)
    text, n = "", 0
    gen = svc.run(ids, thinking, None, max_new, {}, cancel or threading.Event())
    for kind, val in gen:
        if kind == "event" and val.kind == "content":
            text += val.text
            n += len(val.text)
            if stop_after and n >= stop_after:
                gen.close()                          # the client goes away mid-answer
                break
    out[user] = (text, time.monotonic())


class FairLockOrder(unittest.TestCase):
    def test_waiters_get_it_in_arrival_order(self):
        lock, order = FairLock(), []
        lock.acquire()
        threads = []
        for i in range(6):
            th = threading.Thread(target=lambda i=i: (lock.acquire(), order.append(i), lock.release()))
            th.start()
            threads.append(th)
            while lock.waiting() < i + 1:            # the next one asks only once this one is in line
                time.sleep(0.001)
        lock.release()
        for th in threads:
            th.join(5)
        self.assertEqual(order, list(range(6)))
        self.assertFalse(lock.locked())

    def test_a_timeout_gives_up_its_place(self):
        lock = FairLock()
        lock.acquire()
        self.assertFalse(lock.acquire(timeout=0.05))
        self.assertEqual(lock.waiting(), 0)
        lock.release()
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()

    def test_a_place_survives_its_wakeups(self):
        lock = FairLock()
        lock.acquire()
        mine = lock.enqueue()
        self.assertFalse(lock.wait_turn(mine, timeout=0.02))
        other = threading.Thread(target=lambda: (lock.acquire(), lock.release()))
        other.start()
        while lock.waiting() < 2:
            time.sleep(0.001)
        lock.release()
        self.assertTrue(lock.wait_turn(mine, timeout=5))   # still first in line, before the later thread
        self.assertEqual(lock.waiting(), 1)
        lock.release()
        other.join(5)


class FairSlices(unittest.TestCase):
    LONG = "L" * 600
    SHORT = "s" * 20

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ResumingEngine(self.tok, {"long": self.LONG, "short": self.SHORT})
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.env = os.environ.get("STRATA_FAIR_SLICE_S")

    def tearDown(self):
        if self.env is None:
            os.environ.pop("STRATA_FAIR_SLICE_S", None)
        else:
            os.environ["STRATA_FAIR_SLICE_S"] = self.env

    def race(self, **kw):
        out = {}
        a = threading.Thread(target=collect, args=(self.svc, "long one", 4096, out), kwargs=kw)
        a.start()
        while not self.svc.status.get("first_token"):
            time.sleep(0.001)
        b = threading.Thread(target=collect, args=(self.svc, "short one", 4096, out))
        b.start()
        a.join(30)
        b.join(30)
        return out

    def test_off_by_default_the_short_one_waits(self):
        os.environ.pop("STRATA_FAIR_SLICE_S", None)
        out = self.race()
        self.assertEqual(out["long one"][0], self.LONG)
        self.assertEqual(out["short one"][0], self.SHORT)
        self.assertLess(out["long one"][1], out["short one"][1])
        self.assertEqual(len(self.engine.calls), 2)

    def test_the_short_one_goes_first_and_the_long_one_continues(self):
        os.environ["STRATA_FAIR_SLICE_S"] = "0.05"
        out = self.race()
        self.assertEqual(out["short one"][0], self.SHORT)
        self.assertEqual(out["long one"][0], self.LONG)          # nothing lost or repeated at the seam
        self.assertLess(out["short one"][1], out["long one"][1])
        long_calls = [c for c in self.engine.calls if c[0][:len(self.engine.bases[0])] == self.engine.bases[0]]
        self.assertGreaterEqual(len(long_calls), 2)
        first, again = long_calls[0], long_calls[1]
        written = len(again[0]) - len(first[0])
        self.assertGreater(written, 0)
        self.assertEqual(again[1], first[1] - written)          # the budget left, not the whole one again
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.status["busy"])

    def test_pause_keeps_the_tokens_still_on_their_way(self):
        os.environ["STRATA_FAIR_SLICE_S"] = "0.05"
        self.engine = PausingEngine(self.tok, {"long": self.LONG, "short": self.SHORT})
        self.svc.engine = self.engine
        out = self.race()
        self.assertEqual(out["short one"][0], self.SHORT)
        self.assertEqual(out["long one"][0], self.LONG)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertGreaterEqual(self.engine.pauses, 1)
        self.assertFalse(self.svc.fifo.locked())

    def test_alone_it_is_never_cut(self):
        os.environ["STRATA_FAIR_SLICE_S"] = "0.01"
        out = {}
        collect(self.svc, "long one", 4096, out)
        self.assertEqual(out["long one"][0], self.LONG)
        self.assertEqual(len(self.engine.calls), 1)

    def test_a_client_gone_while_queued_again_leaves_the_line(self):
        os.environ["STRATA_FAIR_SLICE_S"] = "0.02"
        self.engine.answers["short"] = "s" * 400               # long enough to hold the engine for a while
        out = {}
        a = threading.Thread(target=collect, args=(self.svc, "long one", 4096, out),
                             kwargs={"stop_after": 50})
        a.start()
        while not self.svc.status.get("first_token"):
            time.sleep(0.001)
        b = threading.Thread(target=collect, args=(self.svc, "short one", 4096, out))
        b.start()
        a.join(30)
        b.join(30)
        self.assertEqual(out["short one"][0], "s" * 400)
        self.assertLess(len(out["long one"][0]), len(self.LONG))
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.svc.status["queued"], 0)


if __name__ == "__main__":
    unittest.main()
