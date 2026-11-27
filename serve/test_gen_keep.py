"""MAYA_GEN_KEEP (server) / MAYA_DS4_GEN_KEEP (maya-ds4): a prompt's GEN line carries only the ids past what the last
prompt already was (`keep=<n> h=<hash>`), and the engine reads the very ids the whole line would have carried.

keep_ids_memo: n is the ids shared with the last prompt, h the BLAKE2b-128 of this prompt's first n (whatever the memo
held), the text the rest - over random histories (appends, edits, cuts, other conversations, the same prompt again).
StrataEngine.generate with an engine that stands in for maya-ds4's side (its state's tokens, the hash of its OWN first
n, RESEND when it does not hold them - src/ds4/ds4_keep.hpp, tests/ds4_keep_test.cpp): the prompt the engine ends up
with is the prompt sent, through appends, an engine started again, another conversation, a state cut short, a state
that went back, the same prompt twice; an engine that does not say INFO gen_keep=1 never gets keep=, nor does any
engine with the switch off, nor a request with pictures.

    python -m unittest serve.test_gen_keep
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import random
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as SV  # noqa: E402
from serve.test_multi import script_exe  # noqa: E402

FAKE = r'''#!/usr/bin/env python3
# maya-ds4's side of the GEN line, tokens only: argv[1] = the log (a JSON line a request), argv[2] = "keep" (says INFO
# gen_keep=1 and takes keep=) or "old" (knows no keep=: an unknown key is skipped, as maya-ds4's parse_gen does)
import array, hashlib, json, sys
log, mode = sys.argv[-2], sys.argv[-1]
state = []
if mode == "keep":
    print("INFO gen_keep=1")
print("READY 1000000 stop", flush=True)
for line in sys.stdin:
    f = line.split()
    if not f or f[0] == "STOP":
        continue
    if f[0] == "QUIT":
        break
    keys = dict(x.split("=", 1) for x in f[2:-1] if "=" in x)
    ids = [int(x) for x in f[-1].split(",")]
    sent = len(ids)
    if "keep" in keys:
        n = int(keys["keep"])
        own = hashlib.blake2b(array.array("i", state[:n]).tobytes(), digest_size=16).hexdigest()
        if mode != "keep" or n <= 0 or n > len(state) or own != keys.get("h"):
            if mode == "keep":
                open(log, "a").write(json.dumps({"resend": n}) + "\n")
                print("RESEND", flush=True)
                continue
        else:
            ids = state[:n] + ids
    open(log, "a").write(json.dumps({"keep": "keep" in keys, "sent": sent, "ids": ids}) + "\n")
    max_new = int(f[1])
    if max_new == 7:                # a read stopped half way: the state holds half the prompt
        state = ids[:len(ids) // 2]
        print(f"DONE 0 {len(ids)} 0.0 0.0 stop 0 0 0 0 0", flush=True)
        continue
    if max_new == 8:                # the prompt refused (too long, ...): the state as it was
        print("ERR refused")
        print(f"DONE 0 {len(ids)} 0.0 0.0 cancel 0 0 0 0 0", flush=True)
        continue
    out = [900000 + i for i in range(max_new)]
    state = ids + out
    for t in out:
        print(f"T {t}")
    print(f"DONE {len(out)} {len(ids)} 0.0 0.0 stop 0 0 0 0 0", flush=True)
'''


def digest(ids) -> str:
    return hashlib.blake2b(array.array("i", ids).tobytes(), digest_size=16).hexdigest()


class Memo(unittest.TestCase):
    def test_random_histories(self):
        rng = random.Random(1121)
        for block in (4, 64, 1024):
            for _ in range(40):
                memo, last = {}, []
                conv = [rng.randrange(200000) for _ in range(rng.randrange(1, 3000))]
                for _ in range(40):
                    op = rng.randrange(10)
                    if op < 5:
                        conv = conv + [rng.randrange(200000) for _ in range(rng.randrange(0, 400))]
                    elif op == 5:
                        conv = conv[:rng.randrange(1, len(conv) + 1)]
                    elif op == 6:
                        at = rng.randrange(len(conv))
                        conv = conv[:at] + [conv[at] ^ 1] + conv[at + 1:]
                    elif op == 7:
                        conv = [rng.randrange(200000) for _ in range(rng.randrange(1, 3000))]
                    elif op == 8:
                        conv = conv[:-1] if len(conv) > 1 else conv     # (the last prompt but its last id)
                    ids = list(conv)
                    n = 0
                    while n < min(len(ids) - 1, len(last)) and ids[n] == last[n]:
                        n += 1
                    got = SV.keep_ids_memo(memo, ids, block)
                    if n == 0:
                        self.assertIsNone(got)
                    else:
                        self.assertEqual(got, (n, digest(ids[:n]), ",".join(map(str, ids[n:]))))
                        self.assertTrue(got[2])
                    last = ids
                    self.assertEqual(memo["ids"], ids)
                    self.assertEqual(len(memo["cks"]), len(ids) // block)
                    cks = memo["cks"]                                   # (the kept running hashes are their prefixes':
                    for k in {0, len(cks) - 1, *(rng.randrange(len(cks)) for _ in range(4))} if cks else ():   # a few)
                        self.assertEqual(cks[k].hexdigest(), digest(ids[:(k + 1) * block]))

    def test_the_memo_is_its_own_copy(self):
        memo, ids = {}, list(range(5000))
        SV.keep_ids_memo(memo, ids)
        ids[10] = -1                                                    # the caller changes its list afterwards
        n, h, text = SV.keep_ids_memo(memo, list(range(5000)) + [7, 8])
        self.assertEqual((n, h, text), (5000, digest(range(5000)), "7,8"))
        self.assertIsNot(memo["ids"], ids)

    def test_not_int32(self):
        with self.assertRaises((TypeError, ValueError, OverflowError)):
            SV.keep_ids_memo(memo := {"ids": [1], "cks": []}, [1, 2 ** 40, 3] + [0] * 2000)
        self.assertEqual(memo, {})                                      # (and nothing half done is kept)


class Engine(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="maya-gen-keep-")
        self.addCleanup(self.dir.cleanup)
        self.fake = Path(self.dir.name) / "fake_keep_engine.py"
        self.fake.write_text(FAKE, encoding="utf-8")
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IXUSR)
        self.log = Path(self.dir.name) / "log.jsonl"
        self.was = SV.GEN_KEEP
        SV.GEN_KEEP = True
        self.addCleanup(lambda: setattr(SV, "GEN_KEEP", self.was))
        self.wait = SV.wait_gpu_release                 # (restart() asks NVML about the old process: no GPU here)
        SV.wait_gpu_release = lambda pid, timeout=60.0: None
        self.addCleanup(lambda: setattr(SV, "wait_gpu_release", self.wait))
        self.engine = None

    def start(self, mode="keep"):
        self.engine = SV.StrataEngine(script_exe(self.fake), [str(self.log), mode])
        self.addCleanup(self.engine.close)
        return self.engine

    def seen(self):
        rows = [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []
        self.log.write_text("")
        return rows

    def ask(self, ids, max_new=3, embeddings=None):
        out = [t for t in self.engine.generate(list(ids), max_new, {}, threading.Event(), embeddings) if t is not None]
        return [int(t) for t in out]

    def test_appends_send_only_what_is_new(self):
        self.start()
        rng = random.Random(7)
        conv = [rng.randrange(200000) for _ in range(30000)]
        self.assertEqual(self.ask(conv), [900000, 900001, 900002])
        self.assertEqual(self.seen(), [{"keep": False, "sent": 30000, "ids": conv}])     # the first: nothing to keep
        for turn in range(6):
            # (as a chat goes: the last prompt but its last id - the generation prompt's - then the answer and more)
            conv = conv[:-1] + [rng.randrange(200000) for _ in range(rng.randrange(1, 300))]
            before = len(conv)
            self.assertEqual(len(self.ask(conv)), 3)
            rows = self.seen()
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["keep"])
            self.assertEqual(rows[0]["ids"], conv)
            self.assertLess(rows[0]["sent"], 301)
            self.assertEqual(self.engine.last["prompt_tokens"], before)

    def test_resend_whenever_the_engine_does_not_hold_it(self):
        self.start()
        rng = random.Random(8)
        a = [rng.randrange(200000) for _ in range(5000)]
        b = a[:700] + [rng.randrange(200000) for _ in range(4000)]      # another conversation, the same start
        self.ask(a)
        self.seen()
        # the engine started again: its state is empty, the server's memo is not
        self.engine.restart()
        a += [1, 2, 3]
        self.ask(a)
        self.assertEqual(self.seen(), [{"resend": 5000}, {"keep": False, "sent": len(a), "ids": a}])
        # another conversation: what the two share is kept, the rest sent
        self.ask(b)
        self.assertEqual(self.seen(), [{"keep": True, "sent": len(b) - 700, "ids": b}])
        # ... and back
        a += [4, 5]
        self.ask(a)
        self.assertEqual(self.seen(), [{"keep": True, "sent": len(a) - 700, "ids": a}])
        # a read stopped half way (the state holds half of what the server sent), then the conversation goes on
        a += [6]
        self.ask(a, max_new=7)
        self.seen()
        a += [7, 8]
        self.ask(a)
        self.assertEqual(self.seen(), [{"resend": len(a) - 2}, {"keep": False, "sent": len(a), "ids": a}])
        # a request the engine refused: its state is the one before, the next prompt still arrives whole in its ids
        a += [9]
        with self.assertRaises(ValueError):
            self.ask(a, max_new=8)
        self.seen()
        a += [10]
        self.ask(a)
        rows = self.seen()
        self.assertEqual(rows[-1]["ids"], a)
        # the same prompt again (a regenerate): all but its last id kept
        self.ask(a)
        self.assertEqual(self.seen(), [{"keep": True, "sent": 1, "ids": a}])
        # an edit far back: the state still holds the start
        a = a[:100] + [0] + a[101:]
        self.ask(a)
        self.assertEqual(self.seen(), [{"keep": True, "sent": len(a) - 100, "ids": a}])
        # a prompt that shares nothing: the whole line, no keep=
        c = [11, 12, 13]
        self.ask(c)
        self.assertEqual(self.seen(), [{"keep": False, "sent": 3, "ids": c}])

    def test_random_traffic_is_the_prompt_sent(self):
        self.start()
        rng = random.Random(9)
        convs = [[rng.randrange(200000) for _ in range(rng.randrange(1, 4000))] for _ in range(3)]
        for step in range(120):
            i = rng.randrange(3)
            op = rng.randrange(8)
            if op < 4:
                convs[i] = convs[i] + [rng.randrange(200000) for _ in range(rng.randrange(1, 200))]
            elif op == 4:
                convs[i] = convs[i][:rng.randrange(1, len(convs[i]) + 1)]
            elif op == 5:
                at = rng.randrange(len(convs[i]))
                convs[i] = convs[i][:at] + [convs[i][at] ^ 3] + convs[i][at + 1:]
            elif op == 6 and step % 5 == 0:
                self.engine.restart()
            max_new = rng.choice([1, 3, 7, 8, 20])
            try:
                self.ask(convs[i], max_new=max_new)
            except ValueError:
                self.assertEqual(max_new, 8)
            rows = [r for r in self.seen() if "ids" in r]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ids"], convs[i])

    def test_an_engine_that_does_not_say_it_never_gets_keep(self):
        self.start("old")
        a = list(range(100, 3000))
        for _ in range(3):
            a += [5, 6]
            self.ask(a)
            self.assertEqual(self.seen(), [{"keep": False, "sent": len(a), "ids": a}])
        self.assertNotIn("_keep_memo", self.engine.__dict__)

    def test_switch_off_and_pictures(self):
        self.start()
        a = list(range(100, 3000))
        self.ask(a)
        SV.GEN_KEEP = False
        a += [5]
        self.ask(a)
        SV.GEN_KEEP = True
        self.assertEqual([r["keep"] for r in self.seen()], [False, False])
        # (a request with pictures is a GENI line with the whole prompt; the fake reads its ids the same way)
        a += [6]
        self.ask(a, embeddings="pics.bin")
        self.assertEqual(self.seen(), [{"keep": False, "sent": len(a), "ids": a}])


if __name__ == "__main__":
    unittest.main()
