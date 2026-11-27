"""serve/test_inkling2_think_budget.py - Inkling 2's thinking budget: its reasoning is a typed segment
(<|content_thinking|> .. <|end_message|>), not a think block, so the engine (maya-ik2: GEN's think_* keys, which the
server sends) counts the segment's tokens and closes it at the budget - at the next sentence or line end, or THINK_GRACE
tokens later whatever the token - with the segment's end and the next message's opening in its own decode; the model
goes on with its text segment, or a call to one of the request's tools.  This file: the server's part (the keys, the
lowering of the budget, the dashboard field and its endpoint); the engine's part is tested on a GPU.

    python -m unittest serve.test_inkling2_think_budget

Offline, no GPU: the same stand-in tokenizer and scripted engine as serve/test_inkling2_frontend.py.
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

from serve import server
from serve.frontend import openai_to_messages
from serve.server import Service
from serve.test_inkling2_frontend import (ASK, CALL, END, EOS, MODEL, TEXT, THINK, TOOLS, InkTok, Scripted, public)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]
import maya_quants  # noqa: E402

# (a byte is a token in the stand-in tokenizer; no text repeats, or the loop guard would step in)
LONG = "".join(f"Step number {i} of the plan is done. " for i in range(80))
NO_STOP = " ".join(f"w{i}" for i in range(400))      # (no sentence or line end at all)
ANSWER = "".join(f"Item {i} of the answer. " for i in range(80))


def first_end(text, n):
    """text up to its first sentence end ("." with a space after it) at or past n characters."""
    i = text.index(". ", n - 1)
    return text[:i + 1]


def service(*scripts, budget=0, starts=False):
    tok = InkTok()
    eng = Scripted(tok, *scripts)
    if starts:
        eng.info["starts"] = 1
    svc = Service(eng, tok, public())
    svc.think_budget = budget
    return svc, eng, tok


def run(svc, req, max_new=5000):
    msgs, tools, kw = openai_to_messages(req)
    ids, thinking, _ = svc.prepare(msgs, tools, kw, max_new)
    sampling = {k: kw.get(k) for k in ("_prefill", "_blocked", "_think_resume", "_think_after")}
    events, done = [], None
    for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
        if kind == "event":
            events.append(x)
        elif kind == "done":
            done = x
    return {"content": "".join(e.text for e in events if e.kind == "content"),
            "reasoning": "".join(e.text for e in events if e.kind == "reasoning"),
            "calls": [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], "done": done}


class Lowering(unittest.TestCase):
    def test_other_models_unchanged(self):   # (one forced token: </think>)
        low = Service.lowered_think_budget
        self.assertEqual(low(32768, 262144), (32768 + 512, 32768))
        self.assertEqual(low(32768, 0), (32768 + 512, 32768))             # no limit named
        self.assertEqual(low(32768, 4096), (32768, None))                 # no room for a grace
        self.assertEqual(low(1000, 4096), (1000 + 512, 1000))
        self.assertEqual(low(1000, 1500), (1000 + 249, 1000))             # half of 1500 - 1000 - 1
        self.assertEqual(low(32768, 1024), (511, None))                   # tiny limit: a reserve of 512 kept
        self.assertEqual(low(256, 400), (199, None))

    def test_inkling_forces_three_tokens(self):
        low = Service.lowered_think_budget
        self.assertEqual(low(32768, 262144, 3), (32768 + 512, 32768))
        self.assertEqual(low(131072, 262144, 3), (131072 + 512, 131072))  # the owner's: 131072 inside 262144
        self.assertEqual(low(1000, 1500, 3), (1000 + 248, 1000))          # half of 1500 - 1000 - 3
        self.assertEqual(low(32768, 1024, 3), (509, None))
        self.assertEqual(low(300, 400, 3), (197, None))


class Field(unittest.TestCase):
    def test_default_and_range(self):
        cfg = {"exe": "x/maya-ik2", "args": ["--model", "m.gguf"], "installer": {"family": "ik2"}}
        self.assertEqual(maya_quants.default_thinking_budget(cfg), 0)       # (an option: none unless set)
        self.assertIsNone(maya_quants.thinking_budget_for(dict(cfg, thinking_budget=8192)))
        for ok in (0, 256, 32768, 131072, 262144):
            self.assertEqual(maya_quants.check_thinking_budget(ok), ok)

    def test_endpoint_takes_inkling(self):
        svc, eng, tok = service("x")
        eng.info["think"] = 1
        self.assertEqual(svc.think_close_ids, [tok.special_tokens[END], tok.special_tokens[MODEL]])
        self.assertEqual(svc.think_open_id, tok.special_tokens[THINK])
        self.assertEqual(svc.think_text_id, tok.special_tokens[TEXT])
        self.assertIsNone(svc.think_end_id)                    # (no </think>: the close is the segment's)
        self.assertTrue(svc.think_budget_info()["supported"])
        for n in (131072, 0, 256):
            self.assertEqual(svc.set_think_budget(n)["budget"], n)
            self.assertEqual(svc.think_budget, n)
        with self.assertRaises(ValueError):
            svc.set_think_budget(100)


class Keys(unittest.TestCase):
    """What the server asks of the engine (the close itself is the engine's: maya-ik2's decode, tested live)."""

    def keys(self, eng):
        return server.StrataEngine.sampling_keys(eng.sampling[0]).split()

    def test_the_close_goes_to_the_engine(self):
        svc, eng, tok = service(MODEL + THINK + "Short." + END + MODEL + TEXT + "Hi." + END + EOS, budget=1000, starts=True)
        eng.info["think"] = 1
        run(svc, {"messages": ASK})
        k = {x.split("=")[0]: x.split("=")[1] for x in self.keys(eng) if "=" in x}
        sp = tok.special_tokens
        self.assertEqual((int(k["think_open"]), int(k["think_end"])), (sp[THINK], sp[END]))
        self.assertEqual((int(k["think_soft"]), int(k["think_hard"])), (1000, 1000 + server.THINK_GRACE))
        self.assertEqual(k["think_after"], f"{sp[MODEL]},{sp[TEXT]}")        # (no tools: the text segment follows)
        self.assertNotIn("think_budget", k)

    def test_a_call_stays_a_choice(self):
        svc, eng, tok = service(MODEL + THINK + "Short." + END + MODEL + TEXT + "Hi." + END + EOS, budget=1000, starts=True)
        eng.info["think"] = 1
        run(svc, {"messages": ASK, "tools": TOOLS})
        sp, name = tok.special_tokens, tok.encode("get_weather")
        k = {x.split("=")[0]: x.split("=")[1] for x in self.keys(eng) if "=" in x}
        want = [[sp[MODEL], sp[TEXT]], [sp[MODEL], *name, sp[CALL]]]
        self.assertEqual(k["think_after"], ";".join(",".join(map(str, q)) for q in want))

    def test_tiny_limit_and_no_limit(self):
        svc, eng, tok = service("Ok." + END + EOS, budget=32768)
        eng.info["think"] = 1
        run(svc, {"messages": ASK}, max_new=600)       # (a side request: reserve 300, the three forced tokens)
        k = {x.split("=")[0]: int(x.split("=")[1]) for x in self.keys(eng) if x.split("=")[0] in ("think_soft", "think_hard")}
        self.assertEqual(k, {"think_soft": 297, "think_hard": 297})
        svc, eng, _ = service("Ok." + END + EOS, budget=0)   # 0: none
        eng.info["think"] = 1
        run(svc, {"messages": ASK})
        self.assertFalse([x for x in self.keys(eng) if x.startswith("think_")])

    def test_no_thinking_no_keys(self):
        svc, eng, _ = service(TEXT + "Hello." + END + EOS, budget=1000)
        eng.info["think"] = 1
        run(svc, {"messages": ASK, "reasoning_effort": "none"})
        self.assertFalse([x for x in self.keys(eng) if x.startswith("think_")])

    def test_an_engine_that_does_not_read_them(self):
        svc, eng, _ = service("Ok." + END + EOS, budget=1000)     # (no INFO think=1: an older maya-ik2)
        self.assertFalse(svc.think_budget_info()["supported"])
        with self.assertRaises(ValueError):
            svc.set_think_budget(4096)
        run(svc, {"messages": ASK})
        self.assertFalse([x for x in self.keys(eng) if x.startswith("think_")])


if __name__ == "__main__":
    unittest.main()
