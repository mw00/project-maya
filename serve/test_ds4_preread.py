"""MAYA_DS4_PREREAD: after a DeepSeek thinking answer the server reads the chat's next turn start while nobody waits
(Service.preread), for clients that send the answer back without its reasoning - no GPU, no engine (mock engines).

    python -m unittest serve.test_ds4_preread -v
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import ds41_encoding as reference  # noqa: E402
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, openai_chunks, openai_collect, preread_on  # noqa: E402
import maya_names  # noqa: E402  (tools/, on the path server.py set)

ROOT = Path(__file__).resolve().parents[1]
ON = maya_names.read_as("MAYA_DS4_PREREAD")
SYSTEM = "You are a careful assistant. " * 20
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Looks a thing up.", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}}}}
ANSWERS = ["The answer is 42.", "Line one.\n\n- a bullet\n- another\n\n```python\nprint('hi')\n```",
           "A literal </think> inside, and <｜User｜> too.", "日本語の答え。😀"]


class Tok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [
        reference.bos_token, reference.eos_token, "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
        "<｜latest_reminder｜>", "<think>", "</think>", "｜DSML｜", reference.IMAGE_PLACEHOLDER]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}


class Calls(MockEngine):
    """Every generate call's prompt and max_new.  A one-token call (the pre-read) can take `read_s` to read, yielding
    None meanwhile as the real engine does while it reads, so a waiting request can stop it."""

    def __init__(self, *a, read_s=0.0, **k):
        super().__init__(*a, **k)
        self.calls, self.read_s, self.stopped = [], read_s, threading.Event()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new))
        if max_new == 1 and self.read_s:
            end = time.time() + self.read_s
            try:
                while time.time() < end:
                    yield None
                    time.sleep(0.005)
            except GeneratorExit:
                self.stopped.set()
                raise
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def wait_preread(timeout=5.0):
    end = time.time() + timeout
    for t in [t for t in threading.enumerate() if t.name == "pre-read"]:
        t.join(max(0.0, end - time.time()))


class PreRead(unittest.TestCase):
    def service(self, script="Let me think.</think>The answer is 42.", **k):
        tok = Tok()
        eng = Calls(tok, script, max_context=1 << 20, **k)
        return Service(eng, tok, ChatTemplate(ROOT / "tools/ds4/chat_template.jinja")), eng

    def request(self, svc, messages, env, tools=None, effort="high"):
        """as the OpenAI handler sends it: _preread only when the setting is on"""
        req = {"messages": messages, "reasoning_effort": effort, **({"tools": tools} if tools else {})}
        msgs, schemas, kw = openai_to_messages(req, deepseek=True)
        ids, thinking, max_new = svc.prepare(msgs, schemas, kw, 200)
        with unittest.mock.patch.dict(os.environ, env):
            if ON not in env:
                os.environ.pop(ON, None)
            if preread_on():
                req = {**req, "_preread": (msgs, schemas, kw)}
            out = openai_collect(openai_chunks(svc, req, ids, thinking, schemas, max_new, threading.Event()))
        wait_preread()
        return ids, out["choices"][0]["message"]

    def next_ids(self, svc, messages, answer, tools=None, effort="high"):
        req = {"messages": messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": "And why?"}],
               "reasoning_effort": effort, **({"tools": tools} if tools else {})}
        msgs, schemas, kw = openai_to_messages(req, deepseek=True)
        return svc.prepare(msgs, schemas, kw, 200)[0]

    def test_settings(self):
        self.assertFalse(preread_on({}))                    # (off by default)
        self.assertTrue(preread_on({ON: "1"}))
        self.assertFalse(preread_on({ON: "0"}))
        self.assertFalse(preread_on({ON: "yes"}))

    def test_its_prompt_starts_the_next_turn(self):
        svc, _ = self.service()
        for tools in (None, [TOOL]):
            for answer in ANSWERS:
                for hist in ([], [{"role": "user", "content": "first"}, {"role": "assistant", "content": "earlier"}]):
                    messages = [{"role": "system", "content": SYSTEM}, *hist, {"role": "user", "content": "question"}]
                    msgs, schemas, kw = openai_to_messages({"messages": messages, "reasoning_effort": "high",
                                                            **({"tools": tools} if tools else {})}, deepseek=True)
                    pre = svc.preread_ids(msgs, schemas, kw, answer)
                    nxt = self.next_ids(svc, messages, answer, tools)
                    with self.subTest(tools=bool(tools), answer=answer[:12], hist=bool(hist)):
                        self.assertTrue(pre)
                        self.assertEqual(pre[-1], svc.ds_user_id)
                        self.assertEqual(nxt[:len(pre)], pre)
                        self.assertLess(len(pre), len(nxt))

    def test_after_a_thinking_answer_one_read(self):
        svc, eng = self.service()
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the answer?"}]
        _, msg = self.request(svc, messages, {ON: "1"})
        self.assertEqual(msg["content"].strip(), "The answer is 42.")
        self.assertEqual([n for _, n in eng.calls][1:], [1])     # the answer, then one GEN 1
        pre = eng.calls[1][0]
        self.assertEqual(self.next_ids(svc, messages, msg["content"])[:len(pre)], pre)

    def test_off_no_thinking_or_a_tool_call_no_read(self):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the answer?"}]
        svc, eng = self.service()
        self.request(svc, messages, {})                          # unset
        self.assertEqual(len(eng.calls), 1)
        svc, eng = self.service(script="The answer is 42.")
        self.request(svc, messages, {ON: "1"}, effort="none")    # no reasoning: the next prompt already holds the answer
        self.assertEqual(len(eng.calls), 1)
        call = ('Let me look.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
                '<｜DSML｜ parameter name="q" string="true">x</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        svc, eng = self.service(script=call)
        _, msg = self.request(svc, messages, {ON: "1"}, tools=[TOOL])   # a tool call: the client's next message is its result
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(len(eng.calls), 1)

    def test_a_request_meanwhile_stops_it(self):
        svc, eng = self.service(read_s=30.0)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the answer?"}]
        req = {"messages": messages, "reasoning_effort": "high", "_preread": None}
        msgs, schemas, kw = openai_to_messages(req, deepseek=True)
        ids, thinking, max_new = svc.prepare(msgs, schemas, kw, 200)
        openai_collect(openai_chunks(svc, {**req, "_preread": (msgs, schemas, kw)}, ids, thinking, schemas, max_new,
                                     threading.Event()))
        for _ in range(200):                                     # the pre-read has the engine
            if len(eng.calls) == 2:
                break
            time.sleep(0.01)
        self.assertEqual(eng.calls[1][1], 1)
        t0 = time.time()
        other = {"messages": [{"role": "user", "content": "Another chat."}], "reasoning_effort": "high"}
        m2, s2, k2 = openai_to_messages(other, deepseek=True)
        ids2, th2, mx2 = svc.prepare(m2, s2, k2, 200)
        out = openai_collect(openai_chunks(svc, other, ids2, th2, s2, mx2, threading.Event()))
        took = time.time() - t0
        wait_preread()
        self.assertTrue(eng.stopped.is_set())
        self.assertLess(took, 5.0)                               # not the pre-read's 30 s
        self.assertEqual(out["choices"][0]["message"]["content"].strip(), "The answer is 42.")


if __name__ == "__main__":
    unittest.main()
