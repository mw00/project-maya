"""serve/test_ernie55.py - ERNIE-5.5-Flash in the frontend and the server: its chat template (tools/ds4/
ernie55_chat_template.jinja, the pack's) detected and rendered by Baidu's renderer (serve/ernie55_encoding.py) with
thinking on and off, its answers (reasoning, answer, <ernie:tool_call> invoke blocks) parsed, and its turn's end
(<|ernie:im_end|>) ending the answer without reaching its text.

    python -m unittest serve.test_ernie55

Offline, no GPU: the tokenizer a small stand-in with the template's tags as single tokens, the engine a scripted one.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, Service

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools" / "ds4" / "ernie55_chat_template.jinja"
IM_START, IM_END, MASK = "<|ernie:im_start|>", "<|ernie:im_end|>", "<|ERNIE_MASK|>"
OPEN, CLOSE = "<ernie:think>", "</ernie:think>"
CALL_OPEN, CALL_CLOSE = "<ernie:tool_call>", "</ernie:tool_call>"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
ASK = [{"role": "user", "content": "What is the weather in Lyon?"}]
CALL = CALL_OPEN + '<invoke name="get_weather"><parameter name="city">Lyon</parameter></invoke>' + CALL_CLOSE


class ErnieTok(ByteTokenizer):
    """ByteTokenizer with ERNIE-5.5's turn, thinking and tool tags as one token each (as its vocabulary has them)."""
    SPECIALS = ByteTokenizer.SPECIALS + [IM_START, IM_END, MASK, OPEN, CLOSE, CALL_OPEN, CALL_CLOSE]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens[IM_END]}


class Scripted:
    """Request k's answer is scripts[k] (the last one again after that), then the turn's end and, as a model that
    goes on would, a next turn's opening - which the server must not show."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.scripts = tok, 32768, scripts
        self.prompts = []
        self.info = {"engine": "ds4", "engine_kind": "ds4", "dspark": "0", "mtp": "0"}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        text = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        tail = [self.tok.special_tokens[IM_END], self.tok.special_tokens[MASK]] + list(b"Whoa")
        for t in (self.tok.encode(text, parse_special=True) + tail)[:max_new]:
            yield t


def service(*scripts):
    tok = ErnieTok()
    eng = Scripted(tok, *scripts)
    return Service(eng, tok, ChatTemplate(TEMPLATE)), eng, tok


def answer(svc, req):
    msgs, tools, kw = openai_to_messages(req)
    ids, thinking, max_new = svc.prepare(msgs, tools, kw, 500)
    sampling = {k: kw.get(k) for k in ("_prefill", "_blocked", "_think_resume")}
    events = []
    for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
        if kind == "event":
            events.append(x)
    return {"content": "".join(e.text for e in events if e.kind == "content"),
            "reasoning": "".join(e.text for e in events if e.kind == "reasoning"),
            "calls": [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]}


class Template(unittest.TestCase):
    def test_detected(self):
        t = ChatTemplate(TEMPLATE)
        self.assertTrue(t.ernie)
        self.assertFalse(t.minimax or t.deepseek or t.hy4)

    def test_other_templates_are_not(self):
        for path in ("serve/chat_template.jinja", "serve/glm5_chat_template.jinja", "tools/mm3/chat_template.jinja"):
            self.assertFalse(ChatTemplate(ROOT / path).ernie, path)

    def test_render(self):
        t = ChatTemplate(TEMPLATE)
        on = t.render(ASK, add_generation_prompt=True)
        off = t.render(ASK, add_generation_prompt=True, enable_thinking=False)
        for p in (on, off):
            self.assertTrue(p.startswith(MASK + IM_START + "system\n"), p[:80])
            self.assertIn(IM_START + "user\nWhat is the weather in Lyon?" + IM_END, p)
            self.assertIn(IM_START + "assistant\n", p)
        self.assertTrue(off.endswith(OPEN + CLOSE), off[-60:])
        self.assertFalse(on.endswith(CLOSE), on[-60:])

    def test_tools_rendered(self):
        p = ChatTemplate(TEMPLATE).render(ASK, tools=TOOLS, add_generation_prompt=True)
        self.assertIn("get_weather", p)


class Answers(unittest.TestCase):
    def test_plain_answer_stops_at_the_turns_end(self):
        svc, eng, tok = service("Lyon is rainy.")
        out = answer(svc, {"messages": ASK, "reasoning_effort": "none"})
        self.assertEqual(out["content"], "Lyon is rainy.")
        self.assertEqual((out["reasoning"], out["calls"]), ("", []))
        self.assertTrue(tok.decode(eng.prompts[0]).endswith(OPEN + CLOSE))

    def test_reasoning(self):
        svc, eng, tok = service("It rains there today." + CLOSE + "Rainy, 7 degrees.")
        out = answer(svc, {"messages": ASK, "reasoning_effort": "high"})
        self.assertEqual(out["reasoning"].strip(), "It rains there today.")
        self.assertEqual(out["content"].strip(), "Rainy, 7 degrees.")

    def test_tool_call(self):
        svc, _, _ = service(CALL)
        out = answer(svc, {"messages": ASK, "tools": TOOLS, "reasoning_effort": "none"})
        self.assertEqual(out["calls"], [("get_weather", {"city": "Lyon"})])
        self.assertNotIn(IM_END, out["content"])
        self.assertEqual(out["content"].strip(), "")

    def test_tool_result_turn(self):
        svc, eng, tok = service("Rainy, 7 degrees.")
        msgs = ASK + [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Lyon"}'}}]},
                      {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 7, "sky": "rain"}'}]
        out = answer(svc, {"messages": msgs, "tools": TOOLS, "reasoning_effort": "none"})
        self.assertEqual(out["content"], "Rainy, 7 degrees.")
        prompt = tok.decode(eng.prompts[0])
        self.assertIn("get_weather", prompt)
        self.assertIn('"temp_c": 7', prompt)


if __name__ == "__main__":
    unittest.main()
