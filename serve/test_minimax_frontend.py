"""serve/test_minimax_frontend.py - MiniMax-M3.1-Flash in the frontend: its chat template rendered as transformers
renders it, and its output (reasoning, answer, tool calls) parsed from any split of the stream.

    python -m unittest serve.test_minimax_frontend          (MAYA_MM3_TEMPLATE=<the model's chat_template.jinja>)

The template test needs the model's chat_template.jinja (MAYA_MM3_TEMPLATE, default the EXL3 pack's in ~/models) and
skips without it; the parser tests need nothing.
"""
from __future__ import annotations

import json
import os
import random
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, MINIMAX_NS, OutputParser, effort_kwargs

NS = MINIMAX_NS
TEMPLATE = Path(os.environ.get("MAYA_MM3_TEMPLATE",
                               Path.home() / "models/MiniMax-M3.1-Flash-EXL3-3.5bpw/chat_template.jinja"))
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"},
        "opts": {"type": "object"}, "tags": {"type": "array"}}, "required": ["city"]}}}]
CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "reasoning_content": "I should call the tool.", "content": "",
     "tool_calls": [{"type": "function", "function": {"name": "get_weather",
                                                      "arguments": {"city": "Paris", "days": 2}}}]},
    {"role": "tool", "content": "{\"temp\": 21}"},
    {"role": "assistant", "content": "It is 21 degrees."},
    {"role": "user", "content": "And tomorrow?"},
]


def events(out: str, tools=TOOLS, thinking=True, seed=0):
    """The parser's events for `out` fed in random pieces: (reasoning, content, tool calls)."""
    rng = random.Random(seed)
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=True, minimax=True)
    evs, i = [], 0
    while i < len(out):
        n = rng.randint(1, 7)
        evs += p.feed(out[i:i + n])
        i += n
    evs += p.finish()
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [e.call for e in evs if e.kind == "tool_call"]
    streamed = "".join(e.text for e in evs if e.kind == "tool_args")
    return reasoning, content, calls, streamed


class Parser(unittest.TestCase):
    CALL = (NS + "<tool_call>\n" + NS + '<invoke name="get_weather">' + NS + "<city>Paris <Île>" + NS + "</city>" +
            NS + "<days>3" + NS + "</days>" + NS + "<opts>" + NS + "<unit>c" + NS + "</unit>" + NS + "<hourly>true" +
            NS + "</hourly>" + NS + "</opts>" + NS + "<tags>" + NS + "<item>a" + NS + "</item>" + NS + "<item>2" + NS +
            "</item>" + NS + "</tags>" + NS + "</invoke>\n" + NS + "</tool_call>")

    def test_reasoning_answer_call(self):
        out = "<mm:think>Let me look it up.</mm:think>Checking." + self.CALL
        for seed in range(20):
            r, c, calls, streamed = events(out, seed=seed)
            self.assertEqual(r, "Let me look it up.")
            self.assertEqual(c.strip(), "Checking.")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].name, "get_weather")
            want = {"city": "Paris <Île>", "days": 3, "opts": {"unit": "c", "hourly": True}, "tags": ["a", 2]}
            self.assertEqual(calls[0].arguments, want)
            self.assertEqual(json.loads(streamed), want)        # the streamed arguments: the same JSON

    def test_no_thinking(self):
        r, c, calls, _ = events("</mm:think>Just the answer.", seed=1)
        self.assertEqual((r, c.strip(), calls), ("", "Just the answer.", []))
        r, c, calls, _ = events("Plain answer.", thinking=False, seed=2)
        self.assertEqual((r, c.strip(), calls), ("", "Plain answer.", []))

    def test_two_calls_and_no_schema(self):
        call = (NS + "<tool_call>\n" + NS + '<invoke name="a">' + NS + "<x>1" + NS + "</x>" + NS + "</invoke>\n" + NS +
                '<invoke name="b">' + NS + "<y>hello" + NS + "</y>" + NS + "</invoke>\n" + NS + "</tool_call>")
        r, c, calls, _ = events("</mm:think>" + call, tools=None, seed=3)
        self.assertEqual([(k.name, k.arguments) for k in calls], [("a", {"x": 1}), ("b", {"y": "hello"})])


@unittest.skipUnless(TEMPLATE.exists(), f"no MiniMax chat template at {TEMPLATE}")
class Template(unittest.TestCase):
    def test_against_transformers(self):
        try:
            import jinja2  # noqa: F401
            from transformers.utils.chat_template_utils import render_jinja_template
        except ImportError:
            self.skipTest("transformers' chat template renderer is not installed")
        tpl = ChatTemplate(TEMPLATE)
        self.assertTrue(tpl.minimax)
        src = TEMPLATE.read_text(encoding="utf-8")
        for level, mode, effort in ((None, "adaptive", None), ("low", "adaptive", "low"), ("high", "enabled", "high"),
                                    ("max", "enabled", "max")):
            kw = {} if level is None else {"reasoning_effort": level}
            ours = tpl.render(CONVERSATION, tools=TOOLS, add_generation_prompt=True, **kw)
            ref_kw = {"thinking_mode": mode}
            if effort:
                ref_kw["reasoning_effort"] = effort
            theirs, _ = render_jinja_template(conversations=[CONVERSATION], tools=TOOLS, chat_template=src,
                                              add_generation_prompt=True, **ref_kw)
            self.assertEqual(ours, theirs[0], f"level {level}")
        off = tpl.render(CONVERSATION, tools=TOOLS, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(off.endswith("]~b]ai\n</mm:think>"))
        on = tpl.render(CONVERSATION, tools=TOOLS, add_generation_prompt=True, reasoning_effort="high")
        self.assertTrue(on.endswith("]~b]ai\n<mm:think>"))



class EffortNames(unittest.TestCase):
    def test_kwargs(self):   # (GLM's four for every template, MiniMax's own word beside them)
        self.assertEqual(effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(effort_kwargs("medium"), {"reasoning_effort": "high", "effort_name": "medium"})
        self.assertEqual(effort_kwargs("XHigh"), {"reasoning_effort": "max", "effort_name": "xhigh"})
        self.assertEqual(effort_kwargs("maximum"), {"reasoning_effort": "max"})
        self.assertEqual(effort_kwargs("none"), {"enable_thinking": False})


@unittest.skipUnless(TEMPLATE.exists(), f"no MiniMax chat template at {TEMPLATE}")
class Effort(unittest.TestCase):
    def test_five_levels(self):
        tpl = ChatTemplate(TEMPLATE)
        msgs = [{"role": "user", "content": "Hi"}]
        for level in ("low", "medium", "high", "xhigh", "max"):
            out = tpl.render(msgs, None, **effort_kwargs(level))
            self.assertIn(f"<effort>{level}</effort>", out, level)
            self.assertNotIn("effort_name", out)
            self.assertEqual(out.endswith("<mm:think>"), level != "low", level)   # (low: adaptive, no prefix)
        off = tpl.render(msgs, None, **effort_kwargs("none"))
        self.assertTrue(off.endswith("</mm:think>"))
        self.assertNotIn("<effort>", off)


if __name__ == "__main__":
    unittest.main()
