"""serve/test_minimax_frontend.py - MiniMax-M3.1-Flash in the frontend: Maya's chat template (tools/mm3/chat_template.jinja)
rendered as MiniMax's own template renders it, its thinking levels, and the model's output (reasoning, answer, tool
calls) parsed from any split of the stream.

    python -m unittest serve.test_minimax_frontend          (MAYA_MM3_TEMPLATE=<another chat_template.jinja to check>)

Offline: the official template is a copy in serve/fixtures/minimax_m31 (its sha256 checked); the test against
transformers' own renderer runs where transformers is installed and skips elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import unittest
from pathlib import Path

from serve.frontend import (AUTO, ChatTemplate, MINIMAX_NS, OutputParser, effort_kwargs, effort_level, effort_name,
                            openai_to_messages)

ROOT = Path(__file__).resolve().parents[1]
NS = MINIMAX_NS
TEMPLATE = Path(os.environ.get("MAYA_MM3_TEMPLATE", ROOT / "tools" / "mm3" / "chat_template.jinja"))
OFFICIAL = ROOT / "serve" / "fixtures" / "minimax_m31" / "chat_template.jinja"
OFFICIAL_SHA256 = "961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431"   # (54e4e69's, unchanged)
LEVELS = ("low", "medium", "high", "xhigh", "max")
ADAPTIVE = ("Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step "
            "reasoning, or when analyzing function/tool results.\n")
DISABLED = "Current thinking mode: disabled. Do not output any thinking process.\n"
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
                                                      "arguments": {"city": "Paris", "days": 2,
                                                                    "opts": {"unit": "c"}, "tags": ["a", True]}}}]},
    {"role": "tool", "content": "{\"temp\": 21}"},
    {"role": "assistant", "content": "It is 21 degrees."},
    {"role": "user", "content": "And tomorrow?"},
]
# what the server hands the template: an OpenAI request through openai_to_messages (pictures in a user turn and in a
# tool result, a root-less chat with no system message)
PICTURES = {"messages": [
    {"role": "user", "content": [{"type": "text", "text": "What is in"},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                 {"type": "text", "text": " this?"}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": "a map:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]},
], "tools": TOOLS}


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
        r, c, calls, _ = events("</mm:think>Just the answer.", seed=1)   # (adaptive: it chose not to think)
        self.assertEqual((r, c.strip(), calls), ("", "Just the answer.", []))
        r, c, calls, _ = events("Plain answer.", thinking=False, seed=2)  # (Off: the prompt closed the think block)
        self.assertEqual((r, c.strip(), calls), ("", "Plain answer.", []))

    def test_string_unions(self):   # (a schema that admits a string keeps "123" / "true" / "null" text a string)
        props = {"a": {"type": ["string", "null"]}, "b": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                 "c": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/S"}]}, "d": {"enum": ["1", "2"]},
                 "e": {"allOf": [{"type": "string"}, {}]}, "f": {"type": ["integer", "null"]}, "g": {}}
        tools = [{"type": "function", "function": {"name": "t", "parameters": {
            "type": "object", "properties": props, "$defs": {"S": {"type": "string"}}}}}]
        vals = {"a": "123", "b": "true", "c": "null", "d": "1", "e": "2.5", "f": "7", "g": "8"}
        call = NS + "<tool_call>\n" + NS + '<invoke name="t">' + "".join(
            NS + f"<{k}>{v}" + NS + f"</{k}>" for k, v in vals.items()) + NS + "</invoke>\n" + NS + "</tool_call>"
        for seed in range(10):
            _, _, calls, streamed = events("</mm:think>" + call, tools=tools, seed=seed)
            want = {"a": "123", "b": "true", "c": "null", "d": "1", "e": "2.5", "f": 7, "g": 8}
            self.assertEqual(calls[0].arguments, want)
            self.assertEqual(json.loads(streamed), want)

    def test_two_calls_and_no_schema(self):
        call = (NS + "<tool_call>\n" + NS + '<invoke name="a">' + NS + "<x>1" + NS + "</x>" + NS + "</invoke>\n" + NS +
                '<invoke name="b">' + NS + "<y>hello" + NS + "</y>" + NS + "</invoke>\n" + NS + "</tool_call>")
        r, c, calls, _ = events("</mm:think>" + call, tools=None, seed=3)
        self.assertEqual([(k.name, k.arguments) for k in calls], [("a", {"x": 1}), ("b", {"y": "hello"})])


def official_render(messages, tools=None, **kw):
    """MiniMax's own template, rendered with the frontend's Jinja settings (transformers')."""
    tpl = ChatTemplate(OFFICIAL)
    tools = [t if "function" in t else {"type": "function", "function": t} for t in tools or []] or None
    return tpl.template.render(messages=messages, tools=tools, add_generation_prompt=True, **kw)


def requests():
    """(name, messages, tools) the shared-part checks run on: tools and tool calls, pictures and tool results with
    pictures (as the server passes them), a system prompt, none, a root (MiniMax's high-priority system) prompt."""
    msgs, tools, _ = openai_to_messages(PICTURES)
    return [("tools", CONVERSATION, TOOLS), ("pictures", msgs, tools),
            ("plain", [{"role": "user", "content": "Hi"}], None),
            ("root", [{"role": "root", "content": "You are MiniMax."}, {"role": "system", "content": "Be brief."},
                      {"role": "user", "content": "Hi"}], None)]


class Template(unittest.TestCase):
    def test_official_copy(self):
        self.assertEqual(hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(), OFFICIAL_SHA256)

    def test_shared_part_is_official(self):
        """Every level but Off - Auto and the five efforts - writes MiniMax's own prompt, byte for byte."""
        tpl = ChatTemplate(TEMPLATE)
        self.assertTrue(tpl.minimax)
        for name, msgs, tools in requests():
            for level in (None, AUTO) + LEVELS:
                kw = {} if level is None else effort_kwargs(level)
                ours = tpl.render(msgs, tools=tools, **kw)
                theirs = official_render(msgs, tools, **({"reasoning_effort": level} if level in LEVELS else {}))
                self.assertEqual(ours, theirs, f"{name}, level {level}")
                self.assertIn("<model_id>8bb8149fe</model_id>", ours)
                self.assertTrue(ours.endswith("]~b]ai\n"), f"{name}, level {level}")   # (adaptive: no prefix)
                self.assertEqual(f"<effort>{level}</effort>" in ours, level in LEVELS)

    def test_off(self):
        """Off: MiniMax-M3's 'disabled' line instead of the adaptive one, the turn opened with </mm:think> - the rest is
        MiniMax's prompt."""
        tpl = ChatTemplate(TEMPLATE)
        for name, msgs, tools in requests():
            off = tpl.render(msgs, tools=tools, **effort_kwargs("none"))
            self.assertTrue(off.endswith("]~b]ai\n</mm:think>"), name)
            self.assertNotIn("<effort>", off)
            self.assertEqual(off[:-len("</mm:think>")].replace(DISABLED, ADAPTIVE), official_render(msgs, tools), name)

    def test_model_id(self):
        tpl = ChatTemplate(TEMPLATE)
        msgs = [{"role": "user", "content": "Hi"}]
        self.assertIn("<model_id>8bb8149fe</model_id>", tpl.render(msgs, None, **effort_kwargs("none")))
        self.assertIn("<model_id>x1</model_id>", tpl.render(msgs, None, template_model_id="x1"))

    def test_against_transformers(self):
        try:
            import jinja2  # noqa: F401
            from transformers.utils.chat_template_utils import render_jinja_template
        except ImportError:
            self.skipTest("transformers' chat template renderer is not installed")
        tpl = ChatTemplate(TEMPLATE)
        src, official = TEMPLATE.read_text(encoding="utf-8"), OFFICIAL.read_text(encoding="utf-8")
        for level in (None, AUTO) + LEVELS:
            ours = tpl.render(CONVERSATION, tools=TOOLS, **({} if level is None else effort_kwargs(level)))
            ref_kw = {"reasoning_effort": level} if level in LEVELS else {}
            for source in (src, official):
                theirs, _ = render_jinja_template(conversations=[CONVERSATION], tools=TOOLS, chat_template=source,
                                                  add_generation_prompt=True, **ref_kw)
                self.assertEqual(ours, theirs[0], f"level {level}")
        off, _ = render_jinja_template(conversations=[CONVERSATION], tools=TOOLS, chat_template=src,
                                       add_generation_prompt=True, thinking_mode="disabled")
        self.assertEqual(tpl.render(CONVERSATION, tools=TOOLS, **effort_kwargs("none")), off[0])


class EffortNames(unittest.TestCase):
    def test_kwargs(self):   # (GLM's four for every template, MiniMax's own word beside them; auto: none at all)
        self.assertEqual(effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(effort_kwargs("medium"), {"reasoning_effort": "high", "effort_name": "medium"})
        self.assertEqual(effort_kwargs("XHigh"), {"reasoning_effort": "max", "effort_name": "xhigh"})
        self.assertEqual(effort_kwargs("maximum"), {"reasoning_effort": "max"})
        self.assertEqual(effort_kwargs("none"), {"enable_thinking": False})
        self.assertEqual(effort_kwargs("auto"), {"reasoning_effort": AUTO})
        self.assertEqual(effort_kwargs("Default"), {"reasoning_effort": AUTO})
        self.assertEqual((effort_level("auto"), effort_name("auto"), effort_name("medium")), (AUTO, AUTO, "medium"))
        with self.assertRaises(ValueError):
            effort_kwargs("extreme")

    def test_auto_elsewhere(self):
        """auto on another model's template: its own default (GLM's template: Max, as with no level)."""
        glm = ChatTemplate(ROOT / "serve" / "glm5_chat_template.jinja")
        msgs = [{"role": "user", "content": "Hi"}]
        self.assertEqual(glm.render(msgs, None, **effort_kwargs("auto")), glm.render(msgs, None))


class Effort(unittest.TestCase):
    def test_levels(self):
        tpl = ChatTemplate(TEMPLATE)
        msgs = [{"role": "user", "content": "Hi"}]
        for level in LEVELS:
            out = tpl.render(msgs, None, **effort_kwargs(level))
            self.assertIn(f"<effort>{level}</effort>", out, level)
            self.assertIn(ADAPTIVE, out, level)
            self.assertNotIn("effort_name", out)
            self.assertTrue(out.endswith("]~b]ai\n"), level)            # (adaptive: the model decides)
        auto = tpl.render(msgs, None, **effort_kwargs("auto"))
        self.assertNotIn("<effort>", auto)
        self.assertEqual(auto, tpl.render(msgs, None))
        off = tpl.render(msgs, None, **effort_kwargs("none"))
        self.assertTrue(off.endswith("</mm:think>"))
        self.assertIn(DISABLED, off)
        self.assertNotIn("<effort>", off)


if __name__ == "__main__":
    unittest.main()
