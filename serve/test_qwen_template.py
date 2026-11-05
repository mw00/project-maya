"""serve/test_qwen_template.py - the Qwen3.8 prompt Maya renders is Qwen's reference one.

serve/chat_golden.json holds transformers' apply_chat_template output (`hf`) for ten conversations (thinking, a past
turn's reasoning, thinking off, tool call + result, no generation prompt, unicode, whitespace); every one goes through
the server's own path (openai_to_messages -> ChatTemplate.render) and must come out the same text.  serve/chat_template
.jinja is unsloth's Qwen3.8-Flash-Next GGUF copy (Qwen's own plus merged system/developer messages and stricter tool
argument checks: the same text for these).  Also: which templates count as Qwen's (tools in OpenAI's form, past
thinking kept) - GLM-5's, DeepSeek's and MiniMax's are not.

    python -m unittest serve.test_qwen_template -v
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402

QWEN = ROOT / "serve/chat_template.jinja"


def maya_render(tpl: ChatTemplate, case: dict) -> str:
    req = {"messages": copy.deepcopy(case["messages"])}
    if case.get("tools"):
        req["tools"] = copy.deepcopy(case["tools"])
    if case.get("enable_thinking") is False:
        req["chat_template_kwargs"] = {"enable_thinking": False}
    m, t, kw = openai_to_messages(req)
    return tpl.render(m, tools=t, add_generation_prompt=case.get("add_generation_prompt", True), **kw)


class Golden(unittest.TestCase):
    def test_every_golden_case_renders_as_transformers_does(self):
        tpl = ChatTemplate(QWEN)
        cases = json.loads((ROOT / "serve/chat_golden.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 10)
        for case in cases:
            with self.subTest(case["name"]):
                self.assertEqual(maya_render(tpl, case), case["hf"])

    def test_tools_flat_or_wrapped_render_the_same(self):
        """A client's tools in OpenAI's form reach the template in that form (Qwen's reference), not flattened."""
        tpl = ChatTemplate(QWEN)
        case = next(c for c in json.loads((ROOT / "serve/chat_golden.json").read_text(encoding="utf-8"))
                    if c.get("tools"))
        out = maya_render(tpl, case)
        self.assertIn('<tools>\n{"type": "function", "function": {"name": "get_weather"', out)
        m, t, kw = openai_to_messages({"messages": case["messages"], "tools": case["tools"]})
        self.assertEqual(tpl.render(m, tools=[x.get("function", x) for x in t], **kw), out)


class Family(unittest.TestCase):
    def test_which_templates_are_qwens(self):
        q = ChatTemplate(QWEN)
        self.assertTrue(q.qwen)
        self.assertTrue(q.keeps_thinking)
        for path in ("serve/glm5_chat_template.jinja", "tools/ds4/chat_template.jinja", "tools/mm3/chat_template.jinja",
                     "tools/glm_chat_template.jinja"):
            with self.subTest(path):
                t = ChatTemplate(ROOT / path)
                self.assertFalse(t.qwen)
                self.assertFalse(t.keeps_thinking)

    def test_past_thinking_sent_back_is_rendered_as_written(self):
        """The web chat sends an answer's reasoning back for a Qwen3.x model: the history then holds the turn as the
        model wrote it after the generation prompt (<think>\\n reasoning \\n</think>\\n\\n answer <|im_end|>)."""
        tpl = ChatTemplate(QWEN)
        u1 = {"role": "user", "content": "What is 17*23?"}
        first = tpl.render(*openai_to_messages({"messages": [u1]})[:2])
        self.assertTrue(first.endswith("<|im_start|>assistant\n<think>\n"))
        written = "17*20 + 17*3 = 391\n</think>\n\n391<|im_end|>"
        a1 = {"role": "assistant", "content": "391", "reasoning_content": "17*20 + 17*3 = 391\n"}
        m, t, kw = openai_to_messages({"messages": [u1, a1, {"role": "user", "content": "Thanks"}]})
        self.assertTrue(tpl.render(m, t, **kw).startswith(first + written + "\n<|im_start|>user\nThanks"))
        # without it (an API client that drops the reasoning) the past turn's think block is empty, as Qwen's is
        m, t, kw = openai_to_messages({"messages": [u1, {"role": "assistant", "content": "391"},
                                                    {"role": "user", "content": "Thanks"}]})
        self.assertIn("<think>\n\n</think>\n\n391<|im_end|>", tpl.render(m, t, **kw))


if __name__ == "__main__":
    unittest.main()
