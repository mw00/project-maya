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


class ServerClose(unittest.TestCase):
    """The thinking closed by the server (its budget at a sentence's or line's end): the tokens the engine goes on from
    are the ones the next request's history renders for that turn, so the next message continues the session."""

    def svc(self, model="qwen"):
        sys.path.insert(0, str(ROOT / "tools"))
        from serve.server import MockEngine, Service
        from serve.test_prompt_memo import MODELS, make_tokenizer
        path, pre, control, user = MODELS[model]
        tok = make_tokenizer(pre, control, user)
        return Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / path)), tok

    def test_the_close_is_the_next_turns_rendering(self):
        svc, tok = self.svc()
        think_end = tok.encode("</think>", parse_special=True)[0]
        im_end = tok.encode("<|im_end|>", parse_special=True)[0]
        u1 = {"role": "user", "content": "What is 17*23?"}
        u2 = {"role": "user", "content": "And times 3?"}
        ids = svc.render_ids(*openai_to_messages({"messages": [u1]}))
        for reasoning in ("Let me add: 17*20 = 340, plus 51.\nSo it is 391.", "First 17*20 = 340.\n\n",
                          "ünïcödé 日本語 🚀 and code:\n```python\nx = 1\n```\n  \n"):
            with self.subTest(repr(reasoning[:20])):
                produced = tok.encode(reasoning)
                closed = svc._closed_as_rendered(ids, produced, think_end)
                self.assertIsNotNone(closed)
                self.assertEqual(closed[-3:-2], [think_end])
                keep = len(tok.encode(reasoning.rstrip())) - 2   # (only the trailing whitespace is written anew)
                self.assertEqual(closed[:keep], produced[:keep])
                written = closed + tok.encode("391") + [im_end]   # the engine's answer after the close
                svc._note_written(ids, written, None)
                a1 = {"role": "assistant", "content": "391", "reasoning_content": tok.decode(produced)}
                nxt = svc._as_written(svc.render_ids(*openai_to_messages({"messages": [u1, a1, u2]})))
                self.assertEqual(nxt[:len(ids) + len(written)], ids + written)
        # leading whitespace: the next render writes the reasoning from its first non-blank character - the bare close
        self.assertIsNone(svc._closed_as_rendered(ids, tok.encode("\n  Hm."), think_end))

    def test_the_history_keeps_the_models_own_ids(self):
        """An answer written in other ids than its text tokenizes to (one character a token) goes back in those ids:
        the next prompt starts with everything the engine holds.  An edited history, another template: not."""
        svc, tok = self.svc()
        im_end = tok.encode("<|im_end|>", parse_special=True)[0]
        u1 = {"role": "user", "content": "Read the file and tell me."}
        u2 = {"role": "user", "content": "And then?"}
        ids = svc.render_ids(*openai_to_messages({"messages": [u1]}))
        reasoning, answer = "I should read the file first.\n", "The quick brown fox."
        own = [i for ch in reasoning for i in tok.encode(ch)] + tok.encode("</think>", parse_special=True) + \
            tok.encode("\n\n") + [i for ch in answer for i in tok.encode(ch)] + [im_end]
        a1 = {"role": "assistant", "content": answer, "reasoning_content": reasoning}
        canon = svc.render_ids(*openai_to_messages({"messages": [u1, a1, u2]}))
        self.assertNotEqual(canon[:len(ids) + len(own)], ids + own)   # (the test's premise: other ids)
        svc._note_written(ids, own, None)
        nxt = svc._as_written(canon)
        self.assertEqual(nxt[:len(ids) + len(own)], ids + own)
        self.assertEqual(tok.decode(nxt), tok.decode(canon))          # the same text
        edited = svc.render_ids(*openai_to_messages({"messages": [{"role": "user", "content": "Read it."}, a1, u2]}))
        self.assertEqual(svc._as_written(edited), edited)
        other = {**a1, "content": answer + " Then a dog."}
        changed = svc.render_ids(*openai_to_messages({"messages": [u1, other, u2]}))
        self.assertEqual(svc._as_written(changed), changed)
        glm, gtok = self.svc("glm5")
        glm._note_written(ids, own, None)
        self.assertFalse(getattr(glm, "_written", None))

    def test_other_templates_and_prompts_keep_the_bare_close(self):
        svc, tok = self.svc("qwen")
        think_end = tok.encode("</think>", parse_special=True)[0]
        produced = tok.encode("Done.")
        self.assertIsNone(svc._closed_as_rendered(tok.encode("Hello"), produced, think_end))   # (no open <think>\n)
        for model in ("glm5", "deepseek", "minimax"):
            with self.subTest(model):
                svc, tok = self.svc(model)
                end = tok.encode("</think>", parse_special=True)[0]
                ids = tok.encode("x<think>\n", parse_special=True)
                self.assertIsNone(svc._closed_as_rendered(ids, tok.encode("Done."), end))


if __name__ == "__main__":
    unittest.main()
