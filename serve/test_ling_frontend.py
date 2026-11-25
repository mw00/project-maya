"""serve/test_ling_frontend.py - Ling-3.2-air frontend, chat template golden cases, thinking levels,
OpenAI / Anthropic / Responses / Model API XML tool parsing, sampling defaults (incl. --model-dir
and model switch), bad thinking rejection before load, and server restore points.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import maya_quants  # noqa: E402
from serve import server  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, effort_kwargs  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, checked_config_effort, ling_restore_points  # noqa: E402

TPL_PATH = ROOT / "tools" / "ling" / "chat_template.jinja"
GOLDEN_PATH = ROOT / "tools" / "ling" / "chat_golden.json"


class LingTok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [
        "<role>", "</role>", "<|role_end|>", "<think>", "</think>",
        "<tool_call>", "</tool_call>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
    ]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens["<|role_end|>"]}

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            s = self.SPECIALS[t - 256] if t >= 256 else None
            raw += s.encode("utf-8") if s is not None else bytes([t])
        return raw.decode("utf-8", errors=errors)


class TestLingFrontend(unittest.TestCase):
    def setUp(self):
        self.tpl = ChatTemplate(TPL_PATH)
        self.tok = LingTok()

    def test_golden_cases_byte_exact(self):
        self.assertTrue(self.tpl.ling)
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(golden), 7)
        for c in golden:
            with self.subTest(case=c["name"]):
                got = self.tpl.render(c["messages"], tools=c.get("tools"), add_generation_prompt=True, **c.get("kwargs", {}))
                self.assertEqual(got, c["rendered"])

    def test_all_thinking_levels_and_off_is_really_off(self):
        self.assertEqual(self.tpl.thinking_levels(), ["none", "high"])
        for lv in ("low", "medium", "high", "xhigh", "max", "auto"):
            with self.subTest(level=lv):
                self.assertEqual(self.tpl.effort_shown(lv), "high")
                kw = effort_kwargs(lv)
                rendered = self.tpl.render([{"role": "user", "content": "Hi"}], **kw)
                self.assertIn("detailed thinking on", rendered)
                self.assertTrue(rendered.endswith("\n<think>"))
                self.assertTrue(self.tpl.opens_thinking(rendered))

        for off_lv in ("none", "off", "minimal"):
            with self.subTest(off=off_lv):
                kw = effort_kwargs(off_lv)
                rendered = self.tpl.render([{"role": "user", "content": "Hi"}], **kw)
                self.assertIn("detailed thinking off", rendered)
                self.assertTrue(rendered.endswith("\n<think></think>"))
                self.assertFalse(self.tpl.opens_thinking(rendered))

    def test_bad_thinking_rejected_before_load(self):
        cfg = {"exe": str(ROOT / "build" / "maya-ling"), "reasoning_effort": "turbo_bogus"}
        with self.assertRaises(SystemExit):
            checked_config_effort(cfg, self.tpl, maya=True)
        # Valid default filled when omitted
        cfg2 = checked_config_effort({"exe": str(ROOT / "build" / "maya-ling")}, self.tpl, maya=True)
        self.assertEqual(cfg2["reasoning_effort"], "high")

    def test_model_defaults_apply_to_any_ling_gguf_and_model_dir(self):
        # Bare config (e.g., --model-dir GGUF with no sampling or installer block)
        bare = {"exe": "/any/path/maya-ling", "args": ["--serve", "--gguf", "/custom/Any-Ling.gguf"]}
        samp, eff = maya_quants.model_defaults(bare)
        self.assertEqual(samp, {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(eff, "high")
        self.assertEqual(maya_quants.family_of(bare), "ling")

    def test_tool_calls_and_reasoning_parser_streaming_and_whole(self):
        tools = [{
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["cmd"],
            },
        }]
        sample = (
            "Let me check files.</think>\n"
            "<tool_call>shell\n"
            "<arg_key>cmd</arg_key>\n"
            "<arg_value>ls -la</arg_value>\n"
            "<arg_key>timeout</arg_key>\n"
            "<arg_value>30</arg_value>\n"
            "</tool_call>"
        )
        # Character-by-character streaming (exercises OpenAI / Anthropic / Responses / Model API parser)
        p = OutputParser(thinking=True, tools=tools, stream_tools=True)
        evs = []
        for ch in sample:
            evs.extend(p.feed(ch))
        evs.extend(p.finish())
        reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual(reasoning, "Let me check files.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "shell")
        self.assertEqual(calls[0].arguments, {"cmd": "ls -la", "timeout": 30})

    def test_ling_restore_points_and_gen_keys(self):
        prompt = self.tpl.render([
            {"role": "system", "content": "S" * 120},
            {"role": "user", "content": "Turn 1 " + "a" * 80},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2 " + "b" * 100},
        ], enable_thinking=True)
        ids = self.tok.encode(prompt)
        hseq = self.tok.encode("<role>HUMAN</role>")
        pts = ling_restore_points(ids, hseq, tail=32, least=16)
        self.assertIn("shared", pts)
        self.assertIn("marks", pts)
        self.assertEqual(len(pts["marks"]), 1)
        self.assertGreater(pts["marks"][0], pts["shared"])
        keys = StrataEngine.restore_keys({"_restore": pts})
        self.assertIn(f"shared={pts['shared']}", keys)
        self.assertIn(f"marks={pts['marks'][0]}", keys)


if __name__ == "__main__":
    unittest.main()
