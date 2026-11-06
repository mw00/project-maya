"""DeepSeek's real template and streamed DSML through the server's tool-loop blocker.

No model or GPU: the engine emits fixed token sequences.
Run with python -m unittest serve.test_deepseek_tool_block -v.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from serve import loop_guard as LG
from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import Service
from serve.test_loop_guard import DeepTok, DS_EOS, agent

ROOT = Path(__file__).resolve().parents[1]
CLOSE = "</｜DSML｜ calls>"


def call(command, name="bash"):
    return (f'<｜DSML｜ calls>\n<｜DSML｜ invoke name="{name}">\n'
            f'<｜DSML｜ parameter name="command" string="true">{command}</｜DSML｜ parameter>\n'
            f'</｜DSML｜ invoke>\n{CLOSE}')


class DsmlEngine:
    def __init__(self, tok, seqs):
        self.tok, self.seqs, self.prompts, self.max_context = tok, seqs, [], 1 << 20

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        seq = self.seqs[min(len(self.prompts), len(self.seqs)) - 1]
        for token in seq[:max_new]:
            if cancel.is_set():
                return
            yield token


class DeepSeekToolBlock(unittest.TestCase):
    command = "read-only fixture | Select-Object -Skip 40"

    def answer(self, *scripts, thinking=False, rounds=5, guard=True, command=None, name="bash",
               max_tokens=3000):
        command = self.command if command is None else command
        tok = DeepTok()
        # One byte per ordinary token also splits every DSML tag and UTF-8 character.
        seqs = [tok.encode(("</think>\n" if thinking else "") + script + DS_EOS,
                           parse_special=True) for script in scripts]
        eng = DsmlEngine(tok, seqs)
        svc = Service(eng, tok, ChatTemplate(ROOT / "tools/ds4/chat_template.jinja"))
        req = agent(*[(command, "(no output)")] * rounds)
        req["tools"][0]["function"]["name"] = name
        for message in req["messages"]:
            for tc in message.get("tool_calls", []):
                tc["function"]["name"] = name
        messages, tools, kwargs = openai_to_messages(req)
        kwargs.update(reasoning_effort="medium" if thinking else "none", enable_thinking=thinking)
        ids, actual_thinking, max_new = svc.prepare(messages, tools, kwargs, max_tokens, guard=guard)
        self.assertEqual(bool(actual_thinking), thinking)
        events, done = [], None
        for kind, value in svc.run(ids, actual_thinking, tools, max_new,
                                   {"_blocked": kwargs.get("_blocked"), "loop_guard": guard},
                                   threading.Event()):
            if kind == "event":
                events.append(value)
            elif kind == "done":
                done = value
        return svc, eng, kwargs, events, done

    def test_blocked_call_is_withheld_and_continuation_has_valid_boundaries(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                svc, eng, kwargs, events, done = self.answer(
                    call(self.command), call("read a different range"), thinking=thinking)
                key = ("bash", LG.args_key({"command": self.command}))
                self.assertIn(key, kwargs["_blocked"])
                calls = [e.call.arguments["command"] for e in events if e.kind == "tool_call"]
                self.assertEqual(calls, ["read a different range"])
                self.assertEqual(sum(e.kind == "tool_start" for e in events), 1)
                self.assertEqual(svc.loop_guard_counts["tool_blocked"], 1)
                self.assertEqual(len(eng.prompts), 2)
                resumed = svc.tok.decode(eng.prompts[1][len(eng.prompts[0]):])
                # The generated DSML block is preserved once, followed by its turn end and tool result.
                self.assertEqual(resumed.count(CLOSE), 1)
                self.assertIn(CLOSE + DS_EOS, resumed)
                self.assertIn(LG.block_result(LG.ToolRepeat("bash", 5)), resumed)
                self.assertIn(LG.block_note(LG.ToolRepeat("bash", 5)), resumed)
                self.assertTrue(resumed.endswith("<think>" + LG.block_steer(
                    LG.ToolRepeat("bash", 5), key[1], 1)) if thinking else resumed.endswith("</think>"))
                self.assertEqual(done["finish"], "stop")

    def test_legacy_marker_in_an_argument_does_not_choose_the_wrong_boundary(self):
        command = "read literal </tool_call> from fixture"
        svc, eng, _, events, _ = self.answer(call(command), call("next"), command=command)
        self.assertEqual([e.call.arguments["command"] for e in events if e.kind == "tool_call"], ["next"])
        resumed = svc.tok.decode(eng.prompts[1][len(eng.prompts[0]):])
        self.assertEqual(resumed.count(CLOSE), 1)
        self.assertIn(CLOSE + DS_EOS, resumed)

    def test_namespaced_call_is_blocked_by_its_full_name(self):
        svc, _, _, events, _ = self.answer(call(self.command, "functions::bash"),
                                           call("next", "functions::bash"), name="functions::bash")
        self.assertEqual(svc.loop_guard_counts["tool_blocked"], 1)
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls],
                         [("functions::bash", {"command": "next"})])

    def test_under_threshold_changed_arguments_and_disabled_guard_pass_through(self):
        cases = [
            ({"rounds": 4}, self.command),
            ({}, "read a different range"),
            ({"guard": False}, self.command),
        ]
        for options, command in cases:
            with self.subTest(options=options, command=command):
                svc, eng, _, events, _ = self.answer(call(command), **options)
                self.assertEqual([e.call.arguments["command"] for e in events if e.kind == "tool_call"],
                                 [command])
                self.assertEqual(svc.loop_guard_counts["tool_blocked"], 0)
                self.assertEqual(len(eng.prompts), 1)

    def test_multi_call_and_trailing_content_keep_existing_guard_policy(self):
        # The current guard intercepts a single complete call only; it must not drop part of a batch.
        batch = call("next").removesuffix(CLOSE) + call(self.command).split("\n", 1)[1]
        for script in (batch, call(self.command) + "\nA finding after the call."):
            with self.subTest(script=script):
                svc, eng, _, events, _ = self.answer(script)
                calls = [e.call.arguments["command"] for e in events if e.kind == "tool_call"]
                self.assertIn(self.command, calls)
                self.assertEqual(len(calls), 2 if script == batch else 1)
                self.assertEqual(svc.loop_guard_counts["tool_blocked"], 0)
                self.assertEqual(len(eng.prompts), 1)

    def test_repeated_block_attempts_stay_within_output_budget(self):
        svc, eng, _, events, done = self.answer(call(self.command), max_tokens=650)
        self.assertGreater(svc.loop_guard_counts["tool_blocked"], 0)
        self.assertLessEqual(done["completion_tokens"], 650)
        self.assertLess(len(eng.prompts), 8)
        # A budget ending inside a held tool call must not become a complete executable call.
        self.assertFalse(any(e.kind == "tool_call" and e.complete for e in events))


if __name__ == "__main__":
    unittest.main()
