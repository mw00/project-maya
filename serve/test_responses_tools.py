"""Responses tool contracts over both model tool syntaxes and the real HTTP/service pipeline."""
import json
import unittest

from serve.test_responses import CALL, TOOLS, ResponseCase
from serve.frontend import OutputParser

QWEN = '<tool_call><function=lookup><parameter=key>alpha</parameter></function></tool_call>'


class ToolContracts(ResponseCase):
    def test_single_tool_stops_at_a_complete_call_and_keeps_real_usage(self):
        for script in (CALL, QWEN):
            with self.subTest(script=script):
                text = script + script.replace("alpha", "beta") + "unwanted tail"
                self.script(text)
                events = self.stream(tools=TOOLS, parallel_tool_calls=False)
                out = events[-1]["response"]
                self.assertEqual(out["status"], "completed")
                self.assertFalse(out["parallel_tool_calls"])
                self.assertEqual(len(out["output"]), 1)
                self.assertEqual(json.loads(out["output"][0]["arguments"]), {"key": "alpha"})
                self.assertEqual(out["usage"]["output_tokens"], len(self.tok.encode(script, parse_special=True)))
                self.assertEqual(self.svc.history[-1]["finish"], "stop")
                self.assertFalse(self.engine.cancel.is_set())
                self.assertTrue(self.engine.closed)
                self.assertEqual(self.svc.active, 0)
                self.script("Next request works")
                self.assertEqual(self.post()[0], 200)

    def test_truncated_calls_never_publish_completed_arguments(self):
        for text in (CALL, QWEN):
            # Cut inside the string, its closing tag, the outer closing tag, or the function name.
            cuts = [text.index("alpha") + 2, text.index("alpha") + 5, len(text) - 2, 13]
            for cut in cuts:
                with self.subTest(text=text, cut=cut):
                    self.script(text)
                    events = self.stream(tools=TOOLS, max_output_tokens=cut)
                    self.assertEqual(events[-1]["type"], "response.incomplete")
                    self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])
                    calls = [item for item in events[-1]["response"]["output"] if item["type"] == "function_call"]
                    self.assertTrue(all(item["status"] == "incomplete" for item in calls))

    def test_early_end_of_turn_inside_a_call_fails(self):
        for text in (CALL, QWEN):
            self.script(text[:text.index("alpha") + 2])
            events = self.stream(tools=TOOLS)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_explicit_outer_close_does_not_fix_missing_parameter_close(self):
        for text in (CALL.replace("</arg_value>", ""), QWEN.replace("</parameter>", "")):
            self.script(text)
            events = self.stream(tools=TOOLS)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_trailing_garbage_inside_a_call_is_not_a_complete_call(self):
        for text in (CALL, QWEN):
            self.script(text.replace("</tool_call>", "unexpected</tool_call>"))
            events = self.stream(tools=TOOLS)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_no_executable_call_is_published_after_a_malformed_call(self):
        self.script(CALL.replace("</arg_value>", "") + CALL)
        events = self.stream(tools=TOOLS)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_non_finite_json_arguments_are_rejected(self):
        tools = [{**TOOLS[0], "parameters": {"type": "object", "properties": {"key": {"type": "number"}}}}]
        self.script(CALL.replace("alpha", "NaN"))
        events = self.stream(tools=tools)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_single_tool_parser_stops_when_multiple_calls_arrive_in_one_token(self):
        for text in (CALL, QWEN):
            parser = OutputParser(thinking=False, tools=[TOOLS[0]], stream_tools=True,
                                  strict_tools=True, single_tool=True)
            events = parser.feed(text + text)
            self.assertEqual(len([e for e in events if e.kind == "tool_call"]), 1)
            self.assertEqual(parser.feed(text) + parser.finish(), [])

    def test_tool_selection_violations_fail_before_an_executable_item(self):
        cases = [(CALL, {"tools": TOOLS, "tool_choice": "none"}),
                 (CALL.replace("lookup", "undeclared"), {"tools": TOOLS})]
        for script, fields in cases:
            self.script(script)
            events = self.stream(**fields)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])
            self.assertNotIn("response.output_item.added", [e["type"] for e in events])

    def test_required_without_a_call_fails_but_budget_exhaustion_is_incomplete(self):
        # Forced tool prefill is tested by existing tests. A generated malformed/incomplete
        # continuation must not fulfill that requirement merely because its name was prefixed.
        self.script("not a valid function body")
        events = self.stream(tools=TOOLS, tool_choice="required")
        self.assertEqual(events[-1]["type"], "response.failed")
        self.script("a" * 100)
        events = self.stream(tools=TOOLS, tool_choice="required", max_output_tokens=3)
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_codex_hints_and_reasoning_disabled_projection(self):
        status, out = self.post(prompt_cache_key="test-session", client_metadata={"session_id": "fixture"},
                                include=["reasoning.encrypted_content"])
        self.assertEqual(status, 200, out)
        self.assertEqual(out["prompt_cache_key"], "test-session")
        self.assertNotIn("client_metadata", out)
        for fields in ({"prompt_cache_key": {}}, {"prompt_cache_key": "a" * 257},
                       {"client_metadata": []}, {"client_metadata": {"session": 1}}):
            self.assertEqual(self.post(**fields)[0], 400)


if __name__ == "__main__":
    unittest.main()
