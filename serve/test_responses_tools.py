"""Responses tool contracts over both model tool syntaxes and the real HTTP/service pipeline."""
import json
import unittest

from serve.test_responses import CALL, TOOLS, ROOT, ResponseCase
from serve.frontend import ChatTemplate, OutputParser

QWEN = '<tool_call><function=lookup><parameter=key>alpha</parameter></function></tool_call>'
DSML = ('<\uff5cDSML\uff5c calls>\n<\uff5cDSML\uff5c invoke name="lookup">\n'
        '<\uff5cDSML\uff5c parameter name="key" string="true">alpha</\uff5cDSML\uff5c parameter>\n'
        '</\uff5cDSML\uff5c invoke>\n</\uff5cDSML\uff5c calls>')


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


class DeepSeekToolContracts(ResponseCase):
    def setUp(self):
        super().setUp()
        self.svc.template = ChatTemplate(ROOT / "tools/ds4/chat_template.jinja")
        self.assertTrue(self.svc.template.deepseek)

    def test_function_call_round_trip_whole_and_streamed(self):
        for streamed in (False, True):
            with self.subTest(stream=streamed):
                self.script([DSML, "The value is 73"])
                if streamed:
                    events = self.stream(tools=TOOLS)
                    out = events[-1]["response"]
                    args = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
                    self.assertEqual(json.loads(args), {"key": "alpha"})
                else:
                    status, out = self.post(tools=TOOLS)
                    self.assertEqual(status, 200, out)
                self.assertEqual(out["status"], "completed")
                call = out["output"][0]
                self.assertEqual((call["name"], json.loads(call["arguments"])), ("lookup", {"key": "alpha"}))
                history = [{"role": "user", "content": "look up alpha"}, *out["output"],
                           {"type": "function_call_output", "call_id": call["call_id"], "output": "73"}]
                status, second = self.post(input=history, tools=TOOLS)
                self.assertEqual(status, 200, second)
                self.assertEqual(second["output"][0]["content"][0]["text"], "The value is 73")
                self.assertIn(DSML, self.tok.decode(self.engine.last_prompt))
                self.assertIn("<tool_result>73</tool_result>", self.tok.decode(self.engine.last_prompt))

    def test_required_and_named_tools_use_dsml_prefill(self):
        for choice, end in (("required", DSML.index("\n") + 1),
                            ({"type": "function", "name": "lookup"}, DSML.index("\n", DSML.index("\n") + 1) + 1)):
            with self.subTest(choice=choice):
                self.script(DSML[end:])
                status, out = self.post(tools=TOOLS, tool_choice=choice)
                self.assertEqual(status, 200, out)
                self.assertEqual(out["status"], "completed")
                self.assertEqual(json.loads(out["output"][0]["arguments"]), {"key": "alpha"})
                self.assertTrue(self.tok.decode(self.engine.last_prompt).endswith("\n\n" + DSML[:end]))

    def test_truncation_never_publishes_completed_arguments(self):
        for cut in (DSML.index("alpha") + 2, DSML.index("</\uff5cDSML\uff5c parameter>") + 3,
                    DSML.index("</\uff5cDSML\uff5c invoke>") + 3):
            with self.subTest(cut=cut):
                self.script(DSML)
                events = self.stream(tools=TOOLS, max_output_tokens=len(self.tok.encode(DSML[:cut])))
                self.assertEqual(events[-1]["type"], "response.incomplete")
                self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])
        self.script(DSML[:DSML.index("alpha") + 2])
        self.assertEqual(self.stream(tools=TOOLS)[-1]["type"], "response.failed")

    def test_single_tool_stops_at_the_first_invoke(self):
        self.script(DSML + DSML.replace("alpha", "beta") + "unwanted tail")
        events = self.stream(tools=TOOLS, parallel_tool_calls=False)
        out = events[-1]["response"]
        self.assertEqual(out["status"], "completed")
        self.assertEqual(len(out["output"]), 1)
        self.assertEqual(json.loads(out["output"][0]["arguments"]), {"key": "alpha"})
        first = DSML[:DSML.index("</\uff5cDSML\uff5c invoke>") + len("</\uff5cDSML\uff5c invoke>")]
        self.assertEqual(out["usage"]["output_tokens"], len(self.tok.encode(first)))
        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, dsml=True,
                              strict_tools=True, single_tool=True)
        self.assertEqual(len([e for e in parser.feed(DSML + DSML) if e.kind == "tool_call"]), 1)
        self.assertEqual(parser.feed(DSML) + parser.finish(), [])


if __name__ == "__main__":
    unittest.main()
