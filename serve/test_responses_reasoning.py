"""Reasoning projections for Responses clients, including Hermes request defaults."""
import json
import unittest

from serve.test_responses import CALL, TOOLS, RecordingEngine, ResponseCase, openai
from serve.responses import normalize_request

REASONING = {"effort": "xhigh", "summary": "auto"}
INCLUDE = ["reasoning.encrypted_content"]
THOUGHT = "Check the result carefully."


class ReasoningContracts(ResponseCase):
    def test_summary_modes_keep_reasoning_effort_and_usage(self):
        usage = None
        for summary in (None, "auto", "concise", "detailed"):
            for streamed in (False, True):
                with self.subTest(summary=summary, stream=streamed):
                    self.script(THOUGHT + "</think>\n\n42")
                    reasoning = {"effort": "xhigh", "summary": summary}
                    if streamed:
                        out = self.stream(reasoning=reasoning, include=INCLUDE)[-1]["response"]
                    else:
                        status, out = self.post(reasoning=reasoning, include=INCLUDE)
                        self.assertEqual(status, 200, out)
                    self.assertEqual(out["status"], "completed")
                    self.assertEqual(out["reasoning"], reasoning)
                    self.assertEqual(self.engine.sampling["reasoning"]["effort"], "xhigh")
                    self.assertEqual(out["output"][1]["content"][0]["text"], "42")
                    item = out["output"][0]
                    channel = "content" if summary is None else "summary"
                    self.assertEqual(item[channel][0]["text"], THOUGHT)
                    self.assertEqual(item["summary" if summary is None else "content"], [])
                    self.assertNotIn("encrypted_content", item)
                    count = out["usage"]["output_tokens_details"]["reasoning_tokens"]
                    self.assertGreater(count, 0)
                    if usage is None:
                        usage = out["usage"]["output_tokens"]
                    self.assertEqual(out["usage"]["output_tokens"], usage)

    def test_summary_stream_lifecycle_and_snapshots_are_consistent(self):
        self.script(THOUGHT + "</think>\n\n42")
        events = self.stream(reasoning=REASONING, include=INCLUDE)
        summary = [e for e in events if e["type"].startswith("response.reasoning_summary_")]
        self.assertEqual(summary[0]["type"], "response.reasoning_summary_part.added")
        self.assertEqual(summary[0]["part"], {"type": "summary_text", "text": ""})
        self.assertEqual([e["type"] for e in summary[-2:]], ["response.reasoning_summary_text.done",
                                                               "response.reasoning_summary_part.done"])
        self.assertEqual("".join(e["delta"] for e in summary if "delta" in e), THOUGHT)
        self.assertEqual(summary[-2]["text"], THOUGHT)
        self.assertEqual(summary[-1]["part"], {"type": "summary_text", "text": THOUGHT})
        item = events[-1]["response"]["output"][0]
        for event in summary:
            self.assertEqual((event["item_id"], event["output_index"], event["summary_index"]), (item["id"], 0, 0))
            self.assertNotIn("content_index", event)
        self.assertFalse(any(e["type"].startswith("response.reasoning_text.") for e in events))
        added = next(e for e in events if e["type"] == "response.output_item.added")
        self.assertEqual(added["item"]["summary"], [])
        done = next(e for e in events if e["type"] == "response.output_item.done")
        self.assertEqual(done["item"], item)

    def test_projection_with_default_effort_and_disabled_reasoning(self):
        self.script(THOUGHT + "</think>\n\n42")
        with self.request({"input": "hi", "include": INCLUDE, "reasoning": {"summary": "auto"}}) as response:
            self.assertEqual(response.status, 200)
            out = json.load(response)
        self.assertEqual(out["output"][0]["summary"][0]["text"], THOUGHT)
        self.script("42")
        status, out = self.post(reasoning={"effort": "none", "summary": "auto"}, include=INCLUDE)
        self.assertEqual(status, 200, out)
        self.assertEqual([item["type"] for item in out["output"]], ["message"])
        self.assertEqual(out["usage"]["output_tokens_details"]["reasoning_tokens"], 0)
        with self.request(path="/v1/status") as response:
            capabilities = json.load(response)["responses"]["reasoning"]
        self.assertEqual(capabilities, {"summary": "model-reasoning", "encrypted_content": False})

    def test_tool_continuation_replays_summary_once(self):
        for stored in (False, True):
            with self.subTest(stored=stored):
                self.script([THOUGHT + "</think>\n\n" + CALL, "Verify the value.</think>\n\n73"])
                first = self.stream(reasoning=REASONING, include=INCLUDE, tools=TOOLS, store=stored)[-1]["response"]
                self.assertEqual(first["status"], "completed", first.get("error"))
                self.assertEqual([item["type"] for item in first["output"]], ["reasoning", "function_call"])
                call = first["output"][1]
                result = {"type": "function_call_output", "call_id": call["call_id"], "output": "73"}
                fields = {"previous_response_id": first["id"], "input": [result]} if stored else {
                    "input": [{"role": "user", "content": "hi"}, *first["output"], result]}
                status, out = self.post(**fields, reasoning=REASONING, include=INCLUDE, tools=TOOLS)
                self.assertEqual(status, 200, out)
                self.assertEqual(out["output"][0]["summary"][0]["text"], "Verify the value.")
                self.assertEqual(out["output"][1]["content"][0]["text"], "73")
                self.assertEqual(self.tok.decode(self.engine.last_prompt).count(THOUGHT), 1)

    def test_incomplete_and_failed_reasoning_close_the_summary(self):
        self.script("abcdefgh</think>\n\n42")
        events = self.stream(reasoning=REASONING, max_output_tokens=3)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        item = events[-1]["response"]["output"][0]
        self.assertEqual((item["status"], item["summary"][0]["text"]), ("incomplete", "abc"))
        self.assertEqual(events[-3]["type"], "response.reasoning_summary_part.done")

        class Failing(RecordingEngine):
            def generate(self, *args, **kwargs):
                yield from super().generate(*args, **kwargs)
                raise ValueError("test reasoning failure")

        self.engine = self.svc.engine = Failing(self.tok, "partial")
        self.engine.script = self.tok.encode("partial")
        events = self.stream(reasoning=REASONING)
        self.assertEqual(events[-1]["type"], "response.failed")
        item = events[-1]["response"]["output"][0]
        self.assertEqual((item["status"], item["summary"][0]["text"]), ("incomplete", "partial"))
        self.assertEqual(events[-3]["type"], "response.reasoning_summary_part.done")

    def test_invalid_options_and_encrypted_input_fail_before_generation(self):
        cases = [{"include": value} for value in ("reasoning.encrypted_content", {}, [None], [{}], ["unknown"])]
        cases += [{"reasoning": {"effort": "xhigh", "summary": value}} for value in ("", "none", True, [], {})]
        cases.append({"input": [{"type": "reasoning", "encrypted_content": "opaque-ciphertext", "summary": []},
                                {"role": "user", "content": "hi"}]})
        for fields in cases:
            with self.subTest(fields=fields):
                self.assertEqual(self.post(**fields)[0], 400)
        self.assertEqual(self.engine.last_prompt, [])
        normalized = normalize_request({"input": "hi", "reasoning": REASONING,
                                        "include": INCLUDE + ["message.output_text.logprobs"], "top_logprobs": 2})
        self.assertEqual((normalized["logprobs"], normalized["top_logprobs"]), (True, 2))
        self.assertEqual(normalized["reasoning"], REASONING)

    @unittest.skipUnless(openai, "optional OpenAI SDK is not installed")
    def test_sdk_accumulates_summary_stream_and_json(self):
        self.script(THOUGHT + "</think>\n\n42")
        with openai.OpenAI(base_url=self.base + "/v1", api_key="local", max_retries=0,
                           _strict_response_validation=True) as client:
            out = client.responses.create(model="m", input="hi", reasoning=REASONING, include=INCLUDE)
            self.assertEqual(out.output[0].summary[0].text, THOUGHT)
            with client.responses.stream(model="m", input="hi", reasoning=REASONING, include=INCLUDE) as stream:
                list(stream)
                final = stream.get_final_response()
            self.assertEqual(final.output[0].summary[0].text, THOUGHT)
            self.assertEqual(final.output_text, "42")


if __name__ == "__main__":
    unittest.main()
