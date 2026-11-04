"""Responses HTTP workflows over the real service and scripted engine; no GPU or API key needed.

    python -m unittest serve.test_responses -v
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from serve.frontend import ChatTemplate, openai_to_messages
from serve.responses import ResponseStore, normalize_request
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"type": "function", "name": "lookup", "description": "Look up a value",
          "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}, "strict": False}]
CALL = '<tool_call>lookup<arg_key>key</arg_key><arg_value>alpha</arg_value></tool_call>'

try:
    import openai
    from pydantic import TypeAdapter
    from openai.types.responses import Response, ResponseStreamEvent
except ImportError:
    openai = None


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.max_new, self.sampling, self.cancel = max_new, sampling, cancel
        self.image_bytes = Path(embeddings).read_bytes() if embeddings else None
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            self.closed = True


class ResponseCase(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = RecordingEngine(self.tok, "Hello", max_context=8192)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def script(self, text, **kwargs):
        self.engine = RecordingEngine(self.tok, text, max_context=8192, **kwargs)
        self.svc.engine = self.engine

    def request(self, body=None, path="/v1/responses", headers=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json", **(headers or {})}, method=method)
        try:
            return urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as exc:
            return exc

    def post(self, **fields):
        with self.request({"input": "hi", "reasoning": {"effort": "none"}, **fields}) as response:
            return response.status, json.load(response)

    def stream(self, **fields):
        with self.request({"input": "hi", "reasoning": {"effort": "none"}, "stream": True, **fields}) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            raw = response.read().decode()
        events = []
        for block in raw.split("\n\n"):
            if not block or block.startswith(":"):
                continue
            name, payload = block.split("\n", 1)
            data = json.loads(payload.removeprefix("data: "))
            self.assertEqual(name, "event: " + data["type"])
            events.append(data)
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        self.assertNotIn("[DONE]", raw)
        if openai:
            adapter = TypeAdapter(ResponseStreamEvent)
            for event in events:
                adapter.validate_python(event)
        return events


class Responses(ResponseCase):
    def test_text_and_instructions(self):
        status, out = self.post(instructions="Answer briefly", metadata={"job": "test"}, temperature=0)
        self.assertEqual(status, 200)
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["output"][0]["content"][0]["text"], "Hello")
        self.assertFalse(out["store"])
        self.assertEqual(out["metadata"], {"job": "test"})
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("Answer briefly", prompt)
        self.assertIn("hi", prompt)
        usage = out["usage"]
        self.assertEqual(usage["input_tokens"], len(self.engine.last_prompt))
        self.assertEqual(usage["total_tokens"], usage["input_tokens"] + usage["output_tokens"])
        self.assertEqual(self.engine.sampling["temperature"], 0)
        self.assertEqual(out["temperature"], 0)
        if openai:
            Response.model_validate(out)

    def test_stream_lifecycle_and_snapshots(self):
        events = self.stream()
        self.assertEqual([e["type"] for e in events[:2]], ["response.created", "response.in_progress"])
        self.assertEqual(events[0]["response"]["output"], [])
        added = next(e for e in events if e["type"] == "response.output_item.added")
        self.assertEqual(added["item"]["content"], [])
        self.assertEqual(added["item"]["status"], "in_progress")
        text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        self.assertEqual(text, "Hello")
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["output"][0]["id"], added["item"]["id"])
        self.assertEqual(events[-2]["type"], "response.output_item.done")

    def test_reasoning_and_output_can_be_replayed(self):
        self.script(["check the sum</think>\n\n42", "Correct"])
        events = self.stream(reasoning={"effort": "low"})
        out = events[-1]["response"]
        self.assertEqual([item["type"] for item in out["output"]], ["reasoning", "message"])
        reasoning = "".join(e["delta"] for e in events if e["type"] == "response.reasoning_text.delta")
        self.assertEqual(reasoning, "check the sum")
        self.assertGreater(out["usage"]["output_tokens_details"]["reasoning_tokens"], 0)
        history = [{"role": "user", "content": "What is the sum?"}, *out["output"],
                   {"role": "user", "content": [{"type": "input_text", "text": "Check it"}]}]
        status, second = self.post(input=history)
        self.assertEqual((status, second["output"][0]["content"][0]["text"]), (200, "Correct"))
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("42", prompt)
        self.assertIn("check the sum", prompt)

    def test_function_call_round_trip_whole_and_streamed(self):
        for streamed in (False, True):
            with self.subTest(stream=streamed):
                self.script([CALL, "The value is 73"])
                if streamed:
                    events = self.stream(tools=TOOLS)
                    out = events[-1]["response"]
                    args = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
                    self.assertEqual(json.loads(args), {"key": "alpha"})
                else:
                    status, out = self.post(tools=TOOLS)
                    self.assertEqual(status, 200)
                call = out["output"][0]
                self.assertEqual((call["type"], call["name"], call["status"]), ("function_call", "lookup", "completed"))
                self.assertEqual(json.loads(call["arguments"]), {"key": "alpha"})
                self.assertNotEqual(call["id"], call["call_id"])
                history = [{"role": "user", "content": "look up alpha"}, *out["output"],
                           {"type": "function_call_output", "call_id": call["call_id"], "output": "73"}]
                status, second = self.post(input=history, tools=TOOLS)
                self.assertEqual((status, second["output"][0]["content"][0]["text"]), (200, "The value is 73"))
                self.assertIn("73", self.tok.decode(self.engine.last_prompt))

    def test_multiple_tool_results_follow_call_ids(self):
        second = CALL.replace("alpha", "beta")
        self.script([CALL + second, "Done"])
        events = self.stream(tools=TOOLS)
        calls = events[-1]["response"]["output"]
        self.assertEqual(len(calls), 2)
        history = [{"role": "user", "content": "look up both"}, *calls,
                   {"type": "function_call_output", "call_id": calls[1]["call_id"], "output": "SECOND-RESULT"},
                   {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "FIRST-RESULT"}]
        status, _ = self.post(input=history, tools=TOOLS)
        self.assertEqual(status, 200)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertLess(prompt.index("FIRST-RESULT"), prompt.index("SECOND-RESULT"))

    def test_output_budget_and_context_errors(self):
        self.script("a" * 100)
        status, out = self.post(max_output_tokens=5)
        self.assertEqual((status, out["status"], self.engine.max_new), (200, "incomplete", 5))
        self.assertEqual(out["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(out["output"][0]["status"], "incomplete")
        events = self.stream(max_output_tokens=5)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        status, out = self.post(max_output_tokens=8192)
        self.assertEqual(status, 400)
        self.assertIn("exceeds the context", out["error"]["message"])
        self.assertEqual(out["error"]["code"], "context_length_exceeded")   # (#432)

    def test_named_tool_choice_and_shared_defaults(self):
        self.script('<parameter=key>alpha</parameter>\n</function>\n</tool_call>')
        status, out = self.post(tools=TOOLS, tool_choice={"type": "function", "name": "lookup"})
        # The forced opening is the existing template's <function=lookup>, not a new prompting path.
        self.assertEqual(status, 200)
        self.assertEqual(out["output"][0]["name"], "lookup")
        self.assertIn("<function=lookup>", self.tok.decode(self.engine.last_prompt))
        self.script("Hello")
        self.svc.shared = {"reasoning_effort": "none", "max_tokens": 3}
        with self.request({"input": "hi"}) as response:
            out = json.load(response)
        self.assertEqual((out["status"], self.engine.max_new), ("incomplete", 3))

    def test_invalid_and_unsupported_requests_fail_before_generation(self):
        cases = [
            {"store": "true"}, {"previous_response_id": []}, {"background": True},
            {"tools": [{"type": "web_search"}]}, {"tools": [{**TOOLS[0], "strict": True}]},
            {"tools": {}}, {"input": [{"type": "item_reference", "id": "msg_old"}]},
            {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]},
            {"input": [{"type": "function_call_output", "call_id": "missing", "output": "value"}]},
            {"input": [{"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"}]},
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "url", "detail": "high"}]}]},
            {"text": {"format": {"type": "json_schema"}}}, {"reasoning": {"summary": "auto"}},
            {"include": ["reasoning.encrypted_content"], "reasoning": {"effort": "low"}},
            {"max_output_tokens": 0}, {"max_output_tokens": True}, {"stream": "true"},
            {"truncation": "auto"}, {"conversation": "conv_1"}, {"input": None}, {"input": []},
            {"tool_choice": "required"}, {"tools": TOOLS, "tool_choice": {"type": "function", "name": "missing"}},
            {"temperature": "invalid"}, {"metadata": {"job": 12}},
            {"model": {}},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                status, out = self.post(**fields)
                self.assertEqual(status, 400, out)
                self.assertEqual(out["error"]["type"], "invalid_request_error")
        self.assertEqual(self.engine.last_prompt, [])

    def test_auth_cors_capabilities_and_reload(self):
        self.svc.api_key = "secret"
        self.svc.cors_origins = ["https://example.test"]
        with self.request({"input": "hi"}) as response:
            self.assertEqual(response.status, 401)
        headers = {"Authorization": "Bearer secret", "Origin": "https://example.test"}
        with self.request({"input": "hi", "reasoning": {"effort": "none"}}, headers=headers) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "https://example.test")
        with self.request(path="/v1/status", headers=headers) as response:
            self.assertIn("/v1/responses", json.load(response)["dialects"])
        self.svc.reload = {"state": "running"}
        with self.request({"input": "hi"}, headers=headers) as response:
            self.assertEqual(response.status, 503)
            self.assertEqual(response.headers["Retry-After"], "30")

    def test_responses_capabilities_keep_maya_ds4_diagnostics(self):
        self.engine.info.update(vision="off", gpu_commit_capped_gb=12, pagefile_hint_gb=64)
        with self.request(path="/v1/status") as response:
            out = json.load(response)
        self.assertEqual(out["service"], "maya")
        self.assertIn("/v1/responses", out["dialects"])
        self.assertIn("vision tower did not load", out["vision"]["error"])
        self.assertIn("12 GB", out["memory_note"])
        self.assertIn("64 GB", out["memory_note"])
        self.assertIn("cache", out["responses"])
        self.assertIn("jobs", out["responses"])

    def test_images_reach_prompt_and_engine(self):
        class Vision:
            owed = False

            def active(self):
                return contextlib.nullcontext()

            def encode(self, source):
                self.sources.append(source)
                path = self.dir / f"image-{len(self.sources)}.sve"
                path.write_bytes(bytes([len(self.sources)]))
                return path, 3

        history = [{"role": "user", "content": "Read the file"},
                   {"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"},
                   {"type": "function_call_output", "call_id": "c", "output": [
                       {"type": "input_text", "text": "Result:"},
                       {"type": "input_image", "image_url": "data:image/png;base64,Ymx1ZQ=="}]},
                   {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,cmVk"},
                                                {"type": "input_text", "text": "compare"}]}]
        status, out = self.post(input=history)
        self.assertEqual(status, 400)
        self.assertIn("vision", out["error"]["message"])
        with tempfile.TemporaryDirectory() as directory:
            vision = Vision()
            vision.dir, vision.sources = Path(directory), []
            self.svc.vision = vision
            status, out = self.post(input=history)
            self.assertEqual(status, 200, out)
            self.assertEqual(vision.sources, ["data:image/png;base64,Ymx1ZQ==", "data:image/png;base64,cmVk"])
            self.assertEqual(self.engine.image_bytes, b"\x01\x02")
            self.assertEqual(self.tok.decode(self.engine.last_prompt).count("<|image_pad|>"), 6)
            self.assertFalse(Path(self.engine.last_embeddings).exists())
            status, _ = self.post(input=history, max_output_tokens=8192)
            self.assertEqual(status, 400)
            self.assertFalse(list(Path(directory).glob("req-*.sve")))

    def test_engine_failure_has_a_terminal_error(self):
        class Failing(RecordingEngine):
            def generate(self, *args, **kwargs):
                yield from super().generate(*args, **kwargs)
                raise ValueError("test engine failure")

        self.engine = Failing(self.tok, "partial")
        self.engine.script = self.tok.encode("partial", parse_special=True)
        self.svc.engine = self.engine
        events = self.stream()
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["error"], {"code": "server_error", "message": "test engine failure"})
        self.assertEqual(events[-2]["type"], "response.output_item.done")
        self.assertEqual(events[-1]["response"]["output"][0]["status"], "incomplete")
        self.assertEqual(self.svc.active, 0)
        status, out = self.post()
        self.assertEqual((status, out["status"]), (500, "failed"))
        with patch.object(self.svc, "prepare", side_effect=OSError("test disk error")):
            status, out = self.post()
        self.assertEqual(status, 500)
        self.assertEqual(out["error"]["type"], "server_error")

    def test_disconnect_stops_generation_and_next_request_works(self):
        self.script("a" * 2000, delay_s=0.002)
        response_id = None
        with self.request({"input": "hi", "stream": True, "store": True, "reasoning": {"effort": "none"}}) as response:
            while True:
                line = response.readline()
                self.assertTrue(line, "stream ended before content arrived")
                if line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "response.created":
                        response_id = event["response"]["id"]
                    elif event["type"] == "response.output_text.delta":
                        break
        deadline = time.monotonic() + 5
        while self.svc.active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.svc.active, 0)
        self.assertTrue(self.engine.cancel.is_set())
        self.assertTrue(self.engine.closed)
        self.assertIsNotNone(response_id)
        with self.request(path="/v1/responses/" + response_id) as response:
            self.assertEqual(response.status, 404)
        self.script("Ready")
        status, out = self.post()
        self.assertEqual((status, out["output"][0]["content"][0]["text"]), (200, "Ready"))

    def test_stored_response_retrieve_delete_and_stateless_default(self):
        status, out = self.post(store=True)
        self.assertEqual((status, out["store"]), (200, True))
        path = "/v1/responses/" + out["id"]
        with self.request(path=path) as response:
            self.assertEqual(json.load(response), out)
        with self.request(path=path, method="DELETE") as response:
            self.assertEqual(json.load(response), {"id": out["id"], "object": "response.deleted", "deleted": True})
        for method in ("GET", "DELETE"):
            with self.request(path=path, method=method) as response:
                self.assertEqual(response.status, 404)
        status, _ = self.post(previous_response_id=out["id"])
        self.assertEqual(status, 404)
        for fields in ({}, {"store": False}):
            status, out = self.post(**fields)
            self.assertEqual(status, 200)
            with self.request(path="/v1/responses/" + out["id"]) as response:
                self.assertEqual(response.status, 404)

    def test_previous_response_inherits_items_but_not_instructions_or_settings(self):
        self.script(["First answer", "Second answer", "Third answer"])
        status, first = self.post(input="First question", instructions="ONLY-FIRST-INSTRUCTIONS", store=True, temperature=0)
        self.assertEqual(status, 200)
        status, second = self.post(input="Second question", previous_response_id=first["id"],
                                   instructions="ONLY-SECOND-INSTRUCTIONS", store=True)
        self.assertEqual(status, 200, second)
        self.assertEqual(second["previous_response_id"], first["id"])
        prompt = self.tok.decode(self.engine.last_prompt)
        for text in ("First question", "First answer", "Second question", "ONLY-SECOND-INSTRUCTIONS"):
            self.assertIn(text, prompt)
        self.assertNotIn("ONLY-FIRST-INSTRUCTIONS", prompt)
        self.assertNotIn("temperature", self.engine.sampling)
        with self.request(path="/v1/responses/" + first["id"], method="DELETE") as response:
            self.assertEqual(response.status, 200)
        status, third = self.post(input="Third question", previous_response_id=second["id"])
        self.assertEqual(status, 200, third)
        prompt = self.tok.decode(self.engine.last_prompt)
        for text in ("First answer", "Second answer", "Third question"):
            self.assertIn(text, prompt)
        self.assertNotIn("ONLY-SECOND-INSTRUCTIONS", prompt)

    def test_stored_tool_call_continues_with_only_the_result(self):
        for streamed in (False, True):
            with self.subTest(stream=streamed):
                self.script([CALL, "The value is 73"])
                first = self.stream(tools=TOOLS, store=True)[-1]["response"] if streamed else self.post(tools=TOOLS, store=True)[1]
                call = first["output"][0]
                status, second = self.post(previous_response_id=first["id"], tools=TOOLS, store=True,
                    input=[{"type": "function_call_output", "call_id": call["call_id"], "output": "73"}])
                self.assertEqual((status, second["output"][0]["content"][0]["text"]), (200, "The value is 73"))
                with self.request(path="/v1/responses/" + second["id"] + "/input_items?order=asc") as response:
                    items = json.load(response)["data"]
                self.assertEqual([item["type"] for item in items], ["message", "function_call", "function_call_output"])
                self.assertEqual(items[1]["call_id"], items[2]["call_id"])

    def test_input_item_pagination_and_invalid_options(self):
        history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"},
                   {"role": "user", "content": "second"}]
        status, out = self.post(input=history, store=True)
        self.assertEqual(status, 200)
        path = "/v1/responses/" + out["id"] + "/input_items"
        with self.request(path=path + "?order=asc&limit=2") as response:
            first = json.load(response)
        self.assertEqual([item["content"][0]["text"] for item in first["data"]], ["first", "answer"])
        self.assertTrue(first["has_more"])
        with self.request(path=path + "?order=asc&limit=2&after=" + first["last_id"]) as response:
            last = json.load(response)
        self.assertEqual(last["data"][0]["content"][0]["text"], "second")
        self.assertFalse(last["has_more"])
        with self.request(path=path) as response:
            descending = json.load(response)
        self.assertEqual(descending["data"][0]["content"][0]["text"], "second")
        with self.request(path=path + "?after=" + descending["last_id"]) as response:
            empty = json.load(response)
        self.assertEqual((empty["data"], empty["first_id"], empty["last_id"], empty["has_more"]), ([], None, None, False))
        for suffix in ("?limit=0", "?limit=101", "?limit=abc", "?order=invalid", "?after=missing", "?before=x",
                       "?limit=1&limit=2", "?include=reasoning.encrypted_content"):
            with self.subTest(suffix=suffix), self.request(path=path + suffix) as response:
                self.assertEqual(response.status, 400)
        with self.request(path="/v1/responses/" + out["id"] + "?stream=true") as response:
            self.assertEqual(response.status, 409)

    def test_stored_history_is_isolated_by_api_key(self):
        self.svc.api_key = ["first-key", "second-key"]
        first_headers = {"Authorization": "Bearer first-key"}
        second_headers = {"x-api-key": "second-key"}
        with self.request({"input": "private", "store": True, "reasoning": {"effort": "none"}},
                          headers=first_headers) as response:
            out = json.load(response)
        path = "/v1/responses/" + out["id"]
        for suffix in ("", "/input_items"):
            with self.request(path=path + suffix, headers=second_headers) as response:
                self.assertEqual(response.status, 404)
            with self.request(path=path + suffix) as response:
                self.assertEqual(response.status, 401)
        with self.request(path=path, method="DELETE", headers=second_headers) as response:
            self.assertEqual(response.status, 404)
        with self.request({"input": "continue", "previous_response_id": out["id"]}, headers=second_headers) as response:
            self.assertEqual(response.status, 404)
        with self.request(path=path, headers={"x-api-key": "first-key"}) as response:
            self.assertEqual(response.status, 200)
        self.svc.cors_origins = ["https://example.test"]
        with self.request(path=path, method="OPTIONS", headers={"Origin": "https://example.test",
                          "Access-Control-Request-Method": "DELETE"}) as response:
            self.assertEqual(response.status, 204)
            self.assertIn("DELETE", response.headers["Access-Control-Allow-Methods"])

    def test_storage_limits_expiry_and_stream_commit(self):
        now = [0]
        self.svc.responses = ResponseStore(max_entries=1, ttl=10, clock=lambda: now[0])
        events = self.stream(store=True)
        first = events[-1]["response"]
        with self.request(path="/v1/responses/" + first["id"]) as response:
            self.assertEqual(response.status, 200)
        _, second = self.post(store=True)
        status, _ = self.post(previous_response_id=first["id"])
        self.assertEqual(status, 404)
        now[0] = 10
        with self.request(path="/v1/responses/" + second["id"]) as response:
            self.assertEqual(response.status, 404)
        self.svc.responses = ResponseStore(max_bytes=1)
        events = self.stream(store=True)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertFalse(events[-1]["response"]["store"])
        self.assertNotIn("response.completed", [event["type"] for event in events])
        status, failed = self.post(store=True)
        self.assertEqual((status, failed["status"]), (500, "failed"))
        with self.request(path="/v1/responses/" + failed["id"]) as response:
            self.assertEqual(response.status, 404)

    @unittest.skipUnless(openai, "optional OpenAI SDK is not installed")
    def test_official_sdk_stored_conversation_and_pagination(self):
        self.script([CALL, "73", "Follow-up"])
        with openai.OpenAI(base_url=self.base + "/v1", api_key="local", max_retries=0,
                           _strict_response_validation=True) as client:
            with client.responses.stream(model="m", input="look up alpha", tools=TOOLS,
                                         reasoning={"effort": "none"}, store=True) as stream:
                for _ in stream:
                    pass
                first = stream.get_final_response()
            self.assertEqual(client.responses.retrieve(first.id).output[0].call_id, first.output[0].call_id)
            second = client.responses.create(model="m", previous_response_id=first.id, store=True, tools=TOOLS,
                reasoning={"effort": "none"}, input=[{"type": "function_call_output", "call_id": first.output[0].call_id,
                                                       "output": "73"}])
            self.assertEqual(second.output_text, "73")
            page = client.responses.input_items.list(second.id, order="asc", limit=1)
            items = list(page)
            self.assertEqual([item.type for item in items], ["message", "function_call", "function_call_output"])
            self.assertEqual(len({item.id for item in items}), 3)
            client.responses.delete(first.id)
            with self.assertRaises(openai.NotFoundError):
                client.responses.retrieve(first.id)
            third = client.responses.create(model="m", previous_response_id=second.id,
                                             input="continue", reasoning={"effort": "none"})
            self.assertEqual(third.output_text, "Follow-up")

    @unittest.skipUnless(openai, "optional OpenAI SDK is not installed")
    def test_official_sdk_create_and_stream(self):
        with openai.OpenAI(base_url=self.base + "/v1", api_key="local", max_retries=0,
                           _strict_response_validation=True) as client:
            out = client.responses.create(model="m", input="hi", reasoning={"effort": "none"}, store=False)
            self.assertEqual(out.output_text, "Hello")
            with client.responses.stream(model="m", input="hi", reasoning={"effort": "none"}, store=False) as stream:
                text = "".join(event.delta for event in stream if event.type == "response.output_text.delta")
                out = stream.get_final_response()
            self.assertEqual((text, out.output_text), ("Hello", "Hello"))
            self.script([CALL, "73"])
            with client.responses.stream(model="m", input="look up alpha", tools=TOOLS,
                                         reasoning={"effort": "none"}, store=False) as stream:
                for _ in stream:
                    pass
                call = stream.get_final_response().output[0]
            history = [{"role": "user", "content": "look up alpha"}, call.to_dict(),
                       {"type": "function_call_output", "call_id": call.call_id, "output": "73"}]
            out = client.responses.create(model="m", input=history, tools=TOOLS,
                                          reasoning={"effort": "none"}, store=False)
            self.assertEqual(out.output_text, "73")


class HistoryValidation(unittest.TestCase):
    def test_duplicate_and_mismatched_call_results(self):
        call = {"type": "function_call", "call_id": "a", "name": "lookup", "arguments": "{}"}
        result = {"type": "function_call_output", "call_id": "a", "output": "73"}
        for items in ([call, call], [call, result, result], [call, {**result, "call_id": "b"}],
                      [{**call, "arguments": "[]"}, result], [{**call, "arguments": "bad"}, result]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                normalize_request({"input": items})

    def test_request_is_not_mutated(self):
        request = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}], "tools": TOOLS}
        before = json.dumps(request)
        openai_to_messages(normalize_request(request))
        self.assertEqual(json.dumps(request), before)


class StoreBounds(unittest.TestCase):
    def test_byte_limit_eviction_and_copy_isolation(self):
        store = ResponseStore(max_bytes=180)
        store.put("owner", {"id": "first", "output": []}, [])
        retrieved = store.get("owner", "first")
        retrieved["response"]["output"].append("changed")
        self.assertEqual(store.get("owner", "first")["response"]["output"], [])
        store.put("owner", {"id": "second", "output": ["x" * 80]}, [])
        with self.assertRaises(KeyError):
            store.get("owner", "first")
        self.assertLessEqual(store._bytes, 180)
        with self.assertRaises(ValueError):
            store.put("owner", {"id": "oversized", "output": ["x" * 300]}, [])
        self.assertEqual(store.get("owner", "second")["response"]["id"], "second")

    def test_concurrent_put_get_delete_and_expiry_do_not_corrupt_accounting(self):
        now = [0]
        store = ResponseStore(max_entries=8, max_bytes=4096, ttl=2, clock=lambda: now[0])

        def work(index):
            for iteration in range(20):
                key = f"{index}-{iteration}"
                store.put(str(index), {"id": key, "output": []}, [])
                try:
                    self.assertEqual(store.get(str(index), key)["response"]["id"], key)
                    store.delete(str(index), key)
                except KeyError:
                    pass  # Another request may evict this snapshot between operations.

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(8)))
        self.assertGreaterEqual(store._bytes, 0)
        self.assertLessEqual(store._bytes, 4096)
        store.put("o", {"id": "expires", "output": []}, [])
        now[0] = 2
        with self.assertRaises(KeyError):
            store.get("o", "expires")
        self.assertEqual(store._bytes, 0)


if __name__ == "__main__":
    unittest.main()
