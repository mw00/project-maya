import json
import unittest
from unittest.mock import patch

from serve.test_responses import CALL, TOOLS, ResponseCase, openai

SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}},
          "required": ["count"], "additionalProperties": False}
FORMAT = {"format": {"type": "json_schema", "name": "count", "schema": SCHEMA, "strict": True}}


class StructuredResponses(ResponseCase):
    def test_schema_success_stream_and_complete(self):
        self.script('{"count":3}')
        status, out = self.post(text=FORMAT)
        self.assertEqual(status, 200, out)
        self.assertEqual(out["text"], FORMAT)
        self.assertEqual(json.loads(out["output"][0]["content"][0]["text"]), {"count": 3})
        self.assertIn("JSON Schema", self.tok.decode(self.engine.last_prompt))
        events = self.stream(text=FORMAT)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"), '{"count":3}')

    def test_invalid_output_never_leaks_as_a_delta_and_usage_is_retained(self):
        for script in ('{"count":true}', '{"count":"three"}', '{}', '{"count":3,"extra":1}',
                       '[]', 'not JSON', '```json\n{"count":3}\n```', '{"count":NaN}'):
            with self.subTest(script=script):
                self.script(script)
                events = self.stream(text=FORMAT)
                out = events[-1]["response"]
                self.assertEqual(out["status"], "failed")
                self.assertFalse(any(e["type"] == "response.output_text.delta" for e in events))
                self.assertGreater(out["usage"]["output_tokens"], 0)
                self.assertEqual(out["output"], [])

    def test_json_object_and_output_budget(self):
        self.script('{"hello":"world"}')
        status, out = self.post(text={"format": {"type": "json_object"}})
        self.assertEqual(status, 200)
        events = self.stream(text=FORMAT, max_output_tokens=4)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertFalse(any(e["type"] == "response.output_text.delta" for e in events))

    def test_strict_function_arguments_are_validated_before_publication(self):
        schema = {"type": "object", "properties": {"key": {"type": "string", "enum": ["alpha"]}},
                  "required": ["key"], "additionalProperties": False}
        tools = [{**TOOLS[0], "parameters": schema, "strict": True}]
        self.script(CALL)
        events = self.stream(tools=tools)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertTrue(any(e["type"] == "response.function_call_arguments.done" for e in events))
        self.script(CALL.replace("alpha", "beta"))
        events = self.stream(tools=tools)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertFalse(any(e["type"] in ("response.output_item.added", "response.function_call_arguments.delta") for e in events))

    def test_schema_subset_rejects_refs_bad_schemas_and_unknown_keywords(self):
        schemas = [{**SCHEMA, "$ref": "https://example.test/schema"},
                   {**SCHEMA, "patternProperties": {}}, {"type": "array"},
                   {"type": "object", "properties": {"a": {"type": "bogus"}}},
                   {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}]
        for schema in schemas:
            status, out = self.post(text={"format": {"type": "json_schema", "name": "invalid", "schema": schema, "strict": True}})
            self.assertEqual(status, 400, out)
        self.assertEqual(self.engine.last_prompt, [])

    def test_nested_nullable_array_and_boundary_constraints(self):
        schema = {"type": "object", "properties": {"items": {"type": "array", "minItems": 1, "maxItems": 3,
                    "items": {"type": ["integer", "null"], "minimum": 0, "maximum": 10}}},
                  "required": ["items"], "additionalProperties": False}
        fmt = {"format": {"type": "json_schema", "name": "items", "schema": schema, "strict": True}}
        self.script('{"items":[1,null,10]}')
        self.assertEqual(self.post(text=fmt)[0], 200)
        for script in ('{"items":[]}', '{"items":[true]}', '{"items":[11]}'):
            self.script(script)
            self.assertEqual(self.post(text=fmt)[0], 500)

    def test_validation_buffer_and_deadline_limits_release_the_engine(self):
        self.script('{"count":3}')
        with patch("serve.response_schema.MAX_BYTES", 512):
            events = self.stream(text=FORMAT)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(self.svc.active, 0)
        with patch("serve.response_schema.ResponseValidation.VALIDATION_SECONDS", 0):
            events = self.stream(text=FORMAT)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.script("Ready")
        self.assertEqual(self.post()[0], 200)

    def test_plain_chat_does_not_require_jsonschema_import(self):
        import builtins
        original = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("not installed")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=importing):
            self.assertEqual(self.post()[0], 200)
            status, out = self.post(text=FORMAT)
        self.assertEqual(status, 400)
        self.assertIn("jsonschema", out["error"]["message"])

    @unittest.skipUnless(openai, "optional OpenAI SDK is not installed")
    def test_official_sdk_parses_structured_output(self):
        from pydantic import BaseModel

        class Count(BaseModel):
            count: int

        self.script('{"count":3}')
        with openai.OpenAI(base_url=self.base + "/v1", api_key="local", max_retries=0) as client:
            out = client.responses.parse(model="m", input="count", reasoning={"effort": "none"}, text_format=Count)
        self.assertEqual(out.output_parsed.count, 3)


if __name__ == "__main__":
    unittest.main()
