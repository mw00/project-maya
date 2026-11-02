"""Tool calls in GLM-5.x's own form (issue #5: every GLM call was "malformed") and Qwen's, streamed and whole, and a
call quoted in a code fence or inline code staying text (Strata #1058).
    python -m unittest serve.test_tools"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser, parse_tool_call  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
        "where": {"type": "object"}, "note": {"type": "string"}}}},
    {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
]
GLM = ("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>"
       "\n<arg_value>3</arg_value>\n<arg_key>metric</arg_key>\n<arg_value>true</arg_value>\n<arg_key>where</arg_key>"
       "\n<arg_value>{\"lat\": 48.9, \"lon\": 2.35}</arg_value>\n<arg_key>note</arg_key>\n<arg_value>123</arg_value>"
       "\n</tool_call>")
GLM_ARGS = {"city": "Paris", "days": 3, "metric": True, "where": {"lat": 48.9, "lon": 2.35}, "note": "123"}
QWEN = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>"
        "\n</function>\n</tool_call>")


def run(text: str, pieces: int | None = None, tools=TOOLS) -> list:
    """The parser's events for `text` after </think>, fed whole (pieces=None) or `pieces` characters at a time."""
    p = OutputParser(thinking=True, tools=tools, stream_tools=True)
    text = "Let me look.</think>\n\n" + text
    out = []
    step = pieces or len(text)
    for i in range(0, len(text), step):
        out += p.feed(text[i:i + step])
    return out + p.finish()


def calls(events) -> list:
    return [e.call for e in events if e.kind == "tool_call"]


def content(events) -> str:
    return "".join(e.text for e in events if e.kind == "content")


class GlmCalls(unittest.TestCase):
    def test_whole(self):
        c = parse_tool_call(GLM[len("<tool_call>"):-len("</tool_call>")], TOOLS[0])
        self.assertEqual((c.name, c.arguments), ("get_weather", GLM_ARGS))

    def test_without_a_schema_and_without_arguments(self):
        c = parse_tool_call("get_weather<arg_key>days</arg_key><arg_value>3</arg_value>")
        self.assertEqual(c.arguments, {"days": 3})
        self.assertEqual(parse_tool_call("now").arguments, {})

    def test_malformed(self):
        for body in ("<arg_key>a</arg_key><arg_value>1</arg_value>", "f<arg_key>a<arg_value>1</arg_value>",
                     "f junk <b>", "<function=f><parameter=a>1</parameter><parameter=b",
                     "<function=f><parameter=a"):   # (#706: an unclosed <parameter= tag is malformed, not a crash)
            with self.assertRaises(ValueError, msg=body):
                parse_tool_call(body)

    def test_malformed_carries_the_whole_body(self):   # (#613: the log shows all of it, the message 80 characters)
        body = "f<arg_key>a<arg_value>" + "x" * 300
        with self.assertRaises(ValueError) as cm:
            parse_tool_call(body)
        self.assertEqual(cm.exception.body, body)
        self.assertLess(len(str(cm.exception)), 120)

    def test_streamed_as_whole(self):
        for pieces in (None, 1, 3, 7):
            ev = run("Checking the weather.\n\n" + GLM, pieces)
            got = calls(ev)
            self.assertEqual(len(got), 1, pieces)
            self.assertEqual((got[0].name, got[0].arguments), ("get_weather", GLM_ARGS), pieces)
            starts = [e for e in ev if e.kind == "tool_start"]
            self.assertEqual([s.call.name for s in starts], ["get_weather"], pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), GLM_ARGS, pieces)       # the pieces add up to the arguments
            self.assertEqual(starts[0].call.id, got[0].id)
            self.assertEqual(content(ev), "Checking the weather.", pieces)

    def test_a_string_value_keeps_its_newlines(self):
        body = "bash<arg_key>command</arg_key><arg_value>\nls -la\n</arg_value>"
        self.assertEqual(parse_tool_call(body, TOOLS[1]).arguments["command"], "\nls -la\n")
        for pieces in (None, 1):
            ev = run("<tool_call>" + body + "</tool_call>", pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"command": "\nls -la\n"}, pieces)

    def test_two_calls(self):
        two = GLM + "\n<tool_call>bash<arg_key>command</arg_key><arg_value>date</arg_value></tool_call>"
        for pieces in (None, 2):
            got = calls(run(two, pieces))
            self.assertEqual([c.name for c in got], ["get_weather", "bash"], pieces)
            self.assertEqual(got[1].arguments, {"command": "date"})

    def test_qwen_form_still_works(self):
        for pieces in (None, 1, 5):
            ev = run(QWEN, pieces)
            got = calls(ev)
            self.assertEqual((got[0].name, got[0].arguments), ("get_weather", {"city": "Paris", "days": 3}), pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"city": "Paris", "days": 3}, pieces)


class QuotedCalls(unittest.TestCase):
    """A call quoted in the answer's code fence or inline code is text; at the top level it is a call."""

    def test_in_a_fence(self):
        for fence in ("```", "~~~", "```xml"):
            text = f"The format looks like this:\n\n{fence}\n" + GLM + "\n" + fence[:3] + "\n\nThat's all."
            for pieces in (None, 1, 4):
                ev = run(text, pieces)
                self.assertEqual(calls(ev), [], (fence, pieces))
                self.assertIn("<tool_call>get_weather", content(ev))
                self.assertTrue(content(ev).endswith("That's all."), (fence, pieces))

    def test_in_inline_code(self):
        text = "Write `<tool_call>bash<arg_key>command</arg_key><arg_value>rm -rf build</arg_value></tool_call>` to run it."
        for pieces in (None, 1, 3):
            ev = run(text, pieces)
            self.assertEqual(calls(ev), [], pieces)
            self.assertEqual(content(ev), text, pieces)

    def test_after_a_closed_fence_and_mid_sentence(self):
        text = "Example:\n```\n<tool_call>x</tool_call>\n```\nNow for real. " + GLM
        for pieces in (None, 1):
            got = calls(run(text, pieces))
            self.assertEqual([c.name for c in got], ["get_weather"], pieces)


class HttpToolCalls(unittest.TestCase):
    """Through the server: a GLM call reaches an OpenAI and an Anthropic client as a tool call."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "Let me look.</think>\n\n" + GLM, max_context=8192), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())

    def test_openai(self):
        tools = [{"type": "function", "function": t} for t in TOOLS]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools,
                                                      "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        fn = msg["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", GLM_ARGS))

    def test_anthropic(self):
        tools = [{"name": t["name"], "input_schema": t["parameters"]} for t in TOOLS]
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "tools": tools,
                                             "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(code, 200, b)
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual((use[0]["name"], use[0]["input"]), ("get_weather", GLM_ARGS))
        self.assertEqual(b["stop_reason"], "tool_use")


class ClientStops(unittest.TestCase):
    """The client's stop strings (OpenAI "stop", Anthropic "stop_sequences") end the answer - not the reasoning - and
    tool_choice "none" offers no tools; an empty assistant turn is not rendered (from Strata 0.1.40)."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "Thinking: END is a word.</think>\n\nHello there END and more", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.tok = tok

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = HttpToolCalls.post

    def test_openai_stop(self):
        msgs = [{"role": "user", "content": "hi"}]
        for stop in ("END", ["nope", "END"]):
            code, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, "stop": stop})
            self.assertEqual(code, 200, b)
            self.assertEqual(b["choices"][0]["message"]["content"].strip(), "Hello there")
            self.assertEqual(b["choices"][0]["finish_reason"], "stop")
            self.assertIn("END is a word", b["choices"][0]["message"].get("reasoning_content", ""))  # not cut there

    def test_openai_stop_streamed(self):
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
            {"model": "m", "stream": True, "stop": ["END"], "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            body = resp.read().decode()
        text = "".join((json.loads(line[6:])["choices"][0]["delta"].get("content") or "")
                       for line in body.splitlines() if line.startswith("data: {"))
        self.assertEqual(text.strip(), "Hello there")
        self.assertNotIn("EN", text.replace("Hello there", ""))

    def test_anthropic_stop_sequence(self):
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 200, "stop_sequences": ["END"],
                                             "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, b)
        self.assertEqual((b["stop_reason"], b["stop_sequence"]), ("stop_sequence", "END"))
        self.assertEqual("".join(c.get("text", "") for c in b["content"] if c["type"] == "text").strip(), "Hello there")
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 200,
                                             "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((b["stop_reason"], b["stop_sequence"]), ("end_turn", None))

    def test_tool_choice_none(self):
        tools = [{"type": "function", "function": TOOLS[0]}]
        self.post("/v1/chat/completions", {"model": "m", "tools": tools, "tool_choice": "none",
                                           "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("get_weather", self.tok.decode(self.engine.last_prompt))
        self.post("/v1/chat/completions", {"model": "m", "tools": tools, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("get_weather", self.tok.decode(self.engine.last_prompt))

    def test_empty_assistant_turns_are_dropped(self):
        from serve.frontend import openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": "a"},
                                                      {"role": "assistant", "content": ""},
                                                      {"role": "user", "content": "b"}]})
        self.assertEqual([m["role"] for m in msgs], ["user", "user"])


class JsonMode(unittest.TestCase):
    """response_format json_object: the JSON alone, out of a code fence or the prose around it (Strata #762)."""

    def test_extraction(self):
        from serve.server import json_from_text
        cases = [('{"a": 1}', '{"a": 1}'), ('Sure:\n```json\n{"a": [1, 2]}\n```\nDone.', '{"a": [1, 2]}'),
                 ('Here it is: {"ok": true, "n": {"x": "}"}} - enjoy', '{"ok": true, "n": {"x": "}"}}'),
                 ('[1, 2, 3]', '[1, 2, 3]'), ('no json here', 'no json here')]
        for given, want in cases:
            self.assertEqual(json_from_text(given), want, given)

    def test_over_http(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, '</think>\n\nSure! ```json\n{"city": "Paris"}\n```', max_context=8192), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, b = HttpToolCalls.post(self, "/v1/chat/completions", {
            "model": "m", "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "json"}]})
        self.assertEqual(json.loads(b["choices"][0]["message"]["content"]), {"city": "Paris"})

    SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
              "additionalProperties": False}
    MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Where?"}]

    def prompt_for(self, body, template="serve/chat_template.jinja"):
        """What the engine is given for a chat request, and the server's own count of that prompt (/api/tokens)."""
        tok = ByteTokenizer()
        engine = MockEngine(tok, '</think>\n\n{"city": "Paris"}', max_context=8192)
        svc = Service(engine, tok, ChatTemplate(ROOT / template))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, b = HttpToolCalls.post(self, "/v1/chat/completions", {"model": "m", **body})
        self.assertEqual(code, 200, b)
        self.assertEqual(json.loads(b["choices"][0]["message"]["content"]), {"city": "Paris"})
        return tok.decode(engine.last_prompt), svc.count_tokens(body)["tokens"], len(engine.last_prompt)

    def test_schema_reaches_the_model(self):
        """A json_schema request: the model is told the schema, at the end of the leading system message - the
        client's system prompt stays one, ahead of it - and the prompt's count includes it."""
        rf = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": self.SCHEMA}}
        for template, user in (("serve/chat_template.jinja", "<|im_start|>user"), ("tools/ds4/chat_template.jinja",
                                                                                    "<｜User｜>")):
            with self.subTest(template=template):
                prompt, counted, given = self.prompt_for({"messages": self.MESSAGES, "response_format": rf}, template)
                self.assertLess(prompt.index("You are terse."), prompt.index(json.dumps(self.SCHEMA)))
                self.assertLess(prompt.index(json.dumps(self.SCHEMA)), prompt.index(user))
                self.assertEqual(prompt.count(user), 1)
                self.assertEqual(counted, given)

    def test_json_object_and_no_format(self):
        """json_object: a plain JSON note, or the schema when one comes as llama.cpp sends it (response_format.schema,
        and no system message to add to); a request without a JSON format: the prompt as before."""
        prompt, _, _ = self.prompt_for({"messages": self.MESSAGES, "response_format": {"type": "json_object"}})
        self.assertIn("Return only a valid JSON object", prompt)
        prompt, _, _ = self.prompt_for({"messages": self.MESSAGES[1:],
                                        "response_format": {"type": "json_object", "schema": self.SCHEMA}})
        self.assertIn("<|im_start|>system\n", prompt)
        self.assertIn(json.dumps(self.SCHEMA), prompt)
        for rf in (None, {"type": "text"}):
            body = {"messages": self.MESSAGES, **({"response_format": rf} if rf else {})}
            prompt, _, _ = self.prompt_for(body)
            self.assertNotIn("Return only", prompt)


class ConditionalThinkingTools(unittest.TestCase):
    """A RED-SNOW-style template must open the block the server's parser expects by default."""

    TEMPLATE = ("{% if tools %}<tools>{% for tool in tools %}{{ tool.name }}{% endfor %}</tools>{% endif %}"
                "<|assistant|>{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}"
                "{# <arg_key> #}")
    TOOL = {"name": "add_numbers", "description": "Add two integers", "parameters": {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}
    CALL = ("<tool_call>add_numbers<arg_key>a</arg_key><arg_value>137</arg_value>"
            "<arg_key>b</arg_key><arg_value>268</arg_value></tool_call>")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "chat_template.jinja"
        path.write_text(self.TEMPLATE, encoding="utf-8")
        self.template = ChatTemplate(path)

    def server(self, script):
        tok = ByteTokenizer()
        engine = MockEngine(tok, script, max_context=8192)
        svc = Service(engine, tok, self.template, model_name="red-snow")
        svc.default_effort = "max"
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine, tok

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def test_auto_max_opens_thinking_and_parses_tool_call(self):
        engine, tok = self.server("I should use the function.</think>\n\n" + self.CALL)
        body = self.post({"model": "red-snow", "reasoning_effort": "max", "tools": [
            {"type": "function", "function": self.TOOL}], "tool_choice": "auto",
            "messages": [{"role": "user", "content": "Use add_numbers for 137+268."}]})
        prompt = tok.decode(engine.last_prompt)
        self.assertIn("<tools>add_numbers</tools>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|><think>"))
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        fn = body["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("add_numbers", {"a": 137, "b": 268}))

    def test_required_call_keeps_thinking_off(self):
        engine, tok = self.server(self.CALL[len("<tool_call>"):])
        body = self.post({"model": "red-snow", "reasoning_effort": "max", "tools": [
            {"type": "function", "function": self.TOOL}], "tool_choice": "required",
            "messages": [{"role": "user", "content": "Use add_numbers for 137+268."}]})
        prompt = tok.decode(engine.last_prompt)
        self.assertTrue(prompt.endswith("<|assistant|><tool_call>"))
        self.assertNotIn("<think>", prompt[-40:])
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        fn = body["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("add_numbers", {"a": 137, "b": 268}))


class ForcedToolCalls(unittest.TestCase):
    """tool_choice "required" / a named function (Anthropic "any" / "tool"): the answer starts with the call - the
    prompt ends with its opening, thinking off - and the model's continuation is the call (from Strata #790)."""

    BODY = "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"

    def server(self, script):
        tok = ByteTokenizer()
        engine = MockEngine(tok, script, max_context=8192)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine, tok

    post = HttpToolCalls.post

    def test_named(self):
        engine, tok = self.server(self.BODY)
        tools = [{"type": "function", "function": TOOLS[0]}]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools, "messages": [
            {"role": "user", "content": "hi"}], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        self.assertEqual(code, 200, b)
        self.assertTrue(tok.decode(engine.last_prompt).endswith("<tool_call>\n<function=get_weather>\n"))
        fn = b["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", {"city": "Paris"}))

    def test_required_and_anthropic_any(self):
        engine, tok = self.server("<function=get_weather>\n" + self.BODY)
        tools = [{"type": "function", "function": TOOLS[0]}]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools, "tool_choice": "required",
                                                      "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls", b)
        self.assertTrue(tok.decode(engine.last_prompt).endswith("<tool_call>\n"))
        atools = [{"name": "get_weather", "input_schema": TOOLS[0]["parameters"]}]
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 300, "tools": atools,
                                             "tool_choice": {"type": "any"},
                                             "messages": [{"role": "user", "content": "hi"}]})
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual((b["stop_reason"], use[0]["name"], use[0]["input"]), ("tool_use", "get_weather", {"city": "Paris"}))

    def test_auto_adds_nothing(self):
        engine, tok = self.server("</think>\n\nhello")
        tools = [{"type": "function", "function": TOOLS[0]}]
        self.post("/v1/chat/completions", {"model": "m", "tools": tools, "tool_choice": "auto",
                                           "messages": [{"role": "user", "content": "hi"}]})
        self.assertFalse(tok.decode(engine.last_prompt).rstrip().endswith("<tool_call>"))


class CallEndRescan(unittest.TestCase):
    """In call state the parser searched for </tool_call> from offset 0 on every delta while the
    body accumulated - O(body) per token on a large file-write argument.  The search now resumes where
    the last one stopped (less a possible partial tag); every split of the same text must parse the same."""

    BODY = ("<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>"
            + "echo line </tool_cal and </tool_callx and </tool_call\n" * 60
            + "</arg_value>\n</tool_call>done")

    def shape(self, events):
        return (content(events),
                [(c.name, c.arguments) for c in calls(events)],
                "".join(e.text for e in events if e.kind == "tool_args"))

    def test_every_split_agrees_with_whole(self):
        whole = self.shape(run(self.BODY))
        self.assertEqual(len(whole[1]), 1)
        self.assertTrue(whole[1][0][1]["command"].startswith("echo line"))
        for pieces in (1, 2, 3, 5, 11, 64):
            self.assertEqual(self.shape(run(self.BODY, pieces=pieces)), whole, f"pieces={pieces}")

    def test_search_resume_advances_with_the_body(self):
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
        p.feed("think</think>\n\n<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>")
        p.feed("x" * 5000)
        self.assertGreater(p._call_resume, 4000)
        p.feed("</arg_value>\n</tool_cal")          # a partial end tag: not a close, and not skipped either
        self.assertEqual(p.state, "call")
        out = p.feed("l>")
        self.assertEqual([e.kind for e in out].count("tool_call"), 1)


if __name__ == "__main__":
    unittest.main()
