"""serve/test_inkling2_frontend.py - Inkling-2-Small in the frontend and the server: thinkingmachines' own chat
template detected and rendered at every level a client names (its effort_map's names, thinking off as "none"), the
messages it refuses as clients send them (a tool call's arguments as a JSON string, None content) taken, its answers
- typed segments, not tags - read into reasoning, answer and tool calls however the stream is cut, its end of turn
a stop, a required tool call opened in its own form, and pictures refused in plain words (maya-ik2 has no vision
tower yet).

    python -m unittest serve.test_inkling2_frontend

Offline, no GPU: the template is tools/ik2/chat_template.jinja, the tokenizer a small stand-in with the template's
marks as single tokens, the engine a scripted one.
"""
from __future__ import annotations

import hashlib
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate, InklingRewriter, OutputParser, effort_kwargs, inkling_messages, \
    openai_to_messages
from serve.server import ByteTokenizer, Service, StrataEngine

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "tools" / "ik2" / "chat_template.jinja"
PUBLIC_SHA256 = "e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7"
USER, MODEL, SYSTEM, TOOL = "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>"
THINK, TEXT, CALL, END = "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>"
EOS = "<|content_model_end_sampling|>"
EFFORT = "<|message_system|><|content_text|>Thinking effort level: "
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Weather in Paris?"}]
PICTURE = {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}


def public() -> ChatTemplate:
    return ChatTemplate(PUBLIC)


def call(name, args) -> str:
    """A tool call as the model writes it (and its template renders a past one)."""
    return MODEL + name + CALL + json.dumps({"name": name, "args": args}, separators=(",", ":")) + END


class InkTok(ByteTokenizer):
    """ByteTokenizer with the template's marks as one token each; its eos the end of sampling (the GGUF's)."""
    SPECIALS = ByteTokenizer.SPECIALS + [USER, MODEL, SYSTEM, TOOL, THINK, TEXT, CALL, END, EOS, "<|content_xml|>",
                                         "<|content_image|>"]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens[EOS]}


class Scripted:
    """Request k's answer is scripts[k] (the last one again after that), then the end of sampling.  Records every
    prompt.  Like maya-ik2: no logprobs, no logit bias, one conversation."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.scripts = tok, 32768, scripts
        self.prompts, self.sampling = [], []
        self.info = {"engine": "ik2", "engine_kind": "ik2"}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.sampling.append(dict(sampling or {}))
        text = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        for t in (self.tok.encode(text, parse_special=True) + [self.tok.special_tokens[EOS]])[:max_new]:
            yield t


def service(*scripts):
    tok = InkTok()
    eng = Scripted(tok, *scripts)
    return Service(eng, tok, public()), eng, tok


def answer(svc, req):
    msgs, tools, kw = openai_to_messages(req)
    ids, thinking, max_new = svc.prepare(msgs, tools, kw, 500)
    sampling = {k: kw.get(k) for k in ("_prefill", "_blocked", "_think_resume")}
    events, done = [], None
    for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
        if kind == "event":
            events.append(x)
        elif kind == "done":
            done = x
    return {"content": "".join(e.text for e in events if e.kind == "content"),
            "reasoning": "".join(e.text for e in events if e.kind == "reasoning"),
            "calls": [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], "done": done}


class Template(unittest.TestCase):
    def test_public_copy(self):   # (line endings aside: a checkout may rewrite them)
        raw = PUBLIC.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PUBLIC_SHA256)

    def test_detected(self):
        t = public()
        self.assertTrue(t.inkling)
        self.assertFalse(t.minimax or t.deepseek or t.qwen or t.hy4 or t.glm_effort or t.glm_dsa)

    def test_other_templates_are_not(self):
        for path in ("serve/chat_template.jinja", "serve/glm5_chat_template.jinja", "tools/mm3/chat_template.jinja",
                     "serve/fixtures/hy4_flash/chat_template.jinja"):
            self.assertFalse(ChatTemplate(ROOT / path).inkling, path)

    def test_levels(self):
        # its own names reach it as they are; thinking off is its "none" (0); none named: its default, 0.9
        t = public()
        for word, num in (("none", "0"), (False, "0"), ("off", "0"), ("minimal", "0"), ("low", "0.2"),
                          ("medium", "0.7"), ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"), ("auto", "0.9"),
                          (None, "0.9")):
            with self.subTest(word=word):
                out = t.render(ASK, **effort_kwargs(word))
                self.assertIn(EFFORT + num + END, out)
                self.assertFalse(t.opens_thinking(out))   # (the model opens its own segments)

    def test_render(self):
        out = public().render([{"role": "system", "content": "Be terse."}] + ASK)
        self.assertEqual(out, SYSTEM + TEXT + "Be terse." + END + EFFORT + "0.9" + END + USER + TEXT +
                         "Weather in Paris?" + END)

    def test_history_as_clients_send_it(self):
        # a past call's arguments as a JSON string (OpenAI's form) and its content None: the template refuses both
        # as given - they go in as an object and ""; the same prompt as the canonical form
        t = public()
        flat = [x["function"] for x in TOOLS]
        tc = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"days": 2, "city": "Paris"}'}}
        given = ASK + [{"role": "assistant", "content": None, "reasoning_content": "Look it up.", "tool_calls": [tc]},
                       {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 21}'}]
        canon = json.loads(json.dumps(given))
        canon[1]["content"] = ""
        canon[1]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris", "days": 2}
        out = t.render(given, tools=flat)
        self.assertEqual(out, t.template.render(messages=canon, tools=TOOLS, add_generation_prompt=True))
        self.assertIn('<|message_system|>tool_declare<|content_xml|>[{"description":"The weather of a city."', out)
        self.assertIn(MODEL + THINK + "Look it up." + END + call("get_weather", {"city": "Paris", "days": 2}) + EOS,
                      out)
        self.assertIn(TOOL + "get_weather" + TEXT + '{"temp": 21}' + END, out)

    def test_messages(self):
        got = inkling_messages([{"role": "developer", "content": "x"},
                                {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [
                                    {"function": {"name": "f", "arguments": ""}},
                                    {"function": {"name": "g", "arguments": "not json"}}]}])
        self.assertEqual(got[0], {"role": "system", "content": "x"})
        self.assertEqual((got[1]["content"], got[1]["reasoning_content"]), ("", "r"))
        self.assertEqual([c["function"]["arguments"] for c in got[1]["tool_calls"]], [{}, {"_raw": "not json"}])


class Parser(unittest.TestCase):
    OUT = {
        "think+text": (MODEL + THINK + "Let me see." + END + MODEL + TEXT + "The answer is 4." + END + EOS,
                       "Let me see.", "The answer is 4.", []),
        "text only": (MODEL + TEXT + "Hi there!" + END + EOS, "", "Hi there!", []),
        "plain": ("Just words.", "", "Just words.", []),
        "think+call": (MODEL + THINK + "Need it." + END +
                       call("get_weather", {"city": "a <b> \"c\"", "days": 3, "opts": {"x": [1, 2]}}) + EOS,
                       "Need it.", "", [("get_weather", {"city": "a <b> \"c\"", "days": 3, "opts": {"x": [1, 2]}})]),
        "text+call": (MODEL + TEXT + "Looking." + END + call("get_weather", {"city": "Rome"}) + EOS,
                      "", "Looking.", [("get_weather", {"city": "Rome"})]),
    }

    def feed(self, text, thinking, step):
        p, events = OutputParser(thinking=thinking, tools=[t["function"] for t in TOOLS], stream_tools=True,
                                 inkling=True), []
        for i in range(0, len(text), step):
            events += p.feed(text[i:i + step])
        events += p.finish()
        return ("".join(e.text for e in events if e.kind == "reasoning"),
                "".join(e.text for e in events if e.kind == "content").strip(),
                [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"])

    def test_however_cut(self):
        for name, (text, reasoning, content, calls) in self.OUT.items():
            for thinking in (False, True):
                for step in (1, 2, 3, 7, 1000):
                    with self.subTest(name=name, thinking=thinking, step=step):
                        self.assertEqual(self.feed(text, thinking, step), (reasoning, content, calls))

    def test_unfinished_call_is_text(self):
        r = InklingRewriter(thinking=False)
        out = r.feed(MODEL + "get_weather" + CALL + '{"name":"get_weather","args":{"ci') + r.finish()
        self.assertEqual(out, '{"name":"get_weather","args":{"ci')


class Serving(unittest.TestCase):
    def test_stop(self):
        svc, _, tok = service("x")
        self.assertIn(tok.special_tokens[EOS], svc.stop_ids)
        self.assertEqual(svc.eos_ids, {tok.special_tokens[EOS]})

    def test_thinking_and_answer(self):
        svc, eng, tok = service(MODEL + THINK + "Thinking." + END + MODEL + TEXT + "Hello." + END)
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"], got["calls"]), ("Thinking.", "Hello.", []))
        self.assertTrue(tok.decode(eng.prompts[0]).endswith(USER + TEXT + "Weather in Paris?" + END))

    def test_effort(self):
        svc, eng, tok = service(MODEL + TEXT + "Hello." + END)
        answer(svc, {"messages": ASK, "reasoning_effort": "none"})
        answer(svc, {"messages": ASK, "reasoning_effort": "low"})
        svc.default_effort = "xhigh"
        answer(svc, {"messages": ASK})
        said = [tok.decode(p) for p in eng.prompts]
        for text, num in zip(said, ("0", "0.2", "0.99")):
            self.assertIn(EFFORT + num + END, text)

    def test_tool_call(self):
        svc, _, _ = service(MODEL + THINK + "Look it up." + END + call("get_weather", {"city": "Paris", "days": 2}))
        got = answer(svc, {"messages": ASK, "tools": TOOLS})
        self.assertEqual(got["calls"], [("get_weather", {"city": "Paris", "days": 2})])
        self.assertEqual(got["reasoning"], "Look it up.")

    def test_required_call_opens_in_its_own_form(self):
        svc, eng, tok = service('{"name":"get_weather","args":{"city":"Rome"}}' + END)
        got = answer(svc, {"messages": ASK, "tools": TOOLS,
                           "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        want = tok.encode(MODEL + "get_weather" + CALL, parse_special=True)
        self.assertEqual(eng.prompts[0][-len(want):], want)
        self.assertIn(EFFORT + "0" + END, tok.decode(eng.prompts[0]))   # (thinking off for a required call)
        self.assertEqual(got["calls"], [("get_weather", {"city": "Rome"})])

    def test_any_required_call_opens_with_a_tools_name(self):
        # tool_choice "required" names no tool, and Inkling's call opens with the tool's name: the prompt can open the
        # model's message only.  Where the engine reads starts=, the message must go on with one of the tools' names and
        # the call's segment (the model's pick among them), not with words
        svc, eng, tok = service("get_weather" + CALL + '{"name":"get_weather","args":{"city":"Rome"}}' + END)
        eng.info["starts"] = 1
        tools = TOOLS + [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]
        msgs, tools, kw = openai_to_messages({"messages": ASK, "tools": tools, "tool_choice": "required"})
        ids, _, _ = svc.prepare(msgs, tools, kw, 500)
        self.assertEqual(tok.decode(ids[-1:]), MODEL)
        seg = tok.special_tokens[CALL]
        self.assertEqual(sorted(kw["_starts"]),
                         sorted([tok.encode("get_weather") + [seg], tok.encode("get_time") + [seg]]))
        self.assertIn(EFFORT + "0" + END, tok.decode(ids))   # (thinking off, as for a named call)

    def test_openings_only_where_needed_and_read(self):
        for choice, info in (("required", {}),                                       # (an engine without it)
                             ({"type": "function", "function": {"name": "get_weather"}}, {"starts": 1}),
                             ("auto", {"starts": 1})):
            svc, eng, _ = service("x")
            eng.info.update(info)
            msgs, tools, kw = openai_to_messages({"messages": ASK, "tools": TOOLS, "tool_choice": choice})
            svc.prepare(msgs, tools, kw, 500)
            self.assertIsNone(kw["_starts"], choice)

    def test_openings_go_to_the_engine_for_the_answers_start(self):
        self.assertIn("starts=3,7;9", StrataEngine.sampling_keys({"_starts": [[3, 7], [9]]}).split())
        self.assertIn("starts=3", StrataEngine.sampling_keys({"_starts": [[3]], "temperature": 0}).split())
        self.assertNotIn("starts", StrataEngine.sampling_keys({"_starts": None, "temperature": 0.7}))
        # a later call for the same answer goes on after what was written: cut inside the opening, with what the
        # sequences it matches have left; once one is whole (or none matches), freely
        self.assertEqual(server.openings_left([[1, 2, 3], [1, 4], [5]], [1]), [[2, 3], [4]])
        self.assertEqual(server.openings_left([[1, 2, 3], [1, 4], [5]], [1, 2]), [[3]])
        self.assertIsNone(server.openings_left([[1, 2, 3], [1, 4]], [1, 4, 9]))
        self.assertIsNone(server.openings_left([[1, 2, 3]], [7]))
        # (through _sliced: the thinking budget's close at a sentence's end stops the first call after "Hi.", the
        # next goes on - the opening was written whole)
        svc, eng, tok = service("Hi. There.")
        hi = tok.encode("Hi.")
        smp = {"_starts": [hi, [2]], "_think_soft": 1, "_think_end": tok.special_tokens[END]}
        list(svc._sliced(tok.encode("Q"), 50, smp, threading.Event(), None, [True, True], {}))
        self.assertEqual([x.get("_starts") for x in eng.sampling], [[hi, [2]], None])

    def test_pictures_are_refused_in_plain_words(self):
        svc, eng, _ = service("x")
        msgs, tools, kw = openai_to_messages({"messages": [PICTURE]})
        with self.assertRaises(ValueError) as e:
            svc.prepare(msgs, tools, kw, 100)
        self.assertIn("on maya-ik2 reads text only without the image encoder", str(e.exception))
        self.assertEqual(eng.prompts, [])

    def test_logprobs_and_bias_are_refused(self):   # (an engine whose INFO names neither)
        svc, _, _ = service("x")
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logprobs": True})
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logit_bias": {"65": 5}})


class Defaults(unittest.TestCase):
    CFG = {"exe": "/opt/maya/build/maya-ik2", "args": ["--model", "/models/Inkling-2-Small-UD-Q3_K_XL-00001-of-00005.gguf",
                                                       "--max-context", "32768"], "model_name": "inkling-2-small"}

    def test_models_own(self):
        Q = server.maya_quants
        self.assertEqual(Q.family_of(self.CFG), "ik2")
        self.assertEqual(Q.model_defaults(self.CFG), ({"temperature": 1.0, "top_p": 1.0}, "auto"))

    def test_config_effort(self):
        cfg = server.checked_config_effort(dict(self.CFG), public(), True)
        self.assertEqual(cfg["reasoning_effort"], "auto")



class Http(unittest.TestCase):
    """A required call through each API: the openings reach the engine with the request (OpenAI's chat, Anthropic's
    messages, Responses)."""

    def setUp(self):
        self.svc, self.eng, self.tok = service("get_weather" + CALL + '{"name":"get_weather","args":{"city":"Rome"}}'
                                               + END)
        self.eng.info["starts"] = 1
        self.httpd = server.serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def check(self, path, body):
        code, out = self.post(path, body)
        self.assertEqual(code, 200, out)
        self.assertEqual(self.eng.sampling[0].get("_starts"),
                         [self.tok.encode("get_weather") + [self.tok.special_tokens[CALL]]])
        return out

    def test_chat(self):
        out = self.check("/v1/chat/completions", {"model": "x", "messages": ASK, "tools": TOOLS,
                                                  "tool_choice": "required"})
        self.assertEqual(out["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_messages(self):
        tools = [{"name": "get_weather", "description": "Weather.", "input_schema": TOOLS[0]["function"]["parameters"]}]
        out = self.check("/v1/messages", {"model": "x", "max_tokens": 100, "messages": ASK, "tools": tools,
                                          "tool_choice": {"type": "any"}})
        self.assertEqual([b["name"] for b in out["content"] if b["type"] == "tool_use"], ["get_weather"])

    def test_responses(self):
        out = self.check("/v1/responses", {"model": "x", "input": "Weather in Paris?", "tool_choice": "required",
                                           "tools": [{"type": "function", **TOOLS[0]["function"]}]})
        self.assertEqual([o["name"] for o in out["output"] if o["type"] == "function_call"], ["get_weather"])

if __name__ == "__main__":
    unittest.main()
