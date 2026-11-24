"""serve/test_hy4_frontend.py - Hy4-Flash in the frontend and the server: Tencent's public chat template detected and
rendered at the two levels the model is made for (high, or no_think for off - the template raises on a level it does
not list, and would also let max through), every client's spelling of a level mapped onto those two, its answers (reasoning, answer, tool calls in GLM's form under the pack's aliases) parsed, its stop
and end-of-thinking tokens found, its own sampling defaults, and what its engine does not have yet (pictures) refused
in plain words.

    python -m unittest serve.test_hy4_frontend

Offline, no GPU: the template is the copy in serve/fixtures/hy4_flash, the tokenizer a small stand-in with the
template's tags as single tokens and the pack's aliases, the engine a scripted one.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import (AUTO, ChatTemplate, OutputParser, anthropic_to_messages, effort_kwargs, effort_level,
                            openai_to_messages)
from serve.server import ByteTokenizer, Service, StrataEngine

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "serve" / "fixtures" / "hy4_flash" / "chat_template.jinja"
PUBLIC_SHA256 = "63ddd5de294df36f10c8f2c8e5f7b2a9dc2011c8481335422f04554256aee4ad"
TAG = ":opensource"
HY_START, HY_MIDDLE, HY_END = (f"<｜hy_{n}{TAG}｜>" for n in ("start", "middle", "end"))
MODE = f"<｜reasoning_mode{TAG}｜>reasoning_effort:"
OPEN, CLOSE = f"<think{TAG}>", f"</think{TAG}>"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Weather in Paris?"}]
CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key>" \
       "<arg_value>2</arg_value></tool_call>"
PICTURE = {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}


def public() -> ChatTemplate:
    return ChatTemplate(PUBLIC)


def with_low(tmp: Path) -> ChatTemplate:
    """The earlier template of the model: another suffix, and "low" among its levels."""
    src = PUBLIC.read_text(encoding="utf-8").replace(TAG, ":6124c78e")
    src = src.replace("not in ['max', 'high', 'no_think']", "not in ['max', 'high', 'low', 'no_think']")
    path = tmp / "chat_template.jinja"
    path.write_text(src, encoding="utf-8", newline="")
    return ChatTemplate(path)


class Hy4Tok(ByteTokenizer):
    """ByteTokenizer with the template's tags as one token each, read back as the pack's tokenizer reads them: the
    thinking and tool tags in GLM's spelling, the <tool_calls> wrapper as nothing (tools/hy4/pack_gguf.py's aliases)."""
    NAMES = ("think", "tool_calls", "tool_call", "arg_key", "arg_value", "tool_response")
    TAGS = [f"<{slash}{n}{TAG}>" for n in NAMES for slash in ("", "/")]
    SPECIALS = ByteTokenizer.SPECIALS + [HY_START, HY_MIDDLE, HY_END, f"<｜reasoning_mode{TAG}｜>"] + TAGS
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens[HY_END]}

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            s = self.SPECIALS[t - 256] if t >= 256 else None
            if s in self.TAGS:
                s = "" if "tool_calls" in s else s.replace(TAG, "")
            raw += s.encode() if s is not None else bytes([t])
        return raw.decode("utf-8", errors=errors)


class Scripted:
    """Request k's answer is scripts[k] (the last one again after that), written in the template's own tags, then the
    turn's end.  Records every prompt and sampling.  Like maya-hy4: no logprobs, no logit bias, one conversation."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.scripts = tok, 32768, scripts
        self.prompts, self.sampling = [], []
        self.info = {"engine": "hy4", "engine_kind": "hy4", "dspark": "0", "mtp": "0"}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.sampling.append(dict(sampling or {}))
        text = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        for t in (self.tok.encode(text, parse_special=True) + [self.tok.special_tokens[HY_END]])[:max_new]:
            yield t


def service(*scripts, template=None):
    tok = Hy4Tok()
    eng = Scripted(tok, *scripts)
    return Service(eng, tok, template or public()), eng, tok


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
        self.assertTrue(t.hy4)
        self.assertEqual(t.hy4_tag, TAG)
        self.assertFalse(t.minimax or t.deepseek or t.qwen or t.glm_effort or t.glm_dsa)

    def test_other_templates_are_not(self):
        for path in ("serve/chat_template.jinja", "serve/glm5_chat_template.jinja", "tools/mm3/chat_template.jinja"):
            t = ChatTemplate(ROOT / path)
            self.assertFalse(t.hy4, path)
            self.assertEqual((t.hy4_tag, t.thinking_levels()), ("", None), path)
            self.assertIsNone(t.effort_shown("high"), path)

    def test_two_levels_whatever_the_template_lists(self):
        import tempfile
        t = public()
        self.assertEqual(t.thinking_levels(), ["none", "high"])
        with tempfile.TemporaryDirectory() as d:
            low = with_low(Path(d))
            self.assertEqual((low.hy4, low.hy4_tag), (True, ":6124c78e"))
            self.assertEqual(low.thinking_levels(), ["none", "high"])
            # a template that does not list its levels: the same two
            bare = Path(d) / "bare.jinja"
            bare.write_text(re.sub(r"\{%- elif reasoning_effort not in \[[^\]]*\] %\}", "{%- elif false %}",
                                   PUBLIC.read_text(encoding="utf-8")), encoding="utf-8", newline="")
            self.assertEqual(ChatTemplate(bare).thinking_levels(), ["none", "high"])

    def test_render_levels(self):
        t = public()
        for kwargs, word in (({}, "high"), ({"reasoning_effort": AUTO}, "high"), ({"reasoning_effort": "high"}, "high"),
                             ({"reasoning_effort": "max"}, "high"), ({"enable_thinking": False}, "no_think"),
                             ({"reasoning_effort": "low"}, "high"),          # (any other level: high)
                             ({"reasoning_effort": "high", "effort_name": "medium"}, "high"),
                             ({"reasoning_effort": "max", "effort_name": "xhigh"}, "high"),
                             ({"reasoning_effort": "none"}, "no_think"),
                             ({"reasoning_effort": None}, "high"), ({"reasoning_effort": "no_think"}, "no_think")):
            with self.subTest(kwargs=kwargs):
                out = t.render(ASK, **kwargs)
                self.assertIn(MODE + word + HY_END, out)
                tail = HY_START + "assistant" + HY_MIDDLE + OPEN + (CLOSE if word == "no_think" else "")
                self.assertTrue(out.endswith(tail), out[-80:])
                self.assertEqual(t.opens_thinking(out), word != "no_think")

    def test_every_spelling_renders(self):
        # (the public template raises on a level it does not list: no client's word may reach it as written)
        t = public()
        for word in ("none", "off", "minimal", "disabled", "false", "no_think", "low", "medium", "high", "xhigh",
                     "max", "maximum", "auto", "default", False):
            with self.subTest(word=word):
                out = t.render(ASK, **effort_kwargs(word))
                level = effort_level(word)
                want = "no_think" if level == "none" else "high"
                self.assertIn(MODE + want + HY_END, out)
                shown = t.effort_shown(word)
                self.assertEqual(shown, "none" if want == "no_think" else want)
                self.assertIn(shown, t.thinking_levels())

    def test_high_where_the_template_has_low_too(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            t = with_low(Path(d))
            for word, want in (("low", "high"), ("medium", "high"), ("high", "high"), ("xhigh", "high"), ("max", "high"),
                               ("none", "no_think")):
                self.assertIn("reasoning_effort:" + want + "<", t.render(ASK, **effort_kwargs(word)), word)
            self.assertEqual([t.effort_shown(w) for w in ("low", "medium", "high", "max", "auto", "none")],
                             ["high", "high", "high", "high", "high", "none"])

    def test_tools_render(self):
        t = public()
        flat = [x["function"] for x in TOOLS]           # (the server's flat tools go in as the template reads them)
        out = t.render(ASK, tools=flat)
        self.assertEqual(out, t.render(ASK, tools=TOOLS))
        self.assertIn('"name": "get_weather"', out)
        self.assertIn(f"<tool_calls{TAG}>", out)
        history = ASK + [
            {"role": "assistant", "reasoning_content": "Call it.", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"}]
        out = t.render(history, tools=flat)
        self.assertIn(f"<tool_calls{TAG}><tool_call{TAG}>get_weather<arg_key{TAG}>city</arg_key{TAG}><arg_value{TAG}>"
                      f"Paris</arg_value{TAG}><arg_key{TAG}>days</arg_key{TAG}><arg_value{TAG}>2</arg_value{TAG}>"
                      f"</tool_call{TAG}></tool_calls{TAG}>{HY_END}", out)
        self.assertIn('{"temp": 21}', out)

    def test_call_opening(self):
        self.assertEqual(public().hy4_call_open("get_weather"), f"<tool_calls{TAG}><tool_call{TAG}>get_weather")


class Requests(unittest.TestCase):
    """Every way a client names a level ends in kwargs the public template renders."""

    def rendered(self, kw):
        kw = {k: v for k, v in kw.items() if not k.startswith("_")}
        out = public().render(ASK, **kw)
        return re.search(r"reasoning_effort:(\w+)", out).group(1)

    def test_openai(self):
        for req, want in (({}, "high"), ({"reasoning_effort": "none"}, "no_think"), ({"reasoning_effort": "minimal"}, "no_think"),
                          ({"reasoning_effort": "low"}, "high"), ({"reasoning_effort": "medium"}, "high"),
                          ({"reasoning_effort": "xhigh"}, "high"), ({"reasoning": {"effort": "max"}}, "high"),
                          # (the model card's own way)
                          ({"chat_template_kwargs": {"reasoning_effort": "no_think"}}, "no_think"),
                          ({"chat_template_kwargs": {"reasoning_effort": "max"}}, "high"),
                          ({"chat_template_kwargs": {"enable_thinking": False}}, "no_think")):
            with self.subTest(req=req):
                kw = openai_to_messages({"messages": ASK, **req})[2]
                self.assertEqual(self.rendered(kw), want)

    def test_anthropic(self):
        base = {"messages": ASK, "max_tokens": 100}
        for req, want in (({}, "high"), ({"thinking": {"type": "disabled"}}, "no_think"),
                          ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, "high"),   # (a "low" budget)
                          ({"thinking": {"type": "enabled", "budget_tokens": 4096}}, "high"),
                          ({"thinking": {"type": "enabled", "budget_tokens": 32000}}, "high"),
                          ({"output_config": {"effort": "low"}}, "high"), ({"output_config": {"effort": "max"}}, "high")):
            with self.subTest(req=req):
                kw = anthropic_to_messages({**base, **req})[2]
                self.assertEqual(self.rendered(kw), want)

    def test_model_api(self):
        from serve import model_api
        for think, want in ((False, "no_think"), (True, "high"), (None, "high")):
            req = {"model": "hy4-flash", "messages": ASK, **({} if think is None else {"think": think})}
            kw = openai_to_messages(model_api.chat_to_openai(req))[2]
            self.assertEqual(self.rendered(kw), want, think)

    def test_unknown_word_is_a_400(self):
        with self.assertRaises(ValueError):
            openai_to_messages({"messages": ASK, "reasoning_effort": "enormous"})


class Parser(unittest.TestCase):
    def feed(self, text, thinking=True, step=7):
        p, events = OutputParser(thinking=thinking, tools=[t["function"] for t in TOOLS], stream_tools=True), []
        for i in range(0, len(text), step):
            events += p.feed(text[i:i + step])
        events += p.finish()
        return events

    def test_reasoning_answer_call(self):   # (as the pack's tokenizer reads the answer: GLM's spelling, no wrapper)
        for step in (1, 5, 1000):
            ev = self.feed("Paris, two days.</think>Let me look." + CALL, step=step)
            self.assertEqual("".join(e.text for e in ev if e.kind == "reasoning"), "Paris, two days.")
            self.assertEqual("".join(e.text for e in ev if e.kind == "content").strip(), "Let me look.")
            calls = [e.call for e in ev if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris", "days": 2})])

    def test_no_thinking(self):
        ev = self.feed("21 degrees.", thinking=False)
        self.assertEqual({e.kind for e in ev}, {"content"})
        self.assertEqual("".join(e.text for e in ev), "21 degrees.")


class Tokens(unittest.TestCase):
    def test_stop_and_think_end(self):
        svc, _, tok = service("x")
        ids = tok.special_tokens
        self.assertLessEqual({ids[HY_END], ids[HY_START]}, svc.stop_ids)
        self.assertEqual(svc.think_end_id, ids[CLOSE])
        self.assertEqual(svc.eos_ids, {ids[HY_END]})

    def test_think_end_reaches_the_engine(self):
        # the budget's close: think_budget + the template's own </think:..> id on the engine's line
        svc, eng, tok = service("Thinking." + CLOSE + "Hello.")
        svc.think_budget = 32768
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"]), ("Thinking.", "Hello."))
        keys = StrataEngine.sampling_keys(eng.sampling[0])
        self.assertIn(f" think_end={tok.special_tokens[CLOSE]}", keys)
        self.assertRegex(keys, r" think_budget=\d+")

    def test_engine_reads_the_keys(self):
        # every key of a plain request's line is one maya-hy4's parse_gen reads (it ignores the others)
        src = ROOT / "src" / "hy4" / "hy4_main.cpp"
        if not src.exists():
            self.skipTest("no engine source here")
        known = set(re.findall(r'k == "(\w+)"', src.read_text(encoding="utf-8", errors="replace")))
        line = StrataEngine.sampling_keys({"temperature": 0.9, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "seed": 7,
                                           "repetition_penalty": 1.1, "frequency_penalty": 0.2, "presence_penalty": 0.1,
                                           "_think_budget": 100, "_think_end": 5})
        sent = set(re.findall(r" (\w+)=", line))
        self.assertGreaterEqual(sent, {"temperature", "top_p", "top_k", "min_p", "seed", "think_budget", "think_end"})
        self.assertEqual(sent - known, set())


class Defaults(unittest.TestCase):
    CFG = {"exe": "/opt/maya/build/maya-hy4", "args": ["--pack", "/models/hy4/maya-hy4", "--max-context", "32768"],
           "model_name": "hy4-flash", "installer": {"family": "hy4", "hy4_quant": "maya-xs", "quant": "Hy4-Flash-Maya-XS"}}

    def test_models_own(self):   # (a config without "sampling" / "reasoning_effort": Tencent's, not GLM's)
        Q = server.maya_quants
        self.assertEqual(Q.model_defaults(self.CFG), ({"temperature": 0.9, "top_p": 1.0}, "high"))
        self.assertNotEqual(Q.model_defaults(self.CFG), Q.model_defaults({"exe": "/opt/maya/build/maya"}))
        self.assertEqual(Q.default_thinking_budget(self.CFG), 32768)

    def test_config_effort(self):
        t = public()
        cfg = server.checked_config_effort(dict(self.CFG), t, True)
        self.assertEqual(cfg["reasoning_effort"], "high")
        for given, shown in (("high", "high"), ("max", "high"), ("none", "none"), ("low", "high"), ("medium", "high"),
                             ("no_think", "none"), ("auto", "high")):
            cfg = server.checked_config_effort(dict(self.CFG, reasoning_effort=given), t, True)
            self.assertEqual(t.effort_shown(cfg["reasoning_effort"]), shown, given)

    def test_sampling_line(self):   # (top_p 1.0 changes nothing: only the temperature goes)
        self.assertEqual(StrataEngine.sampling_keys({"temperature": 0.9, "top_p": 1.0}), " temperature=0.9")


class Serving(unittest.TestCase):
    def test_thinking_and_answer(self):
        svc, eng, tok = service("Thinking." + CLOSE + "Hello.")
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"], got["calls"]), ("Thinking.", "Hello.", []))
        self.assertTrue(tok.decode(eng.prompts[0]).endswith("<think>"))   # (the open block, by its alias)

    def test_default_effort_and_off(self):
        svc, eng, tok = service("Hello.", "Thinking." + CLOSE + "Hello.")
        svc.default_effort = "max"
        answer(svc, {"messages": ASK, "chat_template_kwargs": {"reasoning_effort": "no_think"}})
        self.assertIn(b"reasoning_effort:no_think", bytes(t for t in eng.prompts[0] if t < 256))
        answer(svc, {"messages": ASK})
        self.assertIn(b"reasoning_effort:high", bytes(t for t in eng.prompts[1] if t < 256))   # (any level: high)
        svc.default_effort = "low"
        answer(svc, {"messages": ASK})
        self.assertIn(b"reasoning_effort:high", bytes(t for t in eng.prompts[2] if t < 256))

    def test_tool_call(self):
        hy = CALL.replace("<tool_call>", f"<tool_calls{TAG}><tool_call{TAG}>").replace(
            "</tool_call>", f"</tool_call{TAG}></tool_calls{TAG}>")
        for name in ("arg_key", "arg_value"):
            hy = hy.replace(f"<{name}>", f"<{name}{TAG}>").replace(f"</{name}>", f"</{name}{TAG}>")
        svc, _, _ = service("Look it up." + CLOSE + hy)
        got = answer(svc, {"messages": ASK, "tools": TOOLS})
        self.assertEqual(got["calls"], [("get_weather", {"city": "Paris", "days": 2})])
        self.assertEqual(got["reasoning"], "Look it up.")

    def test_required_call_opens_in_the_models_own_tags(self):
        rest = f"<arg_key{TAG}>city</arg_key{TAG}><arg_value{TAG}>Rome</arg_value{TAG}></tool_call{TAG}></tool_calls{TAG}>"
        svc, eng, tok = service(rest)
        got = answer(svc, {"messages": ASK, "tools": TOOLS,
                           "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        want = tok.encode(f"{OPEN}{CLOSE}<tool_calls{TAG}><tool_call{TAG}>get_weather", parse_special=True)
        self.assertEqual(eng.prompts[0][-len(want):], want)
        self.assertEqual(got["calls"], [("get_weather", {"city": "Rome"})])

    def test_pictures_are_refused_in_plain_words(self):
        svc, eng, _ = service("x")
        self.assertEqual(svc.text_only(), server.HY4_TEXT_ONLY)
        msgs, tools, kw = openai_to_messages({"messages": [PICTURE]})
        with self.assertRaises(ValueError) as e:
            svc.prepare(msgs, tools, kw, 100)
        self.assertIn("Hy4-Flash with its pictures off reads text only", str(e.exception))
        self.assertEqual(eng.prompts, [])
        # ... a picture in a tool result (an agent read an image file): a short text note in its place, no error -
        # its template would write a picture's marker with nothing behind it
        req = {"messages": ASK + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "a map:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}}]}],
            "tools": TOOLS}
        msgs, tools, kw = openai_to_messages(req)
        noted = server.hy4_tool_pictures_noted(msgs)
        self.assertEqual(server.media_of(noted), [])
        self.assertEqual(len(server.media_of(msgs)), 1)               # (the request's own messages: as they were)
        self.assertIn({"type": "text", "text": server.HY4_PICTURE_NOTE}, noted[-1]["content"])
        svc.prepare(msgs, tools, kw, 100)                             # (no error: the agent's turn goes on)
        # a user's picture beside it: still the plain 400
        with self.assertRaises(ValueError):
            svc.prepare(msgs + openai_to_messages({"messages": [PICTURE]})[0], tools, kw, 100)

    def test_logprobs_and_bias_are_refused(self):   # (an engine whose INFO names neither)
        svc, _, _ = service("x")
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logprobs": True})
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logit_bias": {"65": 5}})

    def test_one_conversation_no_drafting(self):
        svc, eng, _ = service("x")
        self.assertFalse(getattr(eng, "multi", 0))
        self.assertEqual(server.speculation_modes(eng.info), [])

    def test_loop_guard_suffixes(self):
        # the loop guard's own prompts with this template: the turn kept going opens a thinking block (its suffixed
        # tag), and a blocked call is answered after the end of the model's <tool_calls> wrapper
        svc, _, tok = service("x")
        self.assertTrue(svc.loop_guard)
        flat = [t["function"] for t in TOOLS]
        ids, thinks = svc._keep_suffix(flat, {})
        self.assertTrue(thinks)
        self.assertEqual(ids[0], tok.special_tokens[HY_END])
        self.assertEqual(ids[-1], tok.special_tokens[OPEN])
        ids, thinks = svc._keep_suffix(flat, {"enable_thinking": False})
        self.assertFalse(thinks)
        from serve import loop_guard as LG
        key = ("get_weather", LG.args_key({"city": "Paris"}))
        out = svc._block_suffixes({key: LG.ToolRepeat("get_weather", 3)}, flat, {})
        self.assertIsNotNone(out)
        _, ids, thinks = out[key]
        self.assertTrue(thinks)
        self.assertEqual(ids[0], tok.special_tokens[HY_END])



class Http(unittest.TestCase):
    """What the page and a client see: /health's levels and images flag, a picture's 400, a level the template lacks."""

    @classmethod
    def setUpClass(cls):
        cls.svc, cls.eng, _ = service("Thinking." + CLOSE + "Hello.")
        cls.svc.default_effort = "high"
        cls.httpd = server.serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_health(self):
        code, h = self.call("/health")
        self.assertEqual(code, 200)
        self.assertEqual((h["images"], h["videos"], h.get("text_only")), (False, False, True))
        self.assertEqual(h["thinking_levels"], ["none", "high"])
        self.assertEqual(h["defaults"]["reasoning_effort"], "high")
        self.svc.default_effort = "low"                 # (a config's level this template lacks: what it runs at)
        try:
            self.assertEqual(self.call("/health")[1]["defaults"]["reasoning_effort"], "high")
        finally:
            self.svc.default_effort = "high"

    def test_chat_at_every_level(self):
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            code, out = self.call("/v1/chat/completions", {"messages": ASK, "reasoning_effort": level})
            self.assertEqual(code, 200, (level, out))
        code, out = self.call("/v1/chat/completions", {"messages": ASK})
        msg = out["choices"][0]["message"]
        self.assertEqual((msg.get("reasoning_content"), msg["content"]), ("Thinking.", "Hello."))

    def test_messages_and_responses(self):
        code, out = self.call("/v1/messages", {"model": "hy4-flash", "max_tokens": 200, "messages": ASK,
                                               "thinking": {"type": "enabled", "budget_tokens": 1024}})
        self.assertEqual(code, 200, out)
        code, out = self.call("/v1/responses", {"model": "hy4-flash", "input": "Hi", "reasoning": {"effort": "low"}})
        self.assertEqual(code, 200, out)

    def test_a_picture_is_a_400(self):
        code, out = self.call("/v1/chat/completions", {"messages": [PICTURE]})
        self.assertEqual(code, 400)
        self.assertIn("with its pictures off reads text only", json.dumps(out))

    def test_logprobs_are_a_400(self):
        code, out = self.call("/v1/chat/completions", {"messages": ASK, "logprobs": True})
        self.assertEqual(code, 400, out)


if __name__ == "__main__":
    unittest.main()
