"""serve/test_logprobs.py - OpenAI's logprobs / top_logprobs over the engine protocol (GEN logprobs=<n>, LP lines) and the
APIs: the request fields, the GEN key, the LP lines read on one conversation and on several (serve/fake_multi_engine.py),
the chat completion's choice logprobs (streamed and whole), the Responses API's output_text logprobs, and the refusal
by an engine that does not report them.  No GPU, no pack: the engines are scripted.

    python3 -m unittest serve.test_logprobs -v
"""
from __future__ import annotations

import io
import json
import queue
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.responses import normalize_request  # noqa: E402
from serve.server import (LOGPROB_FLOOR, ByteTokenizer, LpToken, MockEngine, Service, StrataEngine,  # noqa: E402
                          logprobs_of, parse_lp, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 8192
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a value",
                                           "parameters": {"type": "object",
                                                          "properties": {"key": {"type": "string"}}}}}]
CALL = "<tool_call>lookup<arg_key>key</arg_key><arg_value>alpha</arg_value></tool_call>"


class LpEngine(MockEngine):
    """The mock engine with made-up logprobs: token i of the answer -0.1 * (i + 1), its row the token and then
    id + 1, id + 2, ... each 1 lower; `own` replaces the token's own logprob (-inf: a token the model gave no chance)."""

    def __init__(self, *a, own=None, **kw):
        super().__init__(*a, **kw)
        self.own = own

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        n = (sampling or {}).get("top_logprobs") or 0
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if isinstance(t, LpToken):
                v = -0.1 * (i + 1) if self.own is None else self.own
                t.lp = (v, [(int(t) + j, v - j) for j in range(n)])
            yield t


class Fields(unittest.TestCase):
    """The request's fields: logprobs true asks (top_logprobs 0 by default), anything else is a 400 naming the field."""

    def test_asking(self):
        self.assertIsNone(logprobs_of({}))
        self.assertIsNone(logprobs_of({"logprobs": False}))
        self.assertEqual(logprobs_of({"logprobs": True}), 0)
        self.assertEqual(logprobs_of({"logprobs": True, "top_logprobs": 20}), 20)
        self.assertEqual(logprobs_of({"logprobs": True, "top_logprobs": "5"}), 5)
        self.assertEqual(logprobs_of({"logprobs": True, "top_logprobs": None}), 0)

    def test_refused(self):
        for bad in ({"logprobs": 1}, {"logprobs": "yes"}, {"top_logprobs": 3}, {"logprobs": False, "top_logprobs": 3},
                    {"logprobs": True, "top_logprobs": 21}, {"logprobs": True, "top_logprobs": -1},
                    {"logprobs": True, "top_logprobs": True}, {"logprobs": True, "top_logprobs": 2.5}):
            with self.assertRaises(ValueError, msg=bad):
                logprobs_of(bad)

    def test_gen_key(self):
        keys = StrataEngine.sampling_keys
        self.assertIn(" logprobs=5", keys({"logprobs": True, "top_logprobs": 5}))
        self.assertIn(" logprobs=0", keys({"logprobs": True, "temperature": 0}))   # (greedy's own return)
        self.assertIn(" logprobs=3", keys({"logprobs": True, "top_logprobs": 3, "temperature": 0.7}))
        for none in ({}, {"logprobs": False}, {"temperature": 0}, {"top_logprobs": 3}):
            self.assertNotIn("logprobs", keys(none), none)

    def test_lp_line(self):
        self.assertEqual(parse_lp("42 -0.25 42:-0.25 7:-1.609438".split()), (42, -0.25, [(42, -0.25), (7, -1.609438)]))
        got = parse_lp("9 -inf".split())
        self.assertEqual(got[0], 9)
        self.assertEqual(got[1], float("-inf"))
        self.assertEqual(got[2], [])
        self.assertIsNone(parse_lp(["9"]))                 # (cut short)
        self.assertIsNone(parse_lp("9 -1 7:".split()))


class OneConversation(unittest.TestCase):
    """StrataEngine.generate on one conversation: a token waits for the LP line after it; a token the engine wrote no
    LP line for (an older engine) goes on at the next line, without them; a request that did not ask is as before."""

    def engine(self, lines):
        e = StrataEngine.__new__(StrataEngine)
        e.proc = SimpleNamespace(stdin=io.StringIO())
        e.lines = queue.Queue()
        e.can_stop = False
        e.max_context = 4096
        for line in lines:
            e.lines.put(line + "\n")
        return e

    def test_lines(self):
        e = self.engine(["T 72", "LP 72 -0.5 72:-0.5 73:-1.5", "STAT tok_s=3", "T 105", "LP 105 -inf 104:-0.1",
                         "T 33", "DONE 3 1 1.0 1.0 stop 0 0 0"])
        toks = [t for t in e.generate([1], 3, {"logprobs": True, "top_logprobs": 2}, threading.Event())
                if t is not None]
        self.assertIn(" logprobs=2 ", e.proc.stdin.getvalue())
        self.assertEqual(toks, [72, 105, 33])
        self.assertEqual(toks[0].lp, (-0.5, [(72, -0.5), (73, -1.5)]))
        self.assertEqual(toks[1].lp[0], float("-inf"))
        self.assertEqual(toks[1].lp[1], [(104, -0.1)])
        self.assertIsNone(toks[2].lp)                      # (no LP line came for it)
        self.assertEqual(e.last["finish"], "stop")

    def test_not_asked(self):
        e = self.engine(["T 72", "T 73", "DONE 2 1 1.0 1.0 stop 0 0 0"])
        toks = [t for t in e.generate([1], 2, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [72, 73])
        self.assertFalse(any(isinstance(t, LpToken) for t in toks))
        self.assertNotIn("logprobs", e.proc.stdin.getvalue())

    def test_another_tokens_line_is_not_taken(self):
        e = self.engine(["T 72", "LP 99 -0.5", "DONE 1 1 1.0 1.0 stop 0 0 0"])
        toks = [t for t in e.generate([1], 1, {"logprobs": True}, threading.Event()) if t is not None]
        self.assertEqual(toks, [72])
        self.assertIsNone(toks[0].lp)


class SeveralConversations(unittest.TestCase):
    """--parallel's engine: LP <rid> lines go to their request's queue with its T lines; a request beside it that did
    not ask gets plain tokens (serve/fake_multi_engine.py writes LP lines only for a GEN with logprobs=)."""

    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["2"])

    def tearDown(self):
        self.engine.close()

    def test_routed(self):
        self.assertEqual(self.engine.info.get("logprobs"), 20)
        out = {}

        def one(k, sampling):
            ids = list(range(k * 100, k * 100 + 30))
            out[k] = [t for t in self.engine.generate(ids, 30, sampling, threading.Event()) if t is not None]

        ths = [threading.Thread(target=one, args=(0, {"logprobs": True, "top_logprobs": 3})),
               threading.Thread(target=one, args=(1, {}))]
        [t.start() for t in ths]
        [t.join(30) for t in ths]
        self.assertEqual(out[0], [i + 1000 for i in range(30)])
        self.assertEqual(out[1], [i + 1000 for i in range(100, 130)])
        for i, t in enumerate(out[0]):
            own = -0.25 * (i % 4)
            self.assertEqual(t.lp, (own, [(int(t) + j, own - j) for j in range(3)]), i)
        self.assertFalse(any(getattr(t, "lp", None) for t in out[1]))


class Http(unittest.TestCase):
    """The service over HTTP with LpEngine's answer "abc</think>Hello" (one token a byte)."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = LpEngine(self.tok, "abc</think>Hello", max_context=CTX)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode()
                status = r.status
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())
        if body.get("stream"):
            return status, [json.loads(line[6:]) for line in text.splitlines()
                            if line.startswith("data: ") and line != "data: [DONE]"]
        return status, json.loads(text)

    def chat(self, **kw):
        return {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **kw}


class Api(Http):
    """The chat completion's logprobs: an entry a token of the answer ({token, logprob, bytes, top_logprobs}), the
    reasoning's under reasoning_content, none for the end-of-thinking marker, a tool call or the end of turn; the
    stream's text chunks carry their own; nothing at all for a request that does not ask."""

    def test_whole(self):
        status, out = self.post("/v1/chat/completions", self.chat(logprobs=True, top_logprobs=2))
        self.assertEqual(status, 200, out)
        self.assertEqual(self.engine.sampling.get("top_logprobs"), 2)
        ch = out["choices"][0]
        self.assertEqual(ch["message"]["content"], "Hello")
        lp = ch["logprobs"]
        self.assertIsNone(lp["refusal"])
        self.assertEqual("".join(e["token"] for e in lp["content"]), "Hello")
        self.assertEqual("".join(e["token"] for e in lp["reasoning_content"]), "abc")
        # the answer's tokens are the script's 12th to 16th (after "abc" and "</think>"): their made-up logprobs
        self.assertEqual([e["logprob"] for e in lp["content"]], [-0.1 * i for i in range(12, 17)])
        for e in lp["content"] + lp["reasoning_content"]:
            self.assertEqual(e["bytes"], list(e["token"].encode()))
            self.assertEqual(len(e["top_logprobs"]), 2)
            self.assertEqual(e["top_logprobs"][0]["token"], e["token"])
            self.assertEqual(e["top_logprobs"][1]["logprob"], e["logprob"] - 1)
            self.assertEqual(set(e["top_logprobs"][1]), {"token", "logprob", "bytes"})

    def test_streamed(self):
        status, chunks = self.post("/v1/chat/completions", self.chat(logprobs=True, top_logprobs=1, stream=True))
        self.assertEqual(status, 200, chunks)
        content, reasoning = [], []
        for c in chunks:
            d, lp = c["choices"][0]["delta"], c["choices"][0].get("logprobs")
            if d.get("content"):
                self.assertEqual("".join(e["token"] for e in lp["content"]), d["content"])
                content += lp["content"]
            if d.get("reasoning_content"):
                self.assertEqual(lp["content"], [])
                reasoning += lp["reasoning_content"]
        self.assertEqual("".join(e["token"] for e in content), "Hello")
        self.assertEqual("".join(e["token"] for e in reasoning), "abc")
        _, whole = self.post("/v1/chat/completions", self.chat(logprobs=True, top_logprobs=1))
        self.assertEqual(whole["choices"][0]["logprobs"]["content"], content)

    def test_not_asked(self):
        _, out = self.post("/v1/chat/completions", self.chat())
        self.assertNotIn("logprobs", out["choices"][0])
        self.assertNotIn("logprobs", self.engine.sampling)
        _, chunks = self.post("/v1/chat/completions", self.chat(stream=True, logprobs=False))
        self.assertFalse(any("logprobs" in c["choices"][0] for c in chunks))

    def test_no_alternatives_and_no_chance(self):
        self.svc.engine = self.engine = LpEngine(self.tok, "</think>ok", max_context=CTX, own=float("-inf"))
        _, out = self.post("/v1/chat/completions", self.chat(logprobs=True))
        lp = out["choices"][0]["logprobs"]
        self.assertEqual([e["top_logprobs"] for e in lp["content"]], [[], []])
        self.assertEqual([e["logprob"] for e in lp["content"]], [LOGPROB_FLOOR, LOGPROB_FLOOR])   # (-inf: no JSON)
        self.assertNotIn("reasoning_content", lp)

    def test_tool_call_tokens_are_not_text(self):
        self.svc.engine = self.engine = LpEngine(self.tok, "</think>Hi" + CALL, max_context=CTX)
        _, out = self.post("/v1/chat/completions", self.chat(logprobs=True, tools=TOOLS, max_tokens=200))
        ch = out["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertEqual(ch["message"]["content"], "Hi")
        self.assertEqual("".join(e["token"] for e in ch["logprobs"]["content"]), "Hi")

    def test_json_mode(self):
        self.svc.engine = self.engine = LpEngine(self.tok, '</think>{"a": 1}', max_context=CTX)
        _, out = self.post("/v1/chat/completions", self.chat(logprobs=True, response_format={"type": "json_object"}))
        ch = out["choices"][0]
        self.assertEqual(json.loads(ch["message"]["content"]), {"a": 1})
        self.assertEqual("".join(e["token"] for e in ch["logprobs"]["content"]), '{"a": 1}')

    def test_refused(self):
        status, out = self.post("/v1/chat/completions", self.chat(top_logprobs=2))
        self.assertEqual(status, 400)
        self.assertIn("top_logprobs", out["error"]["message"])
        status, out = self.post("/v1/chat/completions", self.chat(logprobs=True, top_logprobs=30))
        self.assertEqual(status, 400)
        # an engine that does not say INFO logprobs= (GLM's, MiniMax's): a 400, not an answer without them
        self.engine.info = {"logit_bias": 1}
        status, out = self.post("/v1/chat/completions", self.chat(logprobs=True))
        self.assertEqual(status, 400)
        self.assertIn("log-probabilities", out["error"]["message"])
        status, out = self.post("/v1/chat/completions", self.chat())   # (not asked: served as ever)
        self.assertEqual(status, 200)


class Responses(Http):
    """The Responses API: include "message.output_text.logprobs" (or a top_logprobs) asks; the message's output_text
    part and its deltas carry the answer's entries, the reasoning item none; without asking, logprobs [] as before."""

    def respond(self, **kw):
        return self.post("/v1/responses", {"model": "m", "input": "hi", "max_output_tokens": 64, **kw})

    def test_normalized(self):
        n = normalize_request({"input": "hi", "include": ["message.output_text.logprobs"]})
        self.assertEqual((n["logprobs"], n["top_logprobs"]), (True, 0))
        n = normalize_request({"input": "hi", "top_logprobs": 4})
        self.assertEqual((n["logprobs"], n["top_logprobs"]), (True, 4))
        self.assertNotIn("logprobs", normalize_request({"input": "hi"}))
        for bad in ({"top_logprobs": 21}, {"top_logprobs": "2"}, {"include": ["message.output_text.logprobs", "x"]}):
            with self.assertRaises(ValueError, msg=bad):
                normalize_request({"input": "hi", **bad})

    def test_whole_response(self):
        status, out = self.respond(include=["message.output_text.logprobs"], top_logprobs=2)
        self.assertEqual(status, 200, out)
        self.assertEqual(out["top_logprobs"], 2)
        msg = next(o for o in out["output"] if o["type"] == "message")
        part = msg["content"][0]
        self.assertEqual(part["text"], "Hello")
        self.assertEqual("".join(e["token"] for e in part["logprobs"]), "Hello")
        self.assertTrue(all(len(e["top_logprobs"]) == 2 for e in part["logprobs"]))
        self.assertFalse(any("logprobs" in c for o in out["output"] if o["type"] == "reasoning" for c in o["content"]))

    def test_streamed_response(self):
        req = urllib.request.Request(self.base + "/v1/responses", data=json.dumps(
            {"model": "m", "input": "hi", "stream": True, "top_logprobs": 1}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            evs = [json.loads(line[6:]) for line in r.read().decode().splitlines() if line.startswith("data: ")]
        deltas = [e for e in evs if e["type"] == "response.output_text.delta"]
        self.assertEqual("".join(t["token"] for e in deltas for t in e["logprobs"]), "Hello")
        for e in deltas:
            self.assertEqual("".join(t["token"] for t in e["logprobs"]), e["delta"])
        done = next(e for e in evs if e["type"] == "response.output_text.done")
        self.assertEqual(done["logprobs"], [t for e in deltas for t in e["logprobs"]])

    def test_not_asked_response(self):
        status, out = self.respond()
        self.assertEqual(status, 200, out)
        msg = next(o for o in out["output"] if o["type"] == "message")
        self.assertEqual(msg["content"][0]["logprobs"], [])
        self.assertNotIn("top_logprobs", out)
        self.assertNotIn("logprobs", self.engine.sampling)


if __name__ == "__main__":
    unittest.main()
