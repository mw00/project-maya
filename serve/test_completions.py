"""serve/test_completions.py - POST /v1/completions (OpenAI's legacy completions, which editor autocomplete uses):
a raw prompt with no chat template and no thinking, an optional fill-in-the-middle `suffix` rendered with the model's
own markers, the text_completion answer and its SSE chunks - against the mock engine (no GPU, no pack).

    python -m unittest serve.test_completions -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS_FIM = ["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"]
DS_BOS = "<｜begin▁of▁sentence｜>"
QWEN_FIM = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]


class DeepSeekTok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + DS_FIM + [DS_BOS]


class QwenTok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + QWEN_FIM


class Recording(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        self.last_sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Completions(unittest.TestCase):
    TOK, SCRIPT = ByteTokenizer, "    return a + b\n"

    @classmethod
    def setUpClass(cls):
        cls.tok = cls.TOK()
        cls.engine = Recording(cls.tok, cls.SCRIPT)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body, path="/v1/completions"):
        rq = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(rq, timeout=30) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if not body.get("stream") else raw)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    def sse(self, raw):
        chunks = [json.loads(x[6:]) for x in raw.split("\n\n") if x.startswith("data: ") and x != "data: [DONE]"]
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))
        return chunks


class Raw(Completions):
    def test_raw_prompt_no_template(self):
        prompt = "def add(a, b):\n"
        s, r = self.post({"model": "m", "prompt": prompt, "max_tokens": 64})
        self.assertEqual(s, 200, r)
        self.assertEqual(r["object"], "text_completion")
        self.assertEqual(r["choices"][0]["text"], self.SCRIPT)
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertIsNone(r["choices"][0]["logprobs"])
        self.assertEqual(self.engine.last_prompt, self.tok.encode(prompt, parse_special=True))   # nothing around it
        self.assertEqual(r["usage"]["prompt_tokens"], len(prompt))
        self.assertEqual(r["usage"]["completion_tokens"], len(self.SCRIPT) + 1)                 # (+ the end token)

    def test_stream(self):
        s, raw = self.post({"model": "m", "prompt": "def add(a, b):\n", "max_tokens": 64, "stream": True})
        self.assertEqual(s, 200)
        chunks = self.sse(raw)
        self.assertTrue(all(c["object"] == "text_completion" for c in chunks))
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), self.SCRIPT)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", chunks[-1])
        self.assertEqual(len({c["id"] for c in chunks}), 1)

    def test_default_max_tokens_is_openais(self):
        s, r = self.post({"model": "m", "prompt": "x"})
        self.assertEqual(s, 200, r)
        self.assertEqual(self.engine.last_max_new, 16)
        self.assertEqual(r["choices"][0]["text"], self.SCRIPT[:16])
        self.assertEqual(r["choices"][0]["finish_reason"], "length")

    def test_stop_string(self):
        s, r = self.post({"model": "m", "prompt": "x", "max_tokens": 64, "stop": ["+"]})
        self.assertEqual(s, 200, r)
        self.assertEqual(r["choices"][0]["text"], "    return a ")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")

    def test_token_ids_and_batch_form(self):
        ids = self.tok.encode("abc")
        s, r = self.post({"model": "m", "prompt": ids, "max_tokens": 64})
        self.assertEqual(s, 200, r)
        self.assertEqual(self.engine.last_prompt, ids)
        s, r = self.post({"model": "m", "prompt": ["abc"], "max_tokens": 64})
        self.assertEqual(s, 200, r)
        self.assertEqual(self.engine.last_prompt, ids)

    def test_bad_requests(self):
        for body, word in (({"prompt": ""}, "prompt"), ({"prompt": ["a", "b"]}, "prompt"),
                           ({"prompt": "x", "echo": True}, "echo"), ({"prompt": "x", "n": 2}, "n"),
                           ({"prompt": "x", "logprobs": 2}, "logprobs"),
                           ({"prompt": "x", "suffix": "y"}, "suffix")):          # (ByteTokenizer: no FIM tokens)
            with self.subTest(body=body):
                s, r = self.post({"model": "m", **body})
                self.assertEqual(s, 400, r)
                self.assertIn(word, json.dumps(r))

    def test_chat_unchanged(self):
        s, r = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
                         "/v1/chat/completions")
        self.assertEqual(s, 200, r)
        self.assertEqual(r["object"], "chat.completion")

    def test_off_switch(self):
        with unittest.mock.patch.dict(os.environ, {"MAYA_COMPLETIONS": "0"}):
            s, r = self.post({"model": "m", "prompt": "x"})
        self.assertEqual(s, 404, r)


class DeepSeekFim(Completions):
    TOK = DeepSeekTok

    def test_suffix_rendered_with_deepseek_markers_and_bos(self):
        s, r = self.post({"model": "m", "prompt": "def add(a, b):\n", "suffix": "\nprint(add(1, 2))\n", "max_tokens": 64})
        self.assertEqual(s, 200, r)
        want = self.tok.encode(DS_BOS + DS_FIM[0] + "def add(a, b):\n" + DS_FIM[1] + "\nprint(add(1, 2))\n" + DS_FIM[2],
                               parse_special=True)
        self.assertEqual(self.engine.last_prompt, want)
        self.assertEqual(r["choices"][0]["text"], self.SCRIPT)

    def test_client_rendered_prompt(self):
        # Continue renders the prompt itself for an OpenAI-compatible server: the markers are its special tokens
        prompt = DS_FIM[0] + "def add(a, b):\n" + DS_FIM[1] + "\n" + DS_FIM[2]
        s, r = self.post({"model": "m", "prompt": prompt, "max_tokens": 64})
        self.assertEqual(s, 200, r)
        self.assertEqual(self.engine.last_prompt, self.tok.encode(DS_BOS + prompt, parse_special=True))

    def test_bos_not_doubled_and_add_special_false(self):
        s, _ = self.post({"model": "m", "prompt": DS_BOS + "x", "max_tokens": 4})
        self.assertEqual(self.engine.last_prompt, self.tok.encode(DS_BOS + "x", parse_special=True))
        s, _ = self.post({"model": "m", "prompt": "x", "max_tokens": 4, "add_special": False})
        self.assertEqual(self.engine.last_prompt, self.tok.encode("x", parse_special=True))

    def test_no_thinking(self):
        s, r = self.post({"model": "m", "prompt": "x", "max_tokens": 64})
        self.assertEqual(s, 200, r)
        self.assertFalse(self.engine.last_sampling.get("_think_budget"))


class QwenFim(Completions):
    TOK = QwenTok

    def test_suffix_rendered_with_qwen_markers(self):
        s, r = self.post({"model": "m", "prompt": "a", "suffix": "b", "max_tokens": 64})
        self.assertEqual(s, 200, r)
        self.assertEqual(self.engine.last_prompt,
                         self.tok.encode(QWEN_FIM[0] + "a" + QWEN_FIM[1] + "b" + QWEN_FIM[2], parse_special=True))


if __name__ == "__main__":
    unittest.main()
