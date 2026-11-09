"""serve/test_glm_dsa_think_budget.py - GLM-5.3's thinking budget (the owner, 2026-11-07: "normal use stays quick, people
who ask for max effort get the long budget"): the budget stays 32768 by default - also at Max, GLM-5.3's default level -
and a request that asks for Max itself may think up to 65536 tokens while the budget is that default, as far as its
room allows (its max_tokens, or the context left, keeps 32768 for the answer).  A budget the user set applies at every
level, and GLM-5.3-Flash, DeepSeek, MiniMax and Qwen keep one budget for every level.  No GPU, no model:

  Level:   ChatTemplate.think_level is the level GLM-5.3's prompt is rendered at ("Reasoning Effort: X") for every
           spelling a client sends, auto and none named; None with the thinking off and for every other model
  Start:   maya_quants.thinking_budget_is_default and server.think_budget_at_max (what main gives the service): 65536
           for a GLM-5.3 config with the default budget, 0 for a budget the user set and for every other template
  Engine:  the GEN line's think_budget (the real server, serve/glm_dsa_fake_engine.py): an explicit Max (OpenAI,
           Anthropic, Responses, chat_template_kwargs, the MCP tool loop) gets the long budget; no level, auto, the run
           config's level, the shared Chat settings' level, Maya's model API (it names none) and the web chat's level left at its
           default (strata_level_default) do not; the long budget as far as the room allows; a tiny max_tokens's
           reserve; the user's own budget (the dashboard's field); private keys a client sends of its own
  Others:  GLM-5.3-Flash's, DeepSeek's, MiniMax's and Qwen's templates hand run() no ask and get the same budget at
           every level, even on a service with the long budget set; a GLM-5.3-Flash server over HTTP likewise
  Web:     the web chat (serve/web/chats.js, in Node) marks a level its user did not pick, for GLM-5.3 only
  main():  a GLM-5.3 run config without a budget says and serves the long budget to an explicit Max (until the
           dashboard's field sets one); with the user's own budget, and a GLM-5.3-Flash config, everything is as before

    python -m unittest serve.test_glm_dsa_think_budget -v
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import maya_quants as Q  # noqa: E402
from serve import glm_dsa_product as G53  # noqa: E402
from serve import server as SV  # noqa: E402
from serve.frontend import ChatTemplate, deepseek_effort_kwargs, effort_kwargs  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.server import MockEngine, Service, StrataEngine, serve  # noqa: E402
from serve.test_glm_dsa_serve import (FLASH_TEMPLATE, GLM52, OFFICIAL, WIN, fake_exe, free_port,  # noqa: E402
                                      glm53_config, glm53_pack, glm_tokenizer, glm_tokenizer_dir, system_line)

ROOT = Path(__file__).resolve().parents[1]
GGUF_TEMPLATE = OFFICIAL.parent / "glm-5.3.gguf-ud-iq2_m.jinja"
OTHERS = (("GLM-5.3-Flash", FLASH_TEMPLATE), ("GLM (older template)", ROOT / "tools/glm_chat_template.jinja"),
          ("DeepSeek-V4.1", ROOT / "tools/ds4/chat_template.jinja"),
          ("MiniMax-M3.1", ROOT / "tools/mm3/chat_template.jinja"), ("Qwen", ROOT / "serve/chat_template.jinja"))
DEFAULT, AT_MAX, GRACE, RESERVE = 32768, 65536, SV.THINK_GRACE, SV.THINK_LONG_RESERVE
BIG = 100_000                    # a max_tokens with room for the whole long budget and the answer's 32768
HIGH, MAX = DEFAULT + GRACE, AT_MAX + GRACE   # what the engine is sent for it: the budget + its grace (the soft close)
CONTEXT = 262144
# a client's spelling -> the level GLM-5.3's template renders (None: the thinking off)
LEVELS = {"low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max", "maximum": "max", "auto": "max",
          "default": "max", None: "max", "none": None, "minimal": None, False: None}


def budget_of(gen_line: str) -> int | None:
    """A GEN line's think_budget (None: the request sent none)."""
    keys = dict(x.split("=", 1) for x in gen_line.split()[2:-1] if "=" in x)
    return int(keys["think_budget"]) if "think_budget" in keys else None


def tokenizer_with(d: Path, extra: list[str]):
    """glm_tokenizer's byte-level tokenizer with more special tokens (MiniMax's think block's end)."""
    import strata_tokenizer as ST
    glm_tokenizer_dir(d)
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    types = json.loads((d / "token_type.json").read_text(encoding="utf-8"))
    for t in extra:
        vocab[t] = len(vocab)
        types.append(4)
    (d / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (d / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    return ST.Tokenizer.from_dir(d)


class Recorder(MockEngine):
    """The mock engine, keeping the sampling each request sent it (the GEN line's keys come from it)."""

    def __init__(self, tok, context=CONTEXT):
        super().__init__(tok, "Thought.</think>Done.", max_context=context)
        self.sent: list[dict] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sent.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def run_once(svc, kw, max_new, extra=None) -> dict:
    """One request through Service.prepare and Service.run as a handler makes it (prepare's _think_max handed to
    run() in the request's sampling): the sampling the engine got."""
    ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, kw, max_new)
    list(svc.run(ids, thinking, None, max_new, {"_think_max": kw.get("_think_max"), **(extra or {})},
                 threading.Event()))
    return svc.engine.sent[-1]


# ------------------------------------------------------------------------------------------------ the level
class Level(unittest.TestCase):
    def test_glm53_templates_render_the_level_it_names(self):
        for path in (OFFICIAL, GGUF_TEMPLATE, GLM52):      # (GLM-5.2's wording: high / max, its dialect's nearest)
            tpl = ChatTemplate(path, glm_dsa=True)
            for spelling, want in LEVELS.items():
                with self.subTest(template=path.name, effort=spelling):
                    kw = effort_kwargs(spelling)
                    level = tpl.think_level(dict(kw))
                    prompt = tpl.render([{"role": "user", "content": "hi"}], **kw)
                    if want is None:
                        self.assertIsNone(level)
                        self.assertTrue(prompt.endswith("</think>"), prompt[-40:])   # (the thinking closed)
                    else:
                        self.assertEqual(level, "high" if path == GLM52 and want == "low" else want)
                        self.assertEqual(system_line(prompt), level.capitalize())

    def test_every_other_model_has_none(self):
        for name, path in OTHERS:
            tpl = ChatTemplate(path)
            for spelling in LEVELS:
                with self.subTest(model=name, effort=spelling):
                    self.assertIsNone(tpl.think_level(effort_kwargs(spelling)))
        # (and GLM-5.3's own templates without the glm-dsa flag: the server sets it from the pack)
        self.assertIsNone(ChatTemplate(OFFICIAL).think_level({"reasoning_effort": "max"}))


# ------------------------------------------------------------------------------------------------ the start
class Start(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.g53 = glm53_config(glm53_pack(Path(tmp.name) / "pack"))
        self.g53.pop("thinking_budget")                  # (the setup writes none for GLM-5.3: its default)
        self.flash = {"exe": "x/maya", "args": ["--glm-pack", str(Path(tmp.name) / "flash")],
                      "installer": {"quant": "Maya-S24"}}

    def test_default_or_the_users_own(self):
        self.assertEqual((Q.model_key(self.g53), Q.default_thinking_budget(self.g53)), ("glm53", DEFAULT))
        self.assertEqual((Q.GLM53_MAX_THINK_BUDGET, RESERVE), (AT_MAX, DEFAULT))
        self.assertTrue(Q.thinking_budget_is_default(self.g53))                              # none set
        self.assertTrue(Q.thinking_budget_is_default({**self.g53, "thinking_budget": DEFAULT}))
        for own in (12288, AT_MAX, 0):                    # another value in the config: the user's
            self.assertFalse(Q.thinking_budget_is_default({**self.g53, "thinking_budget": own}), own)
        self.assertFalse(Q.thinking_budget_is_default({**self.g53, "thinking_budget": DEFAULT,
                                                       "installer": {"thinking_budget_set": True}}))   # the dashboard's
        self.assertFalse(Q.thinking_budget_is_default({**self.g53, "installer": {"thinking_budget_set": True}}))

    def test_main_gives_it_to_glm53_only(self):
        for path in (OFFICIAL, GGUF_TEMPLATE):
            tpl = ChatTemplate(path, glm_dsa=True)
            self.assertEqual(SV.think_budget_at_max(tpl, self.g53, DEFAULT), AT_MAX)
            self.assertEqual(SV.think_budget_at_max(tpl, {**self.g53, "installer": {"thinking_budget_set": True},
                                                          "thinking_budget": DEFAULT}, DEFAULT), 0)
            self.assertEqual(SV.think_budget_at_max(tpl, {**self.g53, "thinking_budget": 12288}, 12288), 0)
            self.assertEqual(SV.think_budget_at_max(tpl, self.g53, 0), 0)       # (no </think> token: no budget)
        for name, path in OTHERS:                         # every other model: never, whatever its config says
            for cfg in (self.flash, self.g53, {}):
                self.assertEqual(SV.think_budget_at_max(ChatTemplate(path), cfg, DEFAULT), 0, name)
        # the budgets every model starts with are unchanged (GLM-5.3-Flash Maya-XS 8192, every other 32768)
        self.assertEqual(Q.default_thinking_budget(self.flash), DEFAULT)
        self.assertEqual(Q.default_thinking_budget({**self.flash, "installer": {"quant": "Maya-XS-IQ1_S"}}), 8192)


# ------------------------------------------------------------------------------------------------ the engine's budget
class _Server(unittest.TestCase):
    """One fake engine and one real server for the class (serve/test_glm_dsa_api.py's, with a context that has room
    for the long budget); each test starts from the budgets main gives a config with the default one."""
    TEMPLATE, GLM_DSA = OFFICIAL, True

    @classmethod
    def setUpClass(cls):
        env_patch = mock.patch.dict(os.environ, {"STRATA_GPU_RELEASE_S": "0"})
        env_patch.start()
        cls.addClassCleanup(env_patch.stop)
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=WIN)
        cls.addClassCleanup(tmp.cleanup)
        d = Path(tmp.name)
        cls.tok = glm_tokenizer(d / "tok")
        cls.script_file, cls.gen_log = d / "script.json", d / "gen.txt"
        cls.script_file.write_text("[[]]", encoding="utf-8")
        env = dict(os.environ, FAKE_ENGINE_SCRIPT=str(cls.script_file), FAKE_ENGINE_GEN_LOG=str(cls.gen_log),
                   FAKE_ENGINE_DIR=str(d))
        cls.engine = StrataEngine(fake_exe(d), ["--glm-pack", str(d), "--max-context", str(CONTEXT)], env=env,
                                  log=str(d / "engine.log"))
        cls.addClassCleanup(cls.engine.log.close)
        cls.addClassCleanup(cls.engine.close)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(cls.TEMPLATE, glm_dsa=cls.GLM_DSA),
                          model_name="glm-5.3" if cls.GLM_DSA else "glm-5.3-flash")
        cls.svc.model_management = True                  # (Maya's model API, /api/chat)
        cls.httpd = serve(cls.svc, port=0)
        cls.addClassCleanup(cls.httpd.server_close)
        cls.addClassCleanup(cls.httpd.shutdown)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    def setUp(self):
        svc = self.svc
        saved = (svc.think_budget, svc.think_budget_max, svc.default_effort, svc.shared)

        def restore():
            svc.think_budget, svc.think_budget_max, svc.default_effort, svc.shared = saved
        self.addCleanup(restore)
        svc.think_budget = DEFAULT                        # (main's for a config without one)
        svc.think_budget_max = SV.think_budget_at_max(svc.template, {}, DEFAULT)

    def script(self, *answers: str):
        ids = [self.tok.encode(a + ("<|observation|>" if a.endswith("</tool_call>") else "<|user|>"),
                               parse_special=True) for a in answers]
        self.script_file.write_text(json.dumps(ids), encoding="utf-8")

    def request(self, path, body=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.Request(self.base + path, data=data, method=method,
                                      headers={"Content-Type": "application/json"})

    def post(self, path, body):
        try:
            with urllib.request.urlopen(self.request(path, body), timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def get(self, path):
        with urllib.request.urlopen(self.request(path), timeout=60) as r:
            return json.loads(r.read())

    def gen_lines(self) -> list[str]:
        return self.gen_log.read_text(encoding="utf-8").splitlines() if self.gen_log.exists() else []

    def sent(self, path, body, answer="Thought.</think>Done.", engine_calls=1) -> list[int | None]:
        """The think_budget of each GEN line one request sends the engine (it must answer: 200)."""
        n = len(self.gen_lines())
        self.script(answer)
        code, b = self.post(path, body)
        self.assertEqual(code, 200, b)
        lines = self.gen_lines()[n:]
        self.assertEqual(len(lines), engine_calls, lines)
        return [budget_of(x) for x in lines]

    def chat(self, **body) -> int | None:
        return self.sent("/v1/chat/completions", {"model": "glm-5.3", "max_tokens": BIG,
                                                  "messages": [{"role": "user", "content": "think"}], **body})[0]


class EngineBudget(_Server):
    def test_only_an_explicit_max_gets_the_long_budget(self):
        self.assertEqual(self.svc.think_budget_max, AT_MAX)
        for effort, want in (("max", MAX), ("xhigh", MAX), ("maximum", MAX),
                             (None, HIGH), ("auto", HIGH),     # (no level, or the model's own: Max, at the default)
                             ("high", HIGH), ("medium", HIGH), ("low", HIGH), ("none", None)):
            with self.subTest(effort=effort):
                self.assertEqual(self.chat(**({} if effort is None else {"reasoning_effort": effort})), want)
        # chat_template_kwargs (vLLM's / llama.cpp's way) alike
        self.assertEqual(self.chat(chat_template_kwargs={"reasoning_effort": "max"}), MAX)
        self.assertEqual(self.chat(chat_template_kwargs={"reasoning_effort": "low"}), HIGH)
        self.assertIsNone(self.chat(chat_template_kwargs={"enable_thinking": False}))

    def test_the_run_configs_level_is_not_an_ask(self):
        for config, want in (("max", HIGH), ("xhigh", HIGH), ("high", HIGH), ("low", HIGH), ("auto", HIGH),
                             ("none", None)):
            with self.subTest(config=config):
                self.svc.default_effort = config
                self.assertEqual(self.chat(), want)
        self.svc.default_effort = "high"                  # (the request's own Max is one)
        self.assertEqual(self.chat(reasoning_effort="max"), MAX)
        self.svc.default_effort = "max"
        self.assertEqual(self.chat(reasoning_effort="max"), MAX)
        self.assertEqual(self.chat(reasoning_effort="high"), HIGH)

    def test_the_long_budget_as_far_as_the_room_allows(self):
        # max_tokens -> what the engine is sent for an explicit Max (the soft budget + the grace) and for High: the long
        # budget keeps RESERVE of the room for the answer and is never below what every level gets
        for limit, at_max, at_high in ((BIG, MAX, HIGH), (AT_MAX + RESERVE, MAX, HIGH),
                                       (80000, 80000 - RESERVE + GRACE, HIGH), (AT_MAX + 1, DEFAULT + 1 + GRACE, HIGH),
                                       (64000, HIGH, HIGH), (40000, HIGH, HIGH), (DEFAULT, DEFAULT, DEFAULT),
                                       (1000, 499, 499)):
            with self.subTest(max_tokens=limit):
                got = self.chat(reasoning_effort="max", max_tokens=limit)
                self.assertEqual((got, self.chat(reasoning_effort="high", max_tokens=limit)), (at_max, at_high))
                soft = got - GRACE if got > DEFAULT + 1 else got
                self.assertTrue(soft <= DEFAULT or limit - soft >= RESERVE, (limit, soft))   # (room for the answer)

    def test_a_tiny_limit_still_keeps_the_answers_reserve(self):
        import random
        rng = random.Random(7)
        thought = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(900))
        for effort in ("max", "high"):
            n = len(self.gen_lines())
            self.script(thought + "</think>Done.")
            code, b = self.post("/v1/chat/completions", {"model": "glm-5.3", "max_tokens": 1000, "reasoning_effort": effort,
                                                         "messages": [{"role": "user", "content": "think long"}]})
            self.assertEqual(code, 200, b)
            # (1000 - a reserve of 500 - the closing token: as at every level, and as before)
            self.assertEqual(budget_of(self.gen_lines()[n]), 499, effort)
            msg = b["choices"][0]["message"]
            self.assertEqual(msg["reasoning_content"], thought[:499])
            self.assertEqual((msg["content"], b["choices"][0]["finish_reason"]), ("Done.", "stop"))

    def test_the_users_own_budget_applies_at_every_level(self):
        self.assertEqual(self.get("/api/thinking-budget"),
                         {"budget": DEFAULT, "supported": True, "saved": False, "max_level": AT_MAX})
        self.assertEqual(self.get("/health")["thinking_budget_max"], AT_MAX)
        code, out = self.post("/api/thinking-budget", {"budget": 20000})   # (the dashboard's field)
        self.assertEqual((code, out), (200, {"budget": 20000, "supported": True, "saved": False}))
        self.assertEqual(self.get("/api/thinking-budget"), {"budget": 20000, "supported": True, "saved": False})
        self.assertNotIn("thinking_budget_max", self.get("/health"))
        for effort in ("max", "high", None):
            self.assertEqual(self.chat(**({} if effort is None else {"reasoning_effort": effort})), 20000 + GRACE)
        self.svc.set_think_budget(DEFAULT)                # the default's number, set by the user: every level too
        self.assertEqual(self.chat(reasoning_effort="max"), HIGH)
        self.svc.set_think_budget(0)                      # no limit: none sent at any level
        self.assertIsNone(self.chat(reasoning_effort="max"))

    def test_private_keys_a_client_sends_of_its_own(self):
        # (_think_max is the server's own, always set by prepare; the others can only take the long budget away)
        self.assertEqual(self.chat(reasoning_effort="high", _think_max=True), HIGH)
        self.assertEqual(self.chat(_think_max=True), HIGH)
        self.assertIsNone(self.chat(reasoning_effort="none", _think_max=True))
        self.assertEqual(self.chat(reasoning_effort="max", _think_max=False), MAX)
        self.assertEqual(self.chat(reasoning_effort="max", _level_shared=True), HIGH)
        self.assertEqual(self.chat(reasoning_effort="max", strata_level_default=True), HIGH)

    def test_the_web_chats_level_left_at_its_default(self):
        # the page sends its level with every request; one its user did not pick it marks (serve/web/chats.js)
        web = {"strata_resume": True, "stream": False}
        self.assertEqual(self.chat(reasoning_effort="max", **web), MAX)
        self.assertEqual(self.chat(reasoning_effort="max", strata_level_default=True, **web), HIGH)
        self.assertEqual(self.chat(reasoning_effort="high", strata_level_default=True, **web), HIGH)

    def test_anthropic(self):
        def ask(**body):
            return self.sent("/v1/messages", {"model": "glm-5.3", "max_tokens": BIG,
                                              "messages": [{"role": "user", "content": "think"}], **body})[0]
        self.assertEqual(ask(output_config={"effort": "max"}), MAX)
        self.assertEqual(ask(output_config={"effort": "high"}), HIGH)
        self.assertEqual(ask(thinking={"type": "enabled", "budget_tokens": 16000}), MAX)    # (8K and up: Max)
        self.assertEqual(ask(thinking={"type": "enabled", "budget_tokens": 4000}), HIGH)
        self.assertIsNone(ask(thinking={"type": "disabled"}))
        self.assertEqual(ask(), HIGH)                     # (none named: the model's own, Max, at the default budget)

    def test_responses(self):
        def ask(**body):
            return self.sent("/v1/responses", {"model": "glm-5.3", "input": "think", "max_output_tokens": BIG,
                                               **body})[0]
        self.assertEqual(ask(reasoning={"effort": "max"}), MAX)
        self.assertEqual(ask(reasoning={"effort": "xhigh"}), MAX)
        self.assertEqual(ask(reasoning={"effort": "low"}), HIGH)
        self.assertIsNone(ask(reasoning={"effort": "none"}))
        self.assertEqual(ask(), HIGH)

    def test_model_api(self):
        def ask(**body):
            return self.sent("/api/chat", {"stream": False, "options": {"num_predict": BIG},
                                           "messages": [{"role": "user", "content": "think"}], **body})[0]
        self.assertEqual(ask(), HIGH)                     # (its API names no level here: the run config's, or Max)
        self.svc.default_effort = "max"
        self.assertEqual(ask(), HIGH)

    def test_the_shared_chat_settings_level_is_not_an_ask(self):
        for shared, want in (("max", HIGH), ("high", HIGH), ("none", None)):
            self.svc.shared = {"reasoning_effort": shared}
            self.assertEqual(self.chat(), want, shared)
            self.assertEqual(self.sent("/v1/messages", {"model": "glm-5.3", "max_tokens": BIG,
                                                        "messages": [{"role": "user", "content": "x"}]})[0], want)
        self.svc.shared = {"reasoning_effort": "high"}
        self.assertEqual(self.chat(reasoning_effort="max"), MAX)      # (the request's own Max is one)
        self.svc.shared = {"reasoning_effort": "max"}
        self.assertEqual(self.chat(reasoning_effort="max"), MAX)
        self.assertIs(self.svc.with_shared({"messages": []}, "openai")["_level_shared"], True)
        self.assertNotIn("_level_shared", self.svc.with_shared({"reasoning_effort": "max"}, "openai"))

    def test_several_requests_at_once_keep_their_own_levels(self):
        # (the ask rides in each request's own sampling: nothing per request lives on the service)
        results, errors = {}, []

        def one(i, effort):
            try:
                body = {"model": "glm-5.3", "max_tokens": BIG, "reasoning_effort": effort,
                        "messages": [{"role": "user", "content": f"think {i}"}]}
                results[i] = self.post("/v1/chat/completions", body)[0]
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        n = len(self.gen_lines())
        self.script("Thought.</think>Done.")
        efforts = ["max", "high"] * 4
        threads = [threading.Thread(target=one, args=(i, e)) for i, e in enumerate(efforts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        self.assertEqual((errors, sorted(results.values())), ([], [200] * len(efforts)))
        lines = self.gen_lines()[n:]
        self.assertEqual(len(lines), len(efforts))
        by_prompt = {}
        for line in lines:
            text = self.tok.decode([int(x) for x in line.split()[-1].split(",") if x])
            by_prompt[int(re.search(r"think (\d+)<", text).group(1))] = budget_of(line)
        self.assertEqual(by_prompt, {i: MAX if e == "max" else HIGH for i, e in enumerate(efforts)})


class EngineBudgetGgufTemplate(EngineBudget):
    """The same with the template the UD-IQ2_M GGUF carries (its pack is served with it)."""
    TEMPLATE = GGUF_TEMPLATE


class ContextLeft(unittest.TestCase):
    """No max_tokens: the room is the context left - a 64K context keeps the default budget, 128K has room for the
    long one (Service.prepare and run with the mock engine)."""

    def test_the_contexts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tok = glm_tokenizer(Path(tmp.name) / "tok")
        # (a 32K context: no room for the grace either - as every level, and as before)
        for context, at_max, other in ((32768, DEFAULT, DEFAULT), (65536, HIGH, HIGH), (131072, MAX, HIGH),
                                       (CONTEXT, MAX, HIGH)):
            with self.subTest(context=context):
                svc = Service(Recorder(tok, context), tok, ChatTemplate(OFFICIAL, glm_dsa=True))
                svc.think_budget, svc.think_budget_max = DEFAULT, AT_MAX
                self.assertEqual(run_once(svc, effort_kwargs("max"), 0).get("_think_budget"), at_max)
                self.assertEqual(run_once(svc, effort_kwargs("high"), 0).get("_think_budget"), other)
                self.assertEqual(run_once(svc, {}, 0).get("_think_budget"), other)   # (no level: Max, the default)
        # between the two: as far as the room allows, the answer keeping RESERVE of it
        svc = Service(Recorder(tok, 90000), tok, ChatTemplate(OFFICIAL, glm_dsa=True))
        svc.think_budget, svc.think_budget_max = DEFAULT, AT_MAX
        ids, _, room = svc.prepare([{"role": "user", "content": "hi"}], None, effort_kwargs("max"), 0)
        self.assertEqual(run_once(svc, effort_kwargs("max"), 0).get("_think_budget"), room - RESERVE + GRACE)


class McpLoop(_Server):
    """The web app's MCP tool loop (run_with_mcp): every engine call of one answer gets the same budget."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hub = McpHub({"fake": {"command": sys.executable, "args": [str(ROOT / "serve/mcp_fake_server.py")]}},
                         {"timeout_s": 10, "max_result_chars": 500, "max_rounds": 2})
        cls.hub.start(wait=True)
        cls.addClassCleanup(cls.hub.close)
        cls.svc.mcp = cls.hub

    def test_every_round(self):
        for i, (extra, want) in enumerate((({"reasoning_effort": "max"}, MAX), ({"reasoning_effort": "high"}, HIGH),
                                           ({}, HIGH), ({"reasoning_effort": "max", "strata_level_default": True}, HIGH))):
            n = len(self.gen_lines())
            # (a script of its own each time: the fake engine starts it again only when the file changes)
            self.script(f"Let me look.</think><tool_call>fake__echo<arg_key>text</arg_key><arg_value>{i}"
                        "</arg_value></tool_call>", "Got it.</think>The tool said hello.")
            code, b = self.post("/v1/chat/completions", {
                "model": "glm-5.3", "strata_mcp": True, "stream": False, "max_tokens": BIG,
                "messages": [{"role": "user", "content": "check it"}], **extra})
            self.assertEqual(code, 200, b)
            self.assertTrue(b["choices"][0]["message"]["content"].endswith("The tool said hello."), b)
            self.assertEqual([budget_of(x) for x in self.gen_lines()[n:]], [want, want], extra)


# ------------------------------------------------------------------------------------------------ every other model
class OtherModels(unittest.TestCase):
    def test_one_budget_for_every_level(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tok = tokenizer_with(Path(tmp.name) / "tok", ["<mm:think>", "</mm:think>"])
        for name, path in OTHERS:
            tpl = ChatTemplate(path)
            svc = Service(Recorder(tok), tok, tpl)
            self.assertIsNotNone(svc.think_end_id, name)
            # (the long budget set on the service anyway - main never does for these: Start - and still unused)
            svc.think_budget, svc.think_budget_max = DEFAULT, AT_MAX
            for effort in ("max", "xhigh", "auto", None, "high", "low"):
                with self.subTest(model=name, effort=effort):
                    kw = (deepseek_effort_kwargs if tpl.deepseek else effort_kwargs)(effort)
                    self.assertEqual(run_once(svc, kw, BIG).get("_think_budget"), HIGH)
                    self.assertNotIn("_think_max", kw)
            svc.shared = {"reasoning_effort": "max"}
            self.assertNotIn("_level_shared", svc.with_shared({"messages": []}, "openai"))
            self.assertNotIn("_level_shared", svc.with_shared({}, "anthropic"))


class FlashServer(_Server):
    """GLM-5.3-Flash over HTTP: the same budget at Max as at High; /health and the dashboard's field as before."""
    TEMPLATE, GLM_DSA = FLASH_TEMPLATE, False

    def test_max_is_as_before(self):
        self.assertEqual(self.svc.think_budget_max, 0)    # (main's for Flash: think_budget_at_max)
        self.assertEqual(self.get("/api/thinking-budget"), {"budget": DEFAULT, "supported": True, "saved": False})
        self.assertNotIn("thinking_budget_max", self.get("/health"))
        self.svc.think_budget_max = AT_MAX                # (set anyway: no Flash request reaches it)
        for effort in ("max", "xhigh", None, "high", "low"):
            self.assertEqual(self.chat(**({} if effort is None else {"reasoning_effort": effort})), HIGH, effort)
        self.assertEqual(self.chat(reasoning_effort="max", _think_max=True), HIGH)
        self.assertIsNone(self.chat(reasoning_effort="none"))


# ------------------------------------------------------------------------------------------------ the web chat
# serve/web/chats.js in Node: whether a chat request marks its level as one its user did not pick
CHATS_LEVEL_JS = r"""
import {pathToFileURL} from "node:url";
const noop = () => {};
const mem = {};
globalThis.document = {addEventListener: noop, hidden: false, getElementById: () => null};
globalThis.window = globalThis;
globalThis.matchMedia = () => ({matches: false, addEventListener: noop});
globalThis.localStorage = {getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); },
                           removeItem: (k) => { delete mem[k]; }};
globalThis.location = {origin: "http://x", href: "http://x/", search: "", hash: ""};
const [web, arg] = process.argv.slice(2);
const {health} = JSON.parse(arg);
const st = await import(pathToFileURL(web + "/state.js").href);
const {levelLeftAtDefault, levelPicked, newChatObj, setThinking, thinkingOf} = await import(pathToFileURL(web + "/chats.js").href);
const S = st.S;
S.health = health; S.attachments = []; S.toasts = []; S.draft = ""; S.editIndex = null;
st.setFamily(health.model); st.loadPrefs();
const out = {};
const look = (name, c) => { out[name] = {level: thinkingOf(c), picked: levelPicked(c), left: levelLeftAtDefault(c)}; };
S.current = newChatObj(); look("new", S.current);
setThinking("max"); look("tapped_max", S.current);
S.current = newChatObj(); setThinking("high"); setThinking("max"); look("high_then_max", S.current);
S.current = newChatObj(); setThinking("high"); look("tapped_high", S.current);
look("other_model", {messages: [], thinking: "high", thinkingFor: "other|x", thinkingPicked: true});
st.savePrefs({thinking: "max"});                                     // Settings > Chat defaults > Thinking: Max
S.current = newChatObj(); look("settings_max", S.current);
console.log(JSON.stringify(out));
process.exit(0);
"""


# ... and its send(): the request it makes (a stub fetch keeps the body and answers an error, which ends the turn)
CHATS_SEND_JS = r"""
import {pathToFileURL} from "node:url";
const noop = () => {};
const mem = {};
globalThis.document = {addEventListener: noop, hidden: false, getElementById: () => null, querySelector: () => null};
globalThis.window = globalThis;
globalThis.matchMedia = () => ({matches: false, addEventListener: noop});
globalThis.localStorage = {getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); },
                           removeItem: (k) => { delete mem[k]; }};
globalThis.location = {origin: "http://x", href: "http://x/", search: "", hash: ""};
const bodies = [];
globalThis.fetch = async (url, opt) => {
  if (opt && opt.body && String(url).includes("chat/completions")) bodies.push(JSON.parse(opt.body));
  return {ok: false, status: 503, headers: {get: () => null}, json: async () => ({error: {message: "stub"}}),
          text: async () => "stub", body: null};
};
const [web, arg] = process.argv.slice(2);
const {health, pick} = JSON.parse(arg);
const st = await import(pathToFileURL(web + "/state.js").href);
const ch = await import(pathToFileURL(web + "/chats.js").href);
const S = st.S;
S.health = health; S.attachments = []; S.toasts = []; S.draft = ""; S.editIndex = null;
st.setFamily(health.model); st.loadPrefs();
S.current = ch.newChatObj();
if (pick) ch.setThinking(pick);
S.draft = "hello";
await ch.send();
console.log(JSON.stringify(bodies.map((b) => Object.fromEntries(Object.entries(b).filter(([k]) => k !== "messages")))));
process.exit(0);
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebChat(unittest.TestCase):
    def run_page(self, health, script=CHATS_LEVEL_JS, **extra):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "t.mjs").write_text(script, encoding="utf-8")
            r = subprocess.run(["node", str(Path(t) / "t.mjs"), str(ROOT / "serve/web"),
                                json.dumps({"health": health, **extra})],
                               capture_output=True, encoding="utf-8", timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_request_says_so(self):
        g53 = {"model": "glm-5.3", "model_label": "GLM-5.3 (744B)", "thinking_budget_max": AT_MAX}
        flash = {"model": "glm-5.3-flash", "model_label": "GLM-5.3 · Maya-S24"}
        left = self.run_page(g53, CHATS_SEND_JS)
        self.assertEqual(len(left), 1)
        self.assertEqual((left[0]["reasoning_effort"], left[0]["strata_level_default"]), ("max", True))
        picked = self.run_page(g53, CHATS_SEND_JS, pick="max")
        self.assertEqual({k: v for k, v in left[0].items() if k != "strata_level_default"}, picked[0])
        self.assertNotIn("strata_level_default", picked[0])
        for pick in (None, "max", "high"):                 # every other model: the request as before
            sent = self.run_page(flash, CHATS_SEND_JS, pick=pick)[0]
            self.assertNotIn("strata_level_default", sent)
            self.assertEqual(sent["reasoning_effort"], pick or "max")

    def test_glm53_marks_a_level_its_user_did_not_pick(self):
        out = self.run_page({"model": "glm-5.3", "model_label": "GLM-5.3 (744B)", "thinking_budget_max": AT_MAX})
        self.assertEqual(out["new"], {"level": "max", "picked": False, "left": True})        # Max, the model's default
        self.assertEqual(out["tapped_max"], {"level": "max", "picked": True, "left": False})  # the speedometer's tap
        self.assertEqual(out["high_then_max"], {"level": "max", "picked": True, "left": False})
        self.assertEqual(out["tapped_high"], {"level": "high", "picked": True, "left": False})
        self.assertEqual(out["other_model"], {"level": "max", "picked": False, "left": True})  # (set under another one)
        self.assertEqual(out["settings_max"], {"level": "max", "picked": True, "left": False})  # (Settings' own Max)

    def test_every_other_model_and_a_users_own_budget_never(self):
        for health in ({"model": "glm-5.3-flash", "model_label": "GLM-5.3 · Maya-S24"},
                       {"model": "glm-5.3", "model_label": "GLM-5.3 (744B)"}):    # (no long budget: the user's own)
            out = self.run_page(health)
            self.assertEqual({k: v["left"] for k, v in out.items()}, {k: False for k in out}, health)
            self.assertEqual(out["new"]["picked"], False)


# ------------------------------------------------------------------------------------------------ main()
class MainStart(unittest.TestCase):
    """serve/server.py started with a run config (the fake engine): what it says, sends and shows at each budget."""

    def tmpdir(self) -> Path:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=WIN)
        self.addCleanup(tmp.cleanup)                     # (after the server's stop: cleanups run last in, first out)
        return Path(tmp.name)

    def start(self, d: Path, cfg_edit, glm_dsa=True):
        pack = glm53_pack(d / "pack") if glm_dsa else d / "pack"
        tok_dir = glm_tokenizer_dir(pack / "tokenizer", OFFICIAL if glm_dsa else FLASH_TEMPLATE)
        tok = glm_tokenizer(d / "tok2")
        cfg = glm53_config(pack)
        cfg.pop("thinking_budget")
        cfg["args"] = [str(CONTEXT) if a == "65536" else a for a in cfg["args"]]
        cfg.update(exe=fake_exe(d), tokenizer=str(tok_dir), cwd=str(d), log=str(d / "engine.log"), reasoning_effort="max",
                   model_name="glm-5.3" if glm_dsa else "glm-5.3-flash")
        cfg["env"] = dict(cfg["env"], FAKE_ENGINE_DIR=str(d), STRATA_GPU_RELEASE_S="0",
                          FAKE_ENGINE_GEN_LOG=str(d / "gen.txt"),
                          FAKE_ENGINE_IDS=",".join(map(str, tok.encode("Some thought.</think>Hello."))))
        cfg_edit(cfg)
        p = d / "maya-test.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        port = free_port()
        env = dict(os.environ)
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                                 str(p), "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(d), env=env)
        self.addCleanup(self.stop, proc)
        out, end = [], time.time() + 60
        while time.time() < end:
            line = proc.stdout.readline()
            if not line:
                break
            out.append(line)
            if "ready: http://" in line:
                return f"http://127.0.0.1:{port}", "".join(out), p
        self.fail("".join(out))

    @staticmethod
    def stop(proc):
        proc.terminate()
        try:
            proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:      # (a stop that hangs: main's signal handler, every model - not this test's)
            proc.kill()
            proc.communicate()

    @staticmethod
    def call(base, path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def ask(self, base, d: Path, **body) -> int | None:
        out = self.call(base, "/v1/chat/completions", {"model": "glm-5.3", "max_tokens": BIG,
                                                       "messages": [{"role": "user", "content": "hi"}], **body})
        self.assertEqual(out["choices"][0]["message"]["content"], "Hello.")
        return budget_of((d / "gen.txt").read_text(encoding="utf-8").splitlines()[-1])

    def test_glm53_with_the_default_budget(self):
        d = self.tmpdir()
        base, text, cfg_path = self.start(d, lambda cfg: None)
        self.assertIn("thinking budget: 32768 tokens, up to 65536 when a request asks for Max (then the answer)", text)
        self.assertEqual(self.ask(base, d), HIGH)                                 # (the config's level: Max, no ask)
        self.assertEqual(self.ask(base, d, reasoning_effort="max"), MAX)
        self.assertEqual(self.ask(base, d, reasoning_effort="high"), HIGH)
        self.assertEqual(self.call(base, "/api/thinking-budget"),
                         {"budget": DEFAULT, "supported": True, "saved": True, "max_level": AT_MAX})
        self.assertEqual(self.call(base, "/health")["thinking_budget_max"], AT_MAX)
        # the dashboard's field: saved as the user's own, and from then on the same at every level
        self.assertEqual(self.call(base, "/api/thinking-budget", {"budget": 40000}),
                         {"budget": 40000, "supported": True, "saved": True})
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual((saved["thinking_budget"], saved["installer"]["thinking_budget_set"]), (40000, True))
        self.assertEqual(self.call(base, "/api/thinking-budget"), {"budget": 40000, "supported": True, "saved": True})
        self.assertNotIn("thinking_budget_max", self.call(base, "/health"))
        self.assertEqual(self.ask(base, d, reasoning_effort="max"), 40000 + GRACE)
        self.assertEqual(self.ask(base, d, reasoning_effort="high"), 40000 + GRACE)

    def test_glm53_with_the_users_own_budget(self):
        def own(cfg):
            cfg.update(thinking_budget=DEFAULT, installer={"thinking_budget_set": True})
        d = self.tmpdir()
        base, text, _ = self.start(d, own)
        self.assertIn("thinking budget: 32768 tokens (then the answer)", text)
        self.assertEqual(self.ask(base, d), HIGH)
        self.assertEqual(self.ask(base, d, reasoning_effort="max"), HIGH)
        self.assertEqual(self.call(base, "/api/thinking-budget"), {"budget": DEFAULT, "supported": True, "saved": True})
        self.assertNotIn("thinking_budget_max", self.call(base, "/health"))

    def test_flash_is_as_before(self):
        d = self.tmpdir()
        base, text, _ = self.start(d, lambda cfg: None, glm_dsa=False)
        self.assertNotIn("GLM-5.3 (glm-dsa)", text)
        self.assertIn("thinking budget: 32768 tokens (then the answer)", text)
        self.assertEqual(self.ask(base, d), HIGH)
        self.assertEqual(self.ask(base, d, reasoning_effort="max"), HIGH)
        self.assertEqual(self.call(base, "/api/thinking-budget"), {"budget": DEFAULT, "supported": True, "saved": True})
        self.assertNotIn("thinking_budget_max", self.call(base, "/health"))


if __name__ == "__main__":
    unittest.main()
