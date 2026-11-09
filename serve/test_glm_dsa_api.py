"""serve/test_glm_dsa_api.py - GLM-5.3 (glm-dsa) speaks every API the way GLM-5.3-Flash does, over the real HTTP server
and serve/glm_dsa_fake_engine.py (the engine's wire protocol: GEN lines, T ids, the thinking budget, DONE), with
GLM-5.3's chat templates (the official one, and the one the UD-IQ2_M GGUF carries: its pack is served with it) and a
tokenizer that has GLM's special tokens - no GPU, no model:

  Responses:  reasoning items out and back in; function_call / function_call_output by call_id, one call and two
              (streamed, their outputs sent out of order); the stream's events; a picture in a tool's output
  Anthropic:  thinking blocks out and back in, budget_tokens and "disabled"; tool_use / tool_result by id (results sent
              out of order); the stream's events
  OpenAI:     parallel calls with ids (results sent out of order reach the template in the calls' order), streamed
              call deltas; tool_choice required / a named function / none; response_format json_object / json_schema;
              the prompt's count (/api/tokens and /tokenize: Maya serves no /v1/messages/count_tokens, for any model)
  Thinking budget: the GEN line carries think_budget and GLM-5.3's </think> id, the engine closes the thinking there
              and the answer follows; the dashboard's field says it is supported
  The Monitor: every key /metrics and the Monitor read is one the engine's source prints on the GLM path and a real
              GLM-5.3 run printed, and /metrics carries each figure after a request

The answers are scripted per request (FAKE_ENGINE_SCRIPT), each ending the way GLM-5.3 ends a turn: <|user|>, or
<|observation|> after a tool call.

    python -m unittest serve.test_glm_dsa_api -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import Service, StrataEngine, serve  # noqa: E402
from serve.test_glm_dsa_serve import OFFICIAL, WIN, fake_exe, glm_tokenizer  # noqa: E402

CALL = "<tool_call>lookup<arg_key>key</arg_key><arg_value>alpha</arg_value></tool_call>"
CALL_B = CALL.replace("alpha", "beta")
SCHEMA = {"type": "object", "properties": {"key": {"type": "string"}}}
OAI_TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a value", "parameters": SCHEMA}}]
RESP_TOOLS = [{"type": "function", "name": "lookup", "description": "Look up a value", "parameters": SCHEMA}]
ANTH_TOOLS = [{"name": "lookup", "description": "Look up a value", "input_schema": SCHEMA}]
# GLM-5.3's template in a picture's place (data/glm-dsa/templates/glm-5.3.official.jinja, visible_text)
NOTE = ("<reminder>You are unable to process this image because you don't have multi-modal input ability. "
        "Try different methods.</reminder>")
IMG = "data:image/png;base64,iVBORw0KGgo="


class _Glm53Api(unittest.TestCase):
    """One fake GLM-5.3 engine and one server for the class; each test scripts the answers it needs."""
    TEMPLATE = OFFICIAL

    @classmethod
    def setUpClass(cls):
        # (each part undone by a class cleanup as soon as it exists: a start that fails half-way leaves nothing behind)
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
        cls.engine = StrataEngine(fake_exe(d), ["--glm-pack", str(d), "--max-context", "65536"], env=env,
                                  log=str(d / "engine.log"))     # (its start lines: the context's cost, as a server's)
        cls.addClassCleanup(cls.engine.log.close)                # (after the engine has ended: the cleanups run LIFO)
        cls.addClassCleanup(cls.engine.close)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(cls.TEMPLATE, glm_dsa=True), model_name="glm-5.3")
        cls.httpd = serve(cls.svc, port=0)
        cls.addClassCleanup(cls.httpd.server_close)
        cls.addClassCleanup(cls.httpd.shutdown)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    # ------------------------------------------------------------------ helpers
    def script(self, *answers: str):
        """The next requests' answers, as GLM-5.3 writes them: each ends its turn (<|observation|> after a call)."""
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

    def stream(self, path, body) -> list[tuple[str, object]]:
        """A streamed answer as (event name, data) pairs: Responses' and Anthropic's named events, OpenAI's unnamed
        data lines ("" as the name; its final [DONE] as the string)."""
        with urllib.request.urlopen(self.request(path, {**body, "stream": True}), timeout=60) as r:
            self.assertEqual(r.headers.get_content_type(), "text/event-stream")
            raw = r.read().decode("utf-8")
        out = []
        for block in raw.replace("\r\n", "\n").split("\n\n"):
            name, data = "", None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    data = line[6:] if line[6:] == "[DONE]" else json.loads(line[6:])
            if data is not None:
                out.append((name, data))
        return out

    def gen_line(self) -> list[str]:
        return self.gen_log.read_text(encoding="utf-8").splitlines()[-1].split()

    def prompt(self) -> str:
        """The last prompt the engine was given, as text (its GEN line's ids)."""
        return self.tok.decode([int(x) for x in self.gen_line()[-1].split(",") if x])

    def prompt_ids(self) -> list[int]:
        return [int(x) for x in self.gen_line()[-1].split(",") if x]


# ---------------------------------------------------------------------------------------------- Responses
class Responses(_Glm53Api):
    def test_reasoning_out_streamed_and_back_in(self):
        self.script("check the sum</think>42", "checked again</think>Correct.")
        events = self.stream("/v1/responses", {"input": "What is the sum?", "reasoning": {"effort": "high"}})
        names, data = [e for e, _ in events], [d for _, d in events]
        self.assertEqual(names[:2], ["response.created", "response.in_progress"])
        self.assertEqual(names[-1], "response.completed")
        self.assertEqual([d["type"] for d in data], names)
        self.assertEqual([d["sequence_number"] for d in data], list(range(len(data))))
        self.assertEqual("".join(d["delta"] for e, d in events if e == "response.reasoning_text.delta"),
                         "check the sum")
        self.assertEqual("".join(d["delta"] for e, d in events if e == "response.output_text.delta"), "42")
        out = data[-1]["response"]
        self.assertEqual((out["status"], out["model"]), ("completed", "glm-5.3"))
        self.assertEqual([i["type"] for i in out["output"]], ["reasoning", "message"])
        self.assertEqual(out["output"][0]["content"], [{"type": "reasoning_text", "text": "check the sum"}])
        self.assertGreater(out["usage"]["output_tokens_details"]["reasoning_tokens"], 0)
        self.assertIn("<|system|>Reasoning Effort: High", self.prompt())
        # the reasoning item sent back: GLM-5.3's template keeps it (clear_thinking is false unless a client asks)
        history = [{"role": "user", "content": "What is the sum?"}, *out["output"],
                   {"role": "user", "content": [{"type": "input_text", "text": "Check it"}]}]
        code, second = self.post("/v1/responses", {"input": history, "reasoning": {"effort": "high"}})
        self.assertEqual(code, 200, second)
        self.assertEqual(second["output"][-1]["content"][0]["text"], "Correct.")
        self.assertIn("<|user|>What is the sum?<|assistant|><think>check the sum</think>42<|user|>Check it"
                      "<|assistant|><think>", self.prompt())

    def test_reasoning_off(self):
        self.script("Just the answer.")
        code, out = self.post("/v1/responses", {"input": "hi", "reasoning": {"effort": "none"}})
        self.assertEqual(code, 200, out)
        self.assertEqual([i["type"] for i in out["output"]], ["message"])
        self.assertEqual(out["output"][0]["content"][0]["text"], "Just the answer.")
        self.assertTrue(self.prompt().endswith("<|user|>hi<|assistant|><think></think>"), self.prompt()[-60:])

    def test_function_call_round_trip_by_call_id(self):
        self.script("I will look it up.</think>" + CALL, "it is 73</think>The value is 73.")
        code, out = self.post("/v1/responses", {"input": "look up alpha", "tools": RESP_TOOLS})
        self.assertEqual(code, 200, out)
        self.assertEqual(out["status"], "completed")
        self.assertEqual([i["type"] for i in out["output"]], ["reasoning", "function_call"])
        call = out["output"][1]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("lookup", {"key": "alpha"}))
        self.assertTrue(call["call_id"])
        self.assertIn('"name": "lookup", "description": "Look up a value"', self.prompt())   # the tools, as JSON
        history = [{"role": "user", "content": "look up alpha"}, *out["output"],
                   {"type": "function_call_output", "call_id": call["call_id"], "output": "73"}]
        code, second = self.post("/v1/responses", {"input": history, "tools": RESP_TOOLS})
        self.assertEqual(code, 200, second)
        self.assertEqual(second["output"][-1]["content"][0]["text"], "The value is 73.")
        self.assertIn("<|assistant|><think>I will look it up.</think>" + CALL +
                      "<|observation|><tool_response>73</tool_response><|assistant|><think>", self.prompt())

    def test_two_calls_streamed_and_their_outputs_in_call_order(self):
        self.script("Both.</think>" + CALL + CALL_B, "done</think>alpha is 1, beta is 2.")
        events = self.stream("/v1/responses", {"input": "look up alpha and beta", "tools": RESP_TOOLS})
        out = events[-1][1]["response"]
        calls = [i for i in out["output"] if i["type"] == "function_call"]
        self.assertEqual([json.loads(c["arguments"]) for c in calls], [{"key": "alpha"}, {"key": "beta"}])
        self.assertEqual(len({c["call_id"] for c in calls}), 2)
        added = [d["item"]["call_id"] for e, d in events
                 if e == "response.output_item.added" and d["item"]["type"] == "function_call"]
        self.assertEqual(added, [c["call_id"] for c in calls])
        done = [json.loads(d["arguments"]) for e, d in events if e == "response.function_call_arguments.done"]
        self.assertEqual(done, [{"key": "alpha"}, {"key": "beta"}])
        history = [{"role": "user", "content": "look up alpha and beta"}, *out["output"],
                   {"type": "function_call_output", "call_id": calls[1]["call_id"], "output": "2"},
                   {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "1"}]
        code, second = self.post("/v1/responses", {"input": history, "tools": RESP_TOOLS})
        self.assertEqual(code, 200, second)
        self.assertEqual(second["output"][-1]["content"][0]["text"], "alpha is 1, beta is 2.")
        self.assertIn(CALL + CALL_B + "<|observation|><tool_response>1</tool_response><tool_response>2</tool_response>",
                      self.prompt())

    def test_a_cut_call_is_never_published(self):
        # max_output_tokens ends the answer inside the call's argument: incomplete, and no completed arguments go out
        self.script(CALL)
        for cut in (14, 18):                             # (in "alpha"; right after it, before </arg_value>)
            with self.subTest(cut=cut):
                events = self.stream("/v1/responses", {"input": "look up alpha", "tools": RESP_TOOLS,
                                                       "reasoning": {"effort": "none"}, "max_output_tokens": cut})
                self.assertEqual(events[-1][0], "response.incomplete")
                self.assertNotIn("response.function_call_arguments.done", [e for e, _ in events])

    def test_a_picture_in_a_tools_output_is_the_templates_note(self):
        self.script("I will look.</think>" + CALL, "no picture</think>I cannot see pictures.")
        code, out = self.post("/v1/responses", {"input": "show it", "tools": RESP_TOOLS})
        self.assertEqual(code, 200, out)
        call = out["output"][-1]
        history = [{"role": "user", "content": "show it"}, *out["output"],
                   {"type": "function_call_output", "call_id": call["call_id"],
                    "output": [{"type": "input_text", "text": "shot:"}, {"type": "input_image", "image_url": IMG}]}]
        code, second = self.post("/v1/responses", {"input": history, "tools": RESP_TOOLS})
        self.assertEqual(code, 200, second)
        self.assertIn("<tool_response>shot:" + NOTE + "</tool_response>", self.prompt())


# ---------------------------------------------------------------------------------------------- Anthropic
class Anthropic(_Glm53Api):
    def test_thinking_blocks_out_and_back_in(self):
        self.script("Some thought.</think>Hello.", "more thought</think>Hello again.")
        code, b = self.post("/v1/messages", {"model": "glm-5.3", "max_tokens": 2000, "messages": [
            {"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": 1024}})
        self.assertEqual(code, 200, b)
        self.assertEqual((b["model"], b["stop_reason"]), ("glm-5.3", "end_turn"))
        self.assertEqual(b["content"], [{"type": "thinking", "thinking": "Some thought.", "signature": ""},
                                        {"type": "text", "text": "Hello."}])
        self.assertIn("<|system|>Reasoning Effort: Low", self.prompt())          # (budget_tokens under 2K: Low)
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": b["content"]},
                   {"role": "user", "content": "again"}]
        code, b = self.post("/v1/messages", {"model": "glm-5.3", "max_tokens": 2000, "messages": history})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["content"][-1]["text"], "Hello again.")
        self.assertIn("<|user|>hi<|assistant|><think>Some thought.</think>Hello.<|user|>again<|assistant|><think>",
                      self.prompt())
        self.assertIn("<|system|>Reasoning Effort: Max", self.prompt())          # (no thinking field: the default)

    def test_thinking_disabled(self):
        self.script("A plain answer.")
        code, b = self.post("/v1/messages", {"model": "glm-5.3", "max_tokens": 200, "thinking": {"type": "disabled"},
                                             "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["content"], [{"type": "text", "text": "A plain answer."}])
        self.assertTrue(self.prompt().endswith("<|assistant|><think></think>"), self.prompt()[-60:])

    def test_streamed_events(self):
        self.script("Some thought.</think>Hello.")
        events = self.stream("/v1/messages", {"model": "glm-5.3", "max_tokens": 2000,
                                              "messages": [{"role": "user", "content": "hi"}]})
        names = [e for e, _ in events]
        self.assertEqual((names[0], names[-1]), ("message_start", "message_stop"))
        self.assertEqual([d["type"] for _, d in events], names)
        starts = [d["content_block"]["type"] for e, d in events if e == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text"])
        self.assertEqual("".join(d["delta"]["thinking"] for e, d in events
                                 if e == "content_block_delta" and d["delta"]["type"] == "thinking_delta"),
                         "Some thought.")
        self.assertEqual("".join(d["delta"]["text"] for e, d in events
                                 if e == "content_block_delta" and d["delta"]["type"] == "text_delta"), "Hello.")
        delta = [d for e, d in events if e == "message_delta"]
        self.assertEqual(delta[-1]["delta"]["stop_reason"], "end_turn")

    def test_tool_use_round_trip_results_by_id(self):
        self.script("Both.</think>" + CALL + CALL_B, "done</think>alpha is 1, beta is 2.")
        code, b = self.post("/v1/messages", {"model": "glm-5.3", "max_tokens": 2000, "tools": ANTH_TOOLS,
                                             "messages": [{"role": "user", "content": "alpha and beta"}]})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["stop_reason"], "tool_use")
        uses = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual([(u["name"], u["input"]) for u in uses], [("lookup", {"key": "alpha"}),
                                                                   ("lookup", {"key": "beta"})])
        self.assertEqual(len({u["id"] for u in uses}), 2)
        results = [{"type": "tool_result", "tool_use_id": uses[1]["id"], "content": "2"},     # out of order
                   {"type": "tool_result", "tool_use_id": uses[0]["id"], "content": [{"type": "text", "text": "1"}]}]
        history = [{"role": "user", "content": "alpha and beta"}, {"role": "assistant", "content": b["content"]},
                   {"role": "user", "content": results}]
        code, b = self.post("/v1/messages", {"model": "glm-5.3", "max_tokens": 2000, "tools": ANTH_TOOLS,
                                             "messages": history})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["content"][-1]["text"], "alpha is 1, beta is 2.")
        self.assertEqual(b["stop_reason"], "end_turn")
        # the ids reached the template: it wrote the results in the order of the calls
        self.assertIn(CALL + CALL_B + "<|observation|><tool_response>1</tool_response><tool_response>2</tool_response>",
                      self.prompt())


# ---------------------------------------------------------------------------------------------- OpenAI
class OpenAI(_Glm53Api):
    def chat(self, messages, **body):
        return self.post("/v1/chat/completions", {"model": "glm-5.3", "messages": messages, **body})

    def test_parallel_calls_with_ids_and_results_in_call_order(self):
        self.script("Both.</think>" + CALL + CALL_B, "done</think>alpha is 1, beta is 2.")
        user = {"role": "user", "content": "alpha and beta"}
        code, b = self.chat([user], tools=OAI_TOOLS)
        self.assertEqual(code, 200, b)
        choice = b["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        msg = choice["message"]
        self.assertEqual(msg["reasoning_content"], "Both.")
        calls = msg["tool_calls"]
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in calls], [{"key": "alpha"}, {"key": "beta"}])
        self.assertEqual(len({c["id"] for c in calls}), 2)
        history = [user, {"role": "assistant", "content": None, "reasoning_content": "Both.", "tool_calls": calls},
                   {"role": "tool", "tool_call_id": calls[1]["id"], "content": "2"},           # out of order
                   {"role": "tool", "tool_call_id": calls[0]["id"], "content": "1"}]
        code, b = self.chat(history, tools=OAI_TOOLS)
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["message"]["content"], "alpha is 1, beta is 2.")
        self.assertIn("<|assistant|><think>Both.</think>" + CALL + CALL_B +
                      "<|observation|><tool_response>1</tool_response><tool_response>2</tool_response>", self.prompt())

    def test_streamed_call_deltas(self):
        self.script("Both.</think>" + CALL + CALL_B)
        events = self.stream("/v1/chat/completions", {"model": "glm-5.3", "tools": OAI_TOOLS,
                                                      "messages": [{"role": "user", "content": "alpha and beta"}]})
        self.assertEqual(events[-1], ("", "[DONE]"))
        chunks = [d for _, d in events[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        self.assertEqual("".join(d.get("reasoning_content") or "" for d in deltas), "Both.")
        parts = [t for d in deltas for t in d.get("tool_calls") or []]
        by_index = {}
        for t in parts:
            e = by_index.setdefault(t["index"], {"id": None, "args": ""})
            e["id"] = t.get("id") or e["id"]
            e["args"] += (t.get("function") or {}).get("arguments") or ""
        self.assertEqual(sorted(by_index), [0, 1])
        self.assertEqual([json.loads(by_index[i]["args"]) for i in (0, 1)], [{"key": "alpha"}, {"key": "beta"}])
        self.assertTrue(by_index[0]["id"] and by_index[1]["id"] and by_index[0]["id"] != by_index[1]["id"])
        finish = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices") and
                  c["choices"][0].get("finish_reason")]
        self.assertEqual(finish, ["tool_calls"])

    def test_tool_choice_required_named_and_none(self):
        user = [{"role": "user", "content": "look up alpha"}]
        # required: the answer starts with the call (thinking off, the opening in the prompt), the model goes on from it
        self.script(CALL[len("<tool_call>"):])
        code, b = self.chat(user, tools=OAI_TOOLS, tool_choice="required")
        self.assertEqual(code, 200, b)
        self.assertTrue(self.prompt().endswith("<|assistant|><think></think><tool_call>"), self.prompt()[-60:])
        self.assertEqual(json.loads(b["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]),
                         {"key": "alpha"})
        # a named function: its name opens the call
        self.script(CALL[len("<tool_call>lookup"):])
        code, b = self.chat(user, tools=OAI_TOOLS, tool_choice={"type": "function", "function": {"name": "lookup"}})
        self.assertEqual(code, 200, b)
        self.assertTrue(self.prompt().endswith("<think></think><tool_call>lookup"), self.prompt()[-60:])
        self.assertEqual(b["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup")
        # none: no tool reaches the model
        self.script("No tool needed.</think>Fine.")
        code, b = self.chat(user, tools=OAI_TOOLS, tool_choice="none")
        self.assertEqual(code, 200, b)
        self.assertNotIn("lookup", self.prompt())
        self.assertNotIn("# Tools", self.prompt())
        self.assertEqual((b["choices"][0]["message"]["content"], b["choices"][0]["finish_reason"]), ("Fine.", "stop"))
        self.assertFalse(b["choices"][0]["message"].get("tool_calls"))

    def test_response_format_json(self):
        msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Where?"}]
        self.script('A city.</think>Sure! ```json\n{"city": "Paris"}\n```')
        code, b = self.chat(msgs, response_format={"type": "json_object"})
        self.assertEqual(code, 200, b)
        self.assertEqual(json.loads(b["choices"][0]["message"]["content"]), {"city": "Paris"})
        self.assertIn("Return only a valid JSON object", self.prompt())
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
                  "additionalProperties": False}
        body = {"messages": msgs, "response_format": {"type": "json_schema",
                                                      "json_schema": {"name": "city", "strict": True,
                                                                      "schema": schema}}}
        self.script('Paris.</think>{"city": "Paris"}')
        code, b = self.chat(**body)
        self.assertEqual(code, 200, b)
        self.assertEqual(json.loads(b["choices"][0]["message"]["content"]), {"city": "Paris"})
        p = self.prompt()                    # the schema in the system message, after the client's own words
        self.assertLess(p.index("You are terse."), p.index(json.dumps(schema)))
        self.assertLess(p.index(json.dumps(schema)), p.index("<|user|>Where?"))
        self.assertEqual(p.count("<|system|>"), 2)   # (the template's Reasoning Effort line, then the system message)

    def test_the_prompts_count(self):
        """/api/tokens (the page's context meter) and /tokenize count the prompt the engine is given."""
        self.script("Counted.</think>Yes.")
        history = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "alpha?"},
                   {"role": "assistant", "content": "", "tool_calls": [
                       {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{\"key\": \"alpha\"}"}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "1"}]
        body = {"messages": history, "tools": OAI_TOOLS, "reasoning_effort": "low"}
        code, counted = self.post("/api/tokens", body)
        self.assertEqual(code, 200, counted)
        code, tokenized = self.post("/tokenize", body)
        self.assertEqual(code, 200, tokenized)
        code, b = self.chat(**body)
        self.assertEqual(code, 200, b)
        self.assertEqual(counted["tokens"], len(self.prompt_ids()))
        self.assertEqual(tokenized["tokens"], self.prompt_ids())
        self.assertEqual(b["usage"]["prompt_tokens"], len(self.prompt_ids()))
        self.assertIn("<|system|>Reasoning Effort: Low", self.prompt())


# the same with the template the UD-IQ2_M GGUF carries - the one its pack is served with (serve/server.py
# chat_template_path: a GLM-5.3 pack's own); it differs from the official one in five places
GGUF_TEMPLATE = OFFICIAL.parent / "glm-5.3.gguf-ud-iq2_m.jinja"


class ResponsesGgufTemplate(Responses):
    TEMPLATE = GGUF_TEMPLATE


class AnthropicGgufTemplate(Anthropic):
    TEMPLATE = GGUF_TEMPLATE


class OpenAIGgufTemplate(OpenAI):
    TEMPLATE = GGUF_TEMPLATE


# ---------------------------------------------------------------------------------------------- thinking budget
class ThinkingBudget(_Glm53Api):
    def setUp(self):
        self.saved = self.svc.think_budget
        self.svc.think_budget = 300

    def tearDown(self):
        self.svc.think_budget = self.saved

    def keys(self) -> dict:
        return dict(x.split("=", 1) for x in self.gen_line()[2:-1] if "=" in x)

    @staticmethod
    def letters(n: int, seed: int) -> str:
        """n tokens of thinking (one a letter here) with no sentence end and nothing the loop guard would fold."""
        import random
        rng = random.Random(seed)
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))

    def test_the_engine_closes_the_thinking_and_the_answer_follows(self):
        end = self.tok.encode("</think>")
        self.assertEqual(len(end), 1)
        thought = self.letters(600, 1)
        self.script(thought + "</think>Done.")
        code, b = self.post("/v1/chat/completions", {"model": "glm-5.3", "max_tokens": 1000,
                                                     "messages": [{"role": "user", "content": "think long"}]})
        self.assertEqual(code, 200, b)
        # (a max_tokens of 1024 or fewer keeps an answer's reserve: the budget stays 300 here)
        self.assertEqual((self.keys()["think_budget"], self.keys()["think_end"]), ("300", str(end[0])))
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], thought[:300])
        self.assertEqual((msg["content"], b["choices"][0]["finish_reason"]), ("Done.", "stop"))

    def test_a_large_limit_gets_the_grace(self):
        self.script(self.letters(200, 2) + "</think>Short.")
        code, b = self.post("/v1/chat/completions", {"model": "glm-5.3", "max_tokens": 8000,
                                                     "messages": [{"role": "user", "content": "think"}]})
        self.assertEqual(code, 200, b)
        self.assertEqual(self.keys()["think_budget"], "812")       # the budget + its grace (the server's soft close)
        self.assertEqual(b["choices"][0]["message"]["content"], "Short.")

    def test_thinking_off_sends_no_budget(self):
        self.script("No thinking.")
        code, b = self.post("/v1/chat/completions", {"model": "glm-5.3", "reasoning_effort": "none",
                                                     "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, b)
        self.assertNotIn("think_budget", self.keys())
        self.assertEqual(b["choices"][0]["message"]["content"], "No thinking.")

    def test_the_dashboard_field(self):
        self.assertEqual(self.get("/api/thinking-budget")["supported"], True)


# ---------------------------------------------------------------------------------------------- the Monitor's figures
# The keys the engine printed on GLM-5.3 (glm-dsa) in a real run (2026-11-07; the values left out) - the very lines
# GLM-5.3-Flash's engine prints (src/program/generate.cpp: the glm-fast serve start, its decode loop's STAT every 8
# tokens, the 10-field DONE)
REAL_INFO = ("context kv kv_resident expert_slots expert_cache_mib engine_kind experts vram_slots vram_gb ram_slots "
             "ram_gb mtp dflash vision_lend cpu_threads cpu_threads_max pcie_share logit_bias").split()
REAL_STAT = ("tok_s ms_tok vram_hit ram_fetch disk promo vram_used vram_slots ram_used ram_slots ram_gb vram_gb disk_gb "
             "draft_acc draft_off").split()
REAL_VRAM = "dev pci total free weights kv kv_ram act drafter window pool tail borrow vision slots ctx layers".split()
# what monitor.js reads from INFO that GLM's engine does not print: DeepSeek's (its n-gram memory, its helper GPU) and
# the Windows-APU line (INFO gpu_commit_capped_gb / pagefile_hint_gb: printed beside the first only when the page file
# keeps experts off the GPU) - and the server's own max_context
NOT_GLM_INFO = {"engram_lookups", "helper_allocated_bytes", "helper_slots", "gpu_commit_capped_gb", "pagefile_hint_gb",
                "max_context"}


def printf_keys(src: str, start: str, must: str = "") -> list[list[str]]:
    """The key=... names of every printf in `src` whose format begins with `start` (its string literals joined, up to the
    one that ends the line), and that holds `must`."""
    import re
    out = []
    for m in re.finditer(re.escape('printf("' + start), src):
        fmt, pos = "", m.start() + len("printf(")
        while True:
            lit = re.compile(r'\s*"((?:[^"\\]|\\.)*)"').match(src, pos)
            if not lit:
                break
            fmt += lit.group(1)
            pos = lit.end()
            if fmt.endswith("\\n"):
                break
        if must in fmt:
            out.append(re.findall(r"(\w+)=", fmt))
    return out


class MonitorFigures(_Glm53Api):
    """Every figure the Monitor and /metrics show for the GLM engine is printed on GLM-5.3's path: the engine's source
    (the glm-fast serve lines, shared with GLM-5.3-Flash), the real run's keys, the fake's lines, and what the server
    makes of them (/metrics).  The GPU day compares /metrics of GLM-5.3-Flash and GLM-5.3 on the real engine."""

    def test_the_engine_source_prints_them(self):
        src = (Path(__file__).resolve().parents[1] / "src/program/generate.cpp").read_text(encoding="utf-8")
        self.assertEqual(printf_keys(src, "INFO context=", "engine_kind=glm-fast"), [REAL_INFO])
        stats = printf_keys(src, "STAT tok_s=", " ram_slots=")  # (the GLM engine's: the Qwen path's has no RAM tier)
        self.assertEqual(len(stats), 2)                      # one answer, and answers at once (--parallel)
        self.assertEqual(stats[0], REAL_STAT)
        self.assertEqual(stats[1], REAL_STAT + ["active"])
        self.assertEqual(printf_keys(src, "VRAM dev="), [REAL_VRAM])
        self.assertEqual(printf_keys(src, "INFO gpu_commit_capped_gb="), [["gpu_commit_capped_gb", "pagefile_hint_gb"]])

    def test_the_monitor_reads_only_what_the_engine_prints(self):
        import re
        js = (Path(__file__).resolve().parents[1] / "serve/web/monitor.js").read_text(encoding="utf-8")
        read = set(re.findall(r"\beng\.([a-z_0-9]+)", js))
        self.assertTrue(read)
        self.assertFalse(read - set(REAL_INFO) - NOT_GLM_INFO - {"dflash"}, read)
        glm_tiles = re.search(r"glm: \[(.*?)\n  \],", js, re.S).group(1)
        for key in re.findall(r'key: "(\w+)"', glm_tiles) + ["vram_hit", "ms_tok"]:
            self.assertIn(key, REAL_STAT)

    def test_the_fakes_lines_are_the_engines(self):
        self.assertFalse([k for k in REAL_INFO if k not in self.engine.info], self.engine.info)
        self.script("".join(chr(97 + i % 26) for i in range(40)))
        code, _ = self.post("/v1/chat/completions", {"model": "glm-5.3", "reasoning_effort": "none",
                                                     "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual(code, 200)
        self.assertEqual([k for k in self.engine.stat if k != "time"], REAL_STAT)

    def test_metrics_carry_every_figure(self):
        self.script("".join(chr(97 + (i * 7) % 26) for i in range(40)))
        code, b = self.post("/v1/chat/completions", {"model": "glm-5.3", "reasoning_effort": "none",
                                                     "messages": [{"role": "user", "content": "go on"}]})
        self.assertEqual(code, 200, b)
        m = self.get("/metrics")
        eng = m["engine"]
        self.assertEqual((eng["model"], eng["engine_kind"], eng["images"]), ("glm-5.3", "glm-fast", False))
        for k in ("experts", "expert_slots", "ram_gb", "vram_gb", "kv", "kv_resident", "cpu_threads", "pcie_share",
                  "kv_ctx", "kv_gb", "max_context"):
            self.assertIsNotNone(eng.get(k), k)
        self.assertEqual(eng["speculation"], ["MTP"])                   # (INFO mtp=1: the drafting it runs)
        now = m["tiers"]["now"]
        for k in REAL_STAT:
            self.assertIsInstance(now.get(k), float, k)
        last = m["requests"][0]
        for k in ("decode_tok_s", "prefill_tok_s", "hit_rate", "prompt_ms", "decode_ms"):
            self.assertIsNotNone(last.get(k), k)
        self.assertEqual((last["drafts_offered"], last["drafts_accepted"]), (0, 0))   # (the DONE line's: the fake drafts none)


if __name__ == "__main__":
    unittest.main()
