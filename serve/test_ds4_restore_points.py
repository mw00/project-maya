"""MAYA_DS4_RESTORE_POINTS: where the server asks the DeepSeek engine to keep restore points (its GEN line's marks= and
shared=), computed from the chat as the template renders it - no GPU, no engine (the mock engine).

    python -m unittest serve.test_ds4_restore_points -v
"""
from __future__ import annotations

import os
import sys
import threading
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import ds41_encoding as reference  # noqa: E402
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, deepseek_restore_points,  # noqa: E402
                          openai_chunks, openai_collect, restore_point_settings)
import maya_names  # noqa: E402  (tools/, on the path server.py set)

ROOT = Path(__file__).resolve().parents[1]
ON = maya_names.read_as("MAYA_DS4_RESTORE_POINTS")
TAIL = maya_names.read_as("MAYA_DS4_RESTORE_TAIL")
# one id a byte (the stand-in tokenizer): a system prompt and its tools of ~2,900 tokens
SYSTEM = "You are a careful assistant; answer from the documents you are given. " * 40
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Looks a thing up. " * 20, "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}}}}
DOC = "".join(f"Line {i}: the quarterly figures for region {i % 7} rose by {i % 13} percent.\n" for i in range(400))


class Tok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [
        reference.bos_token, reference.eos_token, "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
        "<｜latest_reminder｜>", "<think>", "</think>", "｜DSML｜", reference.IMAGE_PLACEHOLDER]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}


def lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Recording(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class RestorePoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tok()
        cls.engine = Recording(cls.tok, "Fine.", max_context=1 << 20)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "tools/ds4/chat_template.jinja"))

    def ids(self, messages, tools=None, **options):
        request = {"messages": messages, **options}
        if tools:
            request["tools"] = tools
        msgs, schemas, kw = openai_to_messages(request, deepseek=True)
        return self.svc.prepare(msgs, schemas, kw, 100)[0]

    def points(self, ids, tail=512):
        return deepseek_restore_points(ids, self.svc.ds_user_id, self.svc.ds_system_id, tail)

    def chat(self, last="And how did region 3 do in the last quarter?", system=SYSTEM):
        return [{"role": "system", "content": system},
                {"role": "user", "content": "Here are the figures.\n" + DOC},
                {"role": "assistant", "content": "I have read them."},
                {"role": "user", "content": last}]

    def test_last_user_turn_and_the_shared_start(self):
        ids = self.ids(self.chat())
        pts = self.points(ids)
        user = self.svc.ds_user_id
        shared, (boundary,) = pts["shared"], pts["marks"]
        self.assertEqual(ids[shared], user)
        self.assertEqual(ids.index(user), shared)                     # (the first user turn: the system part's end)
        self.assertEqual(boundary, len(ids) - 1 - ids[::-1].index(user))   # (the last one)
        self.assertGreater(shared, 2000)
        # the last message edited: the prompt agrees with this one up to the boundary, and leaves it before the end -
        # past the anchor and both checkpoints of the prompt's end, so without the point it was read from token 0
        edited = self.ids(self.chat("And region 5, over the whole year?"))
        self.assertGreaterEqual(lcp(ids, edited), boundary)
        self.assertLess(lcp(ids, edited), len(ids) - 1)
        # another conversation behind the same system prompt: it agrees up to the shared start
        other = self.ids([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Something else. " * 80}])
        self.assertGreaterEqual(lcp(ids, other), shared)
        self.assertEqual(self.points(other).get("shared"), shared)

    def test_tools_render_in_the_shared_start(self):
        with_tools = self.ids(self.chat(), tools=[TOOL])
        pts = self.points(with_tools)
        self.assertEqual(with_tools[pts["shared"]], self.svc.ds_user_id)
        self.assertGreater(pts["shared"], self.points(self.ids(self.chat()))["shared"])   # (the tools are in it)

    def test_tail_point_for_a_long_last_message(self):
        q1 = self.ids([{"role": "user", "content": DOC + "\nWhich region rose most?"}])
        q2 = self.ids([{"role": "user", "content": DOC + "\nSum the figures of region 2."}])
        pts = self.points(q1)
        self.assertNotIn("shared", pts)                             # (no system prompt: the first turn is at the start)
        # (its own key, tail=: the engine leaves it out where its read stop costs more than reading those tokens)
        self.assertEqual(pts, {"tail": len(q1) - 512})
        self.assertGreaterEqual(lcp(q1, q2), len(q1) - 512)          # (a new question goes back to it)
        self.assertEqual(self.points(q1, tail=0), {})
        self.assertEqual(self.points(q1, tail=300)["tail"], len(q1) - 300)
        # a short last message: the boundary alone (a tail point before it would be a read cut for nothing)
        pts = self.points(self.ids(self.chat()))
        self.assertEqual(len(pts["marks"]), 1)
        self.assertNotIn("tail", pts)

    def test_short_prompts_and_literal_markers(self):
        self.assertEqual(self.points(self.ids([{"role": "user", "content": "Hi"}])), {})
        self.assertEqual(self.points(self.ids(self.chat(system="Be brief."))).get("shared"), None)
        self.assertEqual(deepseek_restore_points(list(range(5000)), None), {})   # (no user token: no DeepSeek prompt)
        # a role marker typed inside a message is text, not a turn: the points sit on the template's own tokens
        literal = self.ids(self.chat("Quote this: <｜User｜> and <｜System｜> are DeepSeek's markers. " * 20))
        pts = self.points(literal, tail=0)
        for p in pts["marks"] + [pts["shared"]]:
            self.assertEqual(literal[p], self.svc.ds_user_id)
        self.assertEqual(sum(1 for t in literal if t == self.svc.ds_user_id), 2)
        self.assertEqual(literal.count(self.svc.ds_system_id), 1)   # (the system prompt's own, at the start)
        self.assertEqual(self.points(literal)["tail"], len(literal) - 512)   # (the tail: inside the long message)

    def test_tool_results_and_a_later_system_message(self):
        call = {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}
        msgs = self.chat()[:2] + [{"role": "assistant", "content": "", "tool_calls": [call]},
                                  {"role": "tool", "tool_call_id": "a", "content": "The result."}]
        ids = self.ids(msgs, tools=[TOOL])
        (boundary,) = self.points(ids)["marks"]
        self.assertEqual(ids[boundary], self.svc.ds_user_id)          # (a tool result's turn opens with <｜User｜>)
        self.assertEqual(boundary, len(ids) - 1 - ids[::-1].index(self.svc.ds_user_id))
        late = self.chat() + [{"role": "assistant", "content": "Region 3 rose."},
                              {"role": "system", "content": "From now on answer in French."}]
        ids = self.ids(late)
        (boundary,) = self.points(ids)["marks"]
        self.assertEqual(ids[boundary], self.svc.ds_system_id)
        self.assertGreater(boundary, len(ids) - 60)

    def test_settings(self):
        self.assertEqual(restore_point_settings({}), (True, 512))   # (on by default: 4)
        self.assertEqual(restore_point_settings({ON: "0"}), (False, 512))
        self.assertEqual(restore_point_settings({ON: "4"}), (True, 512))
        self.assertEqual(restore_point_settings({ON: "yes", TAIL: "x"}), (True, 512))   # (not a number: the default)
        self.assertEqual(restore_point_settings({ON: "8", TAIL: "0"}), (True, 0))
        self.assertEqual(restore_point_settings({ON: "8", TAIL: "2000"}), (True, 2000))

    def run_request(self, messages, env, svc=None, **extra):
        svc = svc or self.svc
        request = {"messages": messages, **extra}
        msgs, schemas, kw = openai_to_messages(request, deepseek=True)
        ids, thinking, max_new = svc.prepare(msgs, schemas, kw, 50)
        with unittest.mock.patch.dict(os.environ, env):
            for k in (ON, TAIL):
                if k not in env:
                    os.environ.pop(k, None)
            openai_collect(openai_chunks(svc, request, ids, thinking, schemas, max_new, threading.Event()))
        return ids, svc.engine.last_sampling

    def test_the_engine_line(self):
        ids, sampling = self.run_request(self.chat(), {ON: "4"})
        pts = self.points(ids)
        self.assertEqual(sampling["_restore"], pts)
        keys = StrataEngine.sampling_keys(sampling).split()
        self.assertIn("marks=" + ",".join(str(p) for p in pts["marks"]), keys)
        self.assertIn(f"shared={pts['shared']}", keys)
        self.assertIn("temperature=0.7", StrataEngine.sampling_keys({**sampling, "temperature": 0.7}).split())
        self.assertIn(f"shared={pts['shared']}", StrataEngine.sampling_keys({**sampling, "temperature": 0.7}).split())
        # MAYA_DS4_RESTORE_TAIL moves the tail point
        _, sampling = self.run_request([{"role": "user", "content": DOC}], {ON: "4", TAIL: "100"})
        self.assertEqual(sampling["_restore"]["tail"], len(self.engine.last_prompt) - 100)
        self.assertNotIn("marks", sampling["_restore"])
        self.assertIn(f"tail={len(self.engine.last_prompt) - 100}", StrataEngine.sampling_keys(sampling).split())

    def test_off_with_0_and_never_the_clients_own(self):
        _, on = self.run_request(self.chat(), {})   # (on by default)
        self.assertIn("_restore", on)
        _, off = self.run_request(self.chat(), {ON: "0"})
        self.assertNotIn("_restore", off)
        self.assertNotIn("marks=", StrataEngine.sampling_keys(off))
        self.assertNotIn("shared=", StrataEngine.sampling_keys(off))
        self.assertNotIn("tail=", StrataEngine.sampling_keys(off))
        # the line with the setting off is the line without this feature (the keys of the request's own sampling)
        self.assertEqual(StrataEngine.sampling_keys(off), StrataEngine.sampling_keys({}))
        # a client's own `_restore` never reaches the engine
        _, forged = self.run_request(self.chat(), {ON: "0"}, _restore={"marks": [5], "shared": 7})
        self.assertNotIn("_restore", forged)
        _, forged = self.run_request(self.chat(), {ON: "4"}, _restore={"marks": [5], "shared": 7})
        self.assertNotEqual(forged["_restore"].get("marks"), [5])
        # another model's template: no points
        glm = Service(Recording(self.tok, "Fine.", max_context=1 << 20), self.tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        _, other = self.run_request(self.chat(), {ON: "4"}, svc=glm)
        self.assertNotIn("_restore", other)


if __name__ == "__main__":
    unittest.main()
