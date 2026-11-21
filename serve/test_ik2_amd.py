"""serve/test_ik2_amd.py - Inkling-2-Small's second engine, maya-inkling2 (AMD), under the one family "ik2", and what
the server does for the model beyond serve/test_inkling2_frontend.py: a config of either engine is the family "ik2"
("inkling2" its older name), the engine line each one gets, the Models page's settings, a tokenizer folder with the
GGUF's template swapped for thinkingmachines' own, a tool result headed by its tool's name, another role's message a
stop, the loop guard's view of a call and the prompts it continues with, and a tool result's picture as a note.

    python -m unittest serve.test_ik2_amd

Offline, no GPU: the template, tokenizer and scripted engine of serve/test_inkling2_frontend.py.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from serve import loop_guard, models, server
from serve.frontend import inkling_messages, openai_to_messages
from serve.test_inkling2_frontend import (ASK, CALL, END, EOS, MODEL, PUBLIC, TEXT, TOOL, TOOLS, USER, public,
                                          service)

AMD = {"exe": "/home/me/project-maya/build-hip/maya-inkling2",
       "args": ["--pack", "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2", "--max-context", "32768"],
       "tokenizer": "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2/tokenizer",
       "model_name": "inkling-2-small", "gpu": [0, 1, 2, 3], "backend": "hip"}
CUDA = {"exe": "/opt/maya/build/maya-ik2", "gpu": [0, 1], "model_name": "inkling-2-small",
        "args": ["--model", "/models/Inkling-2-Small-UD-Q3_K_XL-00001-of-00005.gguf", "--max-context", "32768",
                 "--gpus", "2"]}
ROUND = ASK + [
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": "a map:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}}]}]


class Family(unittest.TestCase):
    def test_one_family_two_engines(self):
        Q = server.maya_quants
        self.assertEqual(Q.family_of(AMD), "ik2")       # (by its engine's name: its --pack is not DeepSeek's)
        self.assertEqual(Q.family_of(CUDA), "ik2")
        old = dict(AMD, exe="/x/engine", installer={"family": "inkling2", "inkling2_quant": "ud-q2_k_xl"})
        self.assertEqual(Q.family_of(old), "ik2")       # (a config from before the two were one family)
        self.assertEqual(Q.family_of(dict(old, installer={"family": "ik2"})), "ik2")
        self.assertEqual(Q.model_defaults(AMD), Q.model_defaults(CUDA))
        self.assertEqual(Q.config_key(old), "ud-q2_k_xl")
        self.assertEqual(Q.config_key(dict(AMD, installer={})), "ud-q2_k_xl")    # (the quant its path names)
        self.assertEqual(Q.config_key(dict(CUDA, installer={})), "ud-q3_k_xl")
        self.assertEqual(Q.shown("ik2", "ud-q2_k_xl"), "Inkling 2 Small · UD-Q2_K_XL")

    def test_engine_line(self):
        # (several GPUs: maya-inkling2 takes the server's layer split; maya-ik2 stops at the flag - its own --gpus)
        self.assertEqual(server.engine_args(AMD), AMD["args"] + ["--layer-split", "auto"])
        self.assertEqual(server.engine_args(CUDA), CUDA["args"])
        self.assertEqual(server.child_env(AMD)["HIP_VISIBLE_DEVICES"], "0,1,2,3")

    def test_models_page(self):
        for cfg in (AMD, CUDA):
            self.assertEqual(models.family(cfg), "ik2")
            s = models.settings_of(cfg)
            self.assertEqual((s["context"], s["gpus"], s["images"]), (32768, cfg["gpu"], False))
            self.assertEqual(models.choices(cfg)["images"], False)
            out = models.apply(cfg, {"context": 65536, "gpus": [0, 1]})
            self.assertEqual(out["args"], [a if a != "32768" else "65536" for a in cfg["args"]])
            self.assertEqual(out["gpu"], [0, 1])
            with self.assertRaises(ValueError):
                models.apply(cfg, {"expert_cache": 100})                 # (GLM's: not these engines')
            with self.assertRaises(ValueError):
                models.apply(cfg, {"images": True})

    def test_setup_side(self):
        import maya
        self.assertEqual(maya.config_family(AMD), "ik2")
        self.assertEqual(maya.setup_title(AMD), "Inkling-2-Small")
        self.assertNotIn("ik2", maya.FAMILIES)                           # (its setup is AMD only: not a row everywhere)
        maya.select_build_backend("hip")
        self.assertEqual(maya.exe_for_target("maya-inkling2").name, "maya-inkling2")
        self.assertEqual(maya.bench_exe(AMD), Path(AMD["exe"]))


class Template(unittest.TestCase):
    def test_the_ggufs_template_is_swapped_for_the_official_one(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "chat_template.jinja").write_text(
                "{# llama.cpp's conversion #}<|message_model|>{{ x }}<|content_invoke_tool_json|>", encoding="utf-8")
            self.assertEqual(server.chat_template_path(d), PUBLIC)
            (Path(d) / "chat_template.jinja").write_text(PUBLIC.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(server.chat_template_path(d), Path(d) / "chat_template.jinja")

    def test_tool_results_named_by_their_call(self):
        calls = [{"type": "function", "function": {"name": n, "arguments": {}}} for n in ("get_weather", "get_time")]
        msgs = ASK + [{"role": "assistant", "content": "", "tool_calls": calls},
                      {"role": "tool", "content": "sunny"}, {"role": "tool", "content": "noon"},
                      {"role": "tool", "content": "one more than the calls"}]
        self.assertEqual([m.get("name") for m in inkling_messages(msgs)[2:]], ["get_weather", "get_time", None])
        text = public().render(msgs, tools=TOOLS)
        self.assertIn(TOOL + "get_weather" + TEXT + "sunny" + END + TOOL + "get_time" + TEXT + "noon" + END, text)
        # (as the server's own conversion leaves a client's messages: no ids, no names)
        got, tools, kw = openai_to_messages({"messages": ROUND[:2] + [dict(ROUND[2], content="18 C")], "tools": TOOLS})
        self.assertIn(TOOL + "get_weather" + TEXT + "18 C" + END, public().render(got, tools=tools))


class Serving(unittest.TestCase):
    def test_another_roles_message_is_a_stop(self):
        svc, _, tok = service("x")
        ids = tok.special_tokens
        self.assertLessEqual({ids[EOS], ids["<|endoftext|>"], ids[USER], ids["<|message_system|>"], ids[TOOL]},
                             svc.stop_ids)
        self.assertNotIn(ids[END], svc.stop_ids)                         # (it ends every block)
        self.assertNotIn(ids[MODEL], svc.stop_ids)

    def test_loop_guard_sees_the_calls(self):
        self.assertTrue(loop_guard._TOOL_OPEN.search(MODEL + "get_weather" + CALL))
        self.assertTrue(loop_guard._TOOL_CLOSE.search("}}" + END))
        self.assertIsNone(loop_guard._TOOL_OPEN.search(MODEL + TEXT + "no call here"))
        svc, _, tok = service("x")
        guard = loop_guard.ToolSafeGuard(piece=lambda t: tok.decode([t]))
        seen = [guard._follow(t) for t in tok.encode(MODEL + TEXT + "a" + END + MODEL + "f" + CALL + "{}" + END + "b",
                                                    parse_special=True)]
        self.assertEqual(seen, [False] * 4 + [False, False, True, True, True, True, False])

    def test_loop_guard_suffixes(self):
        svc, _, tok = service("x")
        flat = [t["function"] for t in TOOLS]
        ids, thinks = svc._keep_suffix(flat, {})
        self.assertFalse(thinks)
        self.assertEqual(ids[:2], [tok.special_tokens[EOS], tok.special_tokens[USER]])   # (after its own END)
        key = ("get_weather", loop_guard.args_key({"city": "Paris"}))
        out = svc._block_suffixes({key: loop_guard.ToolRepeat("get_weather", 3)}, flat, {})
        self.assertIsNotNone(out)
        _, ids, thinks = out[key]
        self.assertFalse(thinks)
        self.assertEqual(ids[:2], [tok.special_tokens[EOS], tok.special_tokens[TOOL]])
        self.assertIn(b"get_weather", bytes(t for t in ids if t < 256))

    def test_a_tool_results_picture_is_a_note(self):
        svc, eng, tok = service(MODEL + TEXT + "ok" + END)
        self.assertEqual(svc.text_only(), "Inkling-2-Small on maya-ik2")
        eng.info = {"engine": "inkling2", "engine_kind": "inkling2"}      # (the AMD engine's INFO)
        self.assertEqual(svc.text_only(), "Inkling-2-Small on maya-inkling2")
        msgs, tools, kw = openai_to_messages({"messages": ROUND, "tools": TOOLS})
        ids, _, _ = svc.prepare(msgs, tools, kw, 100)                    # (no error: the agent's turn goes on)
        self.assertIn(server.HY4_PICTURE_NOTE, tok.decode(ids))


if __name__ == "__main__":
    unittest.main()
