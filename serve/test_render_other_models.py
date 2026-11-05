"""serve/test_render_other_models.py - the Qwen3.8 prompt fixes (tools in OpenAI's form, past thinking sent back, the
server's thinking close) leave every OTHER model's prompt as it was: DeepSeek-V4.1, GLM-4.x/5.x and MiniMax-M3.1
conversations (multi-turn with thinking, tool calls and results, pictures, thinking off and the effort levels) render
to the token ids recorded (count + sha256) in serve/fixtures/render_ids_other_models.json - written by the code before those fixes
(MAYA_WRITE_FIXTURE=1 writes it again).  Each model's own template and special literals, the test tokenizer of
serve/test_prompt_memo.py.

    python -m unittest serve.test_render_other_models -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402
from serve.server import MockEngine, Service  # noqa: E402
from serve.test_prompt_memo import MODELS, TOOLS, make_tokenizer  # noqa: E402

FIXTURE = ROOT / "serve/fixtures/render_ids_other_models.json"
OTHERS = ("deepseek", "glm", "glm5", "minimax")
IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def call(ident, name, args):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def conversations():
    sys_ = {"role": "system", "content": "You are a careful assistant."}
    u1 = {"role": "user", "content": "What is 17*23?"}
    a1 = {"role": "assistant", "content": "391.", "reasoning_content": "17*20 + 17*3 = 340 + 51 = 391.\n"}
    u2 = {"role": "user", "content": "And times 3? ünïcödé 日本語 🚀"}
    a2 = {"role": "assistant", "content": "1173.", "reasoning_content": "391*3 = 1173."}
    u3 = {"role": "user", "content": "Thanks!\n\n```python\nx = 1\n```"}
    ut = {"role": "user", "content": "Read src/app.py, then run the tests."}
    at = {"role": "assistant", "content": "", "reasoning_content": "Read the file first.",
          "tool_calls": [call("c1", "read_file", {"path": "src/app.py"})]}
    tr = {"role": "tool", "tool_call_id": "c1", "content": "def f(x):\n    return x * 7\n"}
    at2 = {"role": "assistant", "content": "It multiplies by 7. Running the tests.", "reasoning_content": "Now run.",
           "tool_calls": [call("c2", "run", {"cmd": "pytest -q"})]}
    tr2 = {"role": "tool", "tool_call_id": "c2", "content": "3 passed"}
    af = {"role": "assistant", "content": "All 3 tests pass.", "reasoning_content": "Done."}
    ui = {"role": "user", "content": [{"type": "text", "text": "What colour is this?"},
                                      {"type": "image_url", "image_url": {"url": IMG}}]}
    ai = {"role": "assistant", "content": "Red.", "reasoning_content": "One red pixel."}
    plain = {"role": "assistant", "content": "391."}
    out = {
        "single": ([sys_, u1], None),
        "multi-turn, reasoning sent back": ([sys_, u1, a1, u2, a2, u3], None),
        "multi-turn, no reasoning": ([sys_, u1, plain, u2], None),
        "tools: calls, results, answer, next": ([sys_, ut, at, tr, at2, tr2, af, u2], TOOLS),
        "tools: open call round": ([sys_, ut, at, tr], TOOLS),
        "picture, then text": ([sys_, ui, ai, u2], None),
    }
    for name, (msgs, tools) in list(out.items()):
        for effort in ("none", "low", "high", "max"):
            if name in ("single", "multi-turn, reasoning sent back", "tools: calls, results, answer, next"):
                out[f"{name} @ {effort}"] = (msgs, tools, effort)
    return {k: (v + (None,) if len(v) == 2 else v) for k, v in out.items()}


def render_all() -> dict:
    got = {}
    for model in OTHERS:
        path, pre, control, user = MODELS[model]
        tok = make_tokenizer(pre, control, user)
        tpl = ChatTemplate(ROOT / path)
        svc = Service(MockEngine(tok, "ok"), tok, tpl)
        deepseek = getattr(tpl, "deepseek", False)
        for name, (msgs, tools, effort) in conversations().items():
            req = {"messages": copy.deepcopy(msgs)}
            if tools:
                req["tools"] = copy.deepcopy(tools)
            if effort is not None:
                req["reasoning_effort"] = effort
            try:
                m, t, kw = openai_to_messages(req, deepseek=deepseek)
                ids = svc.render_ids(m, t, kw)
                got[f"{model} | {name}"] = [len(ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()]
            except Exception as e:  # noqa: BLE001  (a refusal is part of the record too)
                got[f"{model} | {name}"] = f"{type(e).__name__}: {e}"
    return got


class OtherModelsUnchanged(unittest.TestCase):
    def test_every_other_model_renders_the_recorded_ids(self):
        got = render_all()
        if os.environ.get("MAYA_WRITE_FIXTURE") == "1":
            FIXTURE.write_text("{\n" + ",\n".join(f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v)}"
                                                  for k, v in got.items()) + "\n}\n", encoding="utf-8")
        want = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(sorted(got), sorted(want))
        for key in want:
            with self.subTest(key):
                self.assertEqual(got[key], want[key])

    def test_no_other_template_counts_as_qwens(self):
        for model in OTHERS:
            t = ChatTemplate(ROOT / MODELS[model][0])
            self.assertFalse(getattr(t, "qwen", False), model)
            self.assertFalse(getattr(t, "keeps_thinking", False), model)


if __name__ == "__main__":
    unittest.main()
