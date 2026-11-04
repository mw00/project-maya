"""MAYA_PROMPT_MEMO (#813): prompt preparation by increment gives exactly the token ids the whole-way preparation does.

Every model's chat template (DeepSeek-V4.1, GLM-5.x, MiniMax-M3.1, Qwen3.8) with a tokenizer carrying that model's own
special literals: multi-turn agent histories (appends, an edit in the middle, a literal quoted in a message appearing
and going, tool calls and results, <think> blocks, non-ASCII, segments long enough for the tokenizer's segment cache,
re-sent and branched histories), each prompt prepared with the memo on and off - the ids must be equal.  Also
LiteralMemo == mark_literals, deepseek_restore_points_fast == deepseek_restore_points, join_ids_memo == ','.join and
the loop guard's _gist by text == _gist.

A real pack tokenizer runs the same histories too when MAYA_TEST_TOKENIZERS names its folder(s) (os.pathsep between
them; the template is picked from its tokenizer.json's `pre`).  The lent experts (DS4_LENT_RAM_GB) are engine code on
the GPU only - nothing of them runs without one.

    python -m unittest serve.test_prompt_memo
"""
from __future__ import annotations

import collections
import copy
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import loop_guard  # noqa: E402
from serve.frontend import ChatTemplate, LiteralMemo, literal_tags, mark_literals, openai_to_messages  # noqa: E402
from serve.server import (MockEngine, Service, deepseek_restore_points, deepseek_restore_points_fast,  # noqa: E402
                          join_ids_memo)
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer  # noqa: E402

# (template, pre-tokenizer, CONTROL literals (type 3), USER_DEFINED literals (type 4)) - each model's own
MODELS = {
    "deepseek": ("tools/ds4/chat_template.jinja", "deepseek3",
                 ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
                  "<｜deepseek_image｜>", "｜DSML｜"],
                 ["<think>", "</think>", "<tool_result>", "</tool_result>"]),
    "glm": ("tools/glm_chat_template.jinja", "glm4",
            ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>",
             "<|begin_of_image|>", "<|end_of_image|>", "<|image|>"],
            ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
             "</arg_value>", "<tool_response>", "</tool_response>", "<tools>", "</tools>"]),
    "glm5": ("serve/glm5_chat_template.jinja", "glm4",
             ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>"],
             ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
              "</arg_value>", "<tool_response>", "</tool_response>"]),
    "minimax": ("tools/mm3/chat_template.jinja", "minimax",
                ["]~!b[", "]~b]", "[e~[", "]!d~[", "]!p~[", "]<]minimax[>[", "]<]image[>["],
                ["<mm:think>", "</mm:think>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    "qwen": ("serve/chat_template.jinja", "qwen35",
             ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"],
             ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]),
}

CORPUS = ("def read_file(path):\n    with open(path, encoding='utf-8') as f:\n        return f.read()\n\n"
          "The quick brown fox jumps over the lazy dog. Let me look at the file and then run the tests again.\n"
          "for i in range(10):\n    print(i, 'value', i * 2)\n# a comment about the function and its arguments\n"
          "Ünïcödé text: naïve café, 日本語のテキスト, emoji 🚀✓ and numbers 1234567 89.5\n") * 3

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace.",
                                           "parameters": {"type": "object", "properties": {
                                               "path": {"type": "string", "description": "the file's path"}},
                                               "required": ["path"]}}},
         {"type": "function", "function": {"name": "run", "description": "Run a shell command.",
                                           "parameters": {"type": "object", "properties": {
                                               "cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def train_merges(text: str, n: int) -> list[tuple[str, str]]:
    """A small byte-level BPE merge list learned from `text` (enough to make the pieces multi-character)."""
    words = collections.Counter()
    for w in text.replace("\n", " \n ").split(" "):
        if w:
            words[tuple(BYTE_TO_UNICODE[b] for b in (" " + w).encode("utf-8"))] += 1
    merges = []
    for _ in range(n):
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append((a, b))
        nw = collections.Counter()
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            nw[tuple(out)] += c
        words = nw
    return merges


def make_tokenizer(pre: str, control: list[str], user: list[str]) -> Tokenizer:
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)]
    merges = []
    for a, b in train_merges(CORPUS, 160):
        if a + b not in tokens:
            tokens.append(a + b)
        merges.append(f"{a} {b}")
    types = [1] * len(tokens)
    for s in control + user:
        tokens.append(s)
        types.append(3 if s in control else 4)
    return Tokenizer(tokens, merges, types, pre)


def call(ident: str, name: str, args: dict) -> dict:
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def histories(lit: str, close: str):
    """A coding agent's conversation, prompt by prompt, as a client re-sends it: [(messages, tools), ...].
    lit: one of the model's special literals, quoted in messages; close: a think-close literal."""
    code = "".join(f"def f{i}(x):\n    return x * {i}  # ünïcödé {i}\n" for i in range(140))   # (> 2 KB a segment)
    sys_ = {"role": "system", "content": "You are a careful coding agent. Work in small steps."}
    u1 = {"role": "user", "content": "Read src/app.py and tell me what f7 does."}
    a1 = {"role": "assistant", "content": "", "reasoning_content": "I should read the file first.",
          "tool_calls": [call("c1", "read_file", {"path": "src/app.py"})]}
    t1 = {"role": "tool", "tool_call_id": "c1", "content": code}
    a2 = {"role": "assistant", "content": f"f7 multiplies by 7. The template writes {lit} between turns, by the way."}
    u2 = {"role": "user", "content": f"Why does the log show `{lit}` and `{close}` in the text?"}
    a3 = {"role": "assistant", "content": "", "reasoning_content": f"The user quotes {lit} literally.",
          "tool_calls": [call("c2", "run", {"cmd": f"grep -n '{lit}' log.txt"})]}
    t2 = {"role": "tool", "tool_call_id": "c2", "content": f"12: {lit}hello{close}\n13: plain line 日本語 🚀\n" * 20}
    u3 = {"role": "user", "content": "Thanks. Now run the tests."}
    a4 = {"role": "assistant", "content": f"<think>quoted thinking with {lit}</think>The tests pass."}
    u1_edit = {"role": "user", "content": "Read src/app.py and tell me what f8 does, please."}
    u_lit_first = {"role": "user", "content": f"{lit} at the very start of a message"}
    steps = [
        [sys_, u1],
        [sys_, u1, a1, t1],
        [sys_, u1, a1, t1, a2],                               # the first literal appears
        [sys_, u1, a1, t1, a2, u2],
        [sys_, u1, a1, t1, a2, u2, a3, t2],
        [sys_, u1, a1, t1, a2, u2, a3, t2, u3],
        [sys_, u1, a1, t1, a2, u2, a3, t2, u3, a4, {"role": "user", "content": "And f9?"}],
        [sys_, u1_edit, a1, t1, a2, u2, a3, t2, u3],          # an edit near the start
        [sys_, u1, a1, t1],                                  # back before the literal (it went)
        [sys_, u1, a1, t1, a2, u2, a3, t2],                  # sent again, the same
        [sys_, u_lit_first],                                 # a literal right at the start of a message
        [sys_, u_lit_first, {"role": "assistant", "content": "ok"}, u1],
        [sys_, u1, a1, t1, {"role": "user", "content": "short " + close}],
    ]
    for s in steps:
        yield s, TOOLS
    for s in steps[:3]:
        yield s, None


def fuzz_histories(rng: random.Random, lits: list[str], n: int):
    """Random appends, edits, truncations and branches over a pool of fragments with literals and near-literals."""
    frags = ["hello", "日本語", "🚀", "  spaced  ", "\n\n", "x = 1\n", "a" * 2100] + lits + [s[:-1] for s in lits if len(s) > 2]
    def text():
        return "".join(rng.choice(frags) for _ in range(rng.randint(1, 6)))
    convo = [{"role": "system", "content": text()}, {"role": "user", "content": text()}]
    seen = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            convo = convo + [{"role": "assistant", "content": text()}, {"role": "user", "content": text()}]
        elif r < 0.65 and len(convo) > 2:
            i = rng.randrange(1, len(convo))
            convo = convo[:i] + [{**convo[i], "content": text()}] + convo[i + 1:]
        elif r < 0.8 and len(convo) > 3:
            convo = convo[:rng.randrange(2, len(convo))]
            if convo[-1]["role"] != "user":
                convo = convo + [{"role": "user", "content": text()}]
        elif seen:
            convo = copy.deepcopy(rng.choice(seen))
        seen.append(copy.deepcopy(convo))
        yield copy.deepcopy(convo)


class _MemoEnv:
    def setUp(self):
        self.saved = os.environ.pop("MAYA_PROMPT_MEMO", None)

    def tearDown(self):
        os.environ.pop("MAYA_PROMPT_MEMO", None)
        if self.saved is not None:
            os.environ["MAYA_PROMPT_MEMO"] = self.saved


def service(name: str, tok=None) -> Service:
    path, pre, control, user = MODELS[name]
    tok = tok or make_tokenizer(pre, control, user)
    return Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / path))


def ids_both(svc: Service, messages, tools, mode: str = "1"):
    """The prompt's ids the whole way (memo off) and by increment (the memo kept on the service between calls)."""
    req = {"messages": copy.deepcopy(messages)}
    if tools:
        req["tools"] = copy.deepcopy(tools)
    deepseek = getattr(svc.template, "deepseek", False)
    os.environ.pop("MAYA_PROMPT_MEMO", None)
    m, t, kw = openai_to_messages(copy.deepcopy(req), deepseek=deepseek)
    full = svc.render_ids(m, t, kw)
    os.environ["MAYA_PROMPT_MEMO"] = mode
    m, t, kw = openai_to_messages(copy.deepcopy(req), deepseek=deepseek)
    memo = svc.render_ids(m, t, kw)
    os.environ.pop("MAYA_PROMPT_MEMO", None)
    return full, memo


class SameIds(_MemoEnv, unittest.TestCase):
    def check_model(self, name: str, tok=None):
        svc = service(name, tok)
        lits = sorted(svc.literals)
        self.assertTrue(lits, "the tokenizer must carry special literals")
        lit = next((s for s in lits if "User" in s or "user" in s or "~b]" in s or "im_start" in s), lits[0])
        close = "</think>" if "</think>" in lits else lits[-1]
        for k, (messages, tools) in enumerate(histories(lit, close)):
            full, memo = ids_both(svc, messages, tools)
            self.assertEqual(memo, full, f"{name}: step {k}: the memo's ids differ from the whole prompt's")
        self.assertTrue(svc._prompt_memo.usable(), f"{name}: the memo must run (not fall back) on this tokenizer")
        rng = random.Random(813 + len(name))
        for k, messages in enumerate(fuzz_histories(rng, lits[:6], 40)):
            full, memo = ids_both(svc, messages, None)
            self.assertEqual(memo, full, f"{name}: fuzz step {k}: the memo's ids differ")

    def test_deepseek(self):
        self.check_model("deepseek")

    def test_glm(self):
        self.check_model("glm")

    def test_glm5(self):
        self.check_model("glm5")

    def test_minimax(self):
        self.check_model("minimax")

    def test_qwen(self):
        self.check_model("qwen")

    def test_verify_mode_passes_and_says_so(self):
        svc = service("deepseek")
        lit = "<｜User｜>"
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for messages, tools in histories(lit, "</think>"):
                full, memo = ids_both(svc, messages, tools, mode="verify")
                self.assertEqual(memo, full)
        self.assertIn("prompt memo verified", out.getvalue())

    def test_off_by_default_the_memo_is_never_built(self):
        svc = service("glm")
        req = {"messages": [{"role": "user", "content": "hi <|user|> there"}]}
        m, t, kw = openai_to_messages(req)
        svc.render_ids(m, t, kw)
        self.assertIsNone(getattr(svc, "_prompt_memo", None))
        self.assertIsNone(getattr(svc, "_lit_memo", None))

    def test_a_suffix_takes_the_whole_way(self):
        svc = service("qwen")
        os.environ["MAYA_PROMPT_MEMO"] = "1"
        m, t, kw = openai_to_messages({"messages": [{"role": "user", "content": "hi"}]})
        a = svc.render_ids(m, t, kw, suffix="<tool_call>")
        os.environ.pop("MAYA_PROMPT_MEMO")
        m, t, kw = openai_to_messages({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(a, svc.render_ids(m, t, kw, suffix="<tool_call>"))


@unittest.skipUnless(os.environ.get("MAYA_TEST_TOKENIZERS"), "MAYA_TEST_TOKENIZERS names no pack tokenizer")
class SameIdsRealTokenizers(SameIds):
    """The same histories with real packs' tokenizers (MAYA_TEST_TOKENIZERS=<pack>/tokenizer[<os.pathsep>...])."""
    def test_real(self):
        by_pre = {"deepseek3": ["deepseek"], "glm4": ["glm", "glm5"], "glm5": ["glm", "glm5"], "minimax": ["minimax"],
                  "qwen35": ["qwen"]}
        for d in os.environ["MAYA_TEST_TOKENIZERS"].split(os.pathsep):
            tok = Tokenizer.from_dir(d)
            for name in by_pre.get(tok.pre, []):
                with self.subTest(tokenizer=d, template=name):
                    self.check_model(name, tok)


class LiteralMemoSame(unittest.TestCase):
    def test_the_same_as_mark_literals(self):
        for name, (_, pre, control, user) in MODELS.items():
            tags = literal_tags(control + user)
            memo = LiteralMemo(tags)
            lit = control[2]
            for k, (messages, tools) in enumerate(histories(lit, "</think>")):
                m, t, _ = openai_to_messages({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
                want = mark_literals(copy.deepcopy(m), copy.deepcopy(t), tags)
                for _ in range(2):   # (the second time from the memo)
                    with self.subTest(model=name, step=k):
                        self.assertEqual(memo.mark_literals(copy.deepcopy(m), copy.deepcopy(t)), want)

    def test_a_small_cap_keeps_the_answers(self):
        tags = literal_tags(["<|user|>", "<think>", "</think>", "<tool_call>"])
        memo = LiteralMemo(tags, cap=50)   # (generations turn over every few strings)
        rng = random.Random(5)
        for messages in fuzz_histories(rng, list(tags), 60):
            self.assertEqual(memo.mark_literals(copy.deepcopy(messages), None),
                             mark_literals(copy.deepcopy(messages), None, tags))


class RestorePointsFast(unittest.TestCase):
    def test_the_same_as_the_original(self):
        rng = random.Random(7)
        for _ in range(3000):
            n = rng.randint(0, 3000)
            ids = [rng.choice((1, 2, 3, 4, 5, 6, 7)) if rng.random() < 0.002 else rng.randint(10, 99) for _ in range(n)]
            user, system = rng.choice(((1, 2), (1, None), (None, 2), (3, 1)))
            tail = rng.choice((0, 64, 512))
            least = rng.choice((0, 16, 256))
            self.assertEqual(deepseek_restore_points_fast(ids, user, system, tail, least),
                             deepseek_restore_points(ids, user, system, tail, least), (n, user, system, tail, least))

    def test_default_least(self):
        ids = [9] * 900 + [1] + [9] * 900 + [2] + [9] * 900 + [1] + [9] * 600
        self.assertEqual(deepseek_restore_points_fast(ids, 1, 2), deepseek_restore_points(ids, 1, 2))


class JoinIdsMemo(unittest.TestCase):
    def test_the_same_string_as_join(self):
        rng = random.Random(11)
        memo = {}
        ids = [rng.randint(0, 200000) for _ in range(9000)]
        for _ in range(200):
            r = rng.random()
            if r < 0.4:
                ids = ids + [rng.randint(0, 200000) for _ in range(rng.randint(0, 5000))]
            elif r < 0.6 and ids:
                i = rng.randrange(len(ids))
                ids = ids[:i] + [rng.randint(0, 200000)] + ids[i + 1:]
            elif r < 0.8:
                ids = ids[:rng.randrange(len(ids) + 1)]
            else:
                ids = [rng.randint(0, 9) for _ in range(rng.choice((0, 1, 4096, 8192, 4097)))]
            for block in (4096, 7):
                with self.subTest(n=len(ids), block=block):
                    self.assertEqual(join_ids_memo(memo.setdefault(block, {}), ids, block),
                                     ",".join(str(t) for t in ids))

    def test_a_tuple_of_ids(self):
        self.assertEqual(join_ids_memo({}, (1, 2, 3)), "1,2,3")


class GistByText(_MemoEnv, unittest.TestCase):
    def test_the_same(self):
        samples = ["bytes received: 12 ok", "日本語 99 🚀 x", "", "  \t42\n", "a" * 5000 + "7"]
        want = [loop_guard._gist(s) for s in samples]
        os.environ["MAYA_PROMPT_MEMO"] = "1"
        self.assertEqual([loop_guard._gist(s) for s in samples], want)
        self.assertEqual([loop_guard._gist(s) for s in samples], want)


if __name__ == "__main__":
    unittest.main()
