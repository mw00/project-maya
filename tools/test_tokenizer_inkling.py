"""tools/test_tokenizer_inkling.py - Inkling 2's pre-tokenizer, checked without a model.

unsloth's Inkling-2-Small GGUFs name it two ways - "gpt-4o" (UD-Q4_K_XL) and "inkling" (UD-Q3_K_XL, a later
conversion) - for the one split its tokenizer.json has: GPT-4o's o200k regex (MINIMAX_PATTERN), no normalizer, BPE
with ignore_merges.  Both names must read the same; the real 201,024-token vocabulary was checked against the
model's own tokenizer.json (tokenizers) on 442 strings with both files, all equal - no model ships here.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_tokenizer import BYTE_TO_UNICODE, MINIMAX_PATTERN, O200K, Tokenizer  # noqa: E402


def tiny_vocab() -> tuple[list[str], list[str]]:
    tokens = list(BYTE_TO_UNICODE.values()) + ["hello", "Ġworld", "Ġ123"]
    return tokens, []


class InklingPre(unittest.TestCase):
    CORPUS = ["hello world", "hello world 123456 and 9876", "I'M you're THEY'LL", "UTF-8: héllo, combining é",
              "tabs\tand\nnewlines\r\n  spaces   ", "emoji 👍🏽 and CJK 你好世界", "x = f(a)->b; // done"]

    def test_every_name(self):   # (gpt-4o and inkling: unsloth's conversions; inkling2: llama.cpp master's, Mini's files)
        self.assertEqual(O200K, ("gpt-4o", "inkling", "inkling2"))
        tokens, merges = tiny_vocab()
        tks = [Tokenizer(tokens, merges, pre=p) for p in O200K]
        for tk in tks:
            self.assertTrue(tk.ignore_merges)
            self.assertFalse(tk._nfc)                      # (MiniMax's NFC is MiniMax's alone)
            self.assertEqual(tk._re.pattern, MINIMAX_PATTERN)
        a = tks[0]
        for s in self.CORPUS:
            for tk in tks[1:]:
                self.assertEqual(a.encode(s), tk.encode(s), s)
            self.assertEqual(a.decode(a.encode(s)), s, s)

    def test_whole_vocabulary_hits(self):   # (ignore_merges: a piece that is a token is that token)
        tokens, merges = tiny_vocab()
        tk = Tokenizer(tokens, merges, pre="inkling")
        self.assertEqual(tk.encode("hello world"), [tk.ids["hello"], tk.ids["Ġworld"]])

    def test_written_pattern(self):   # (extract: the pattern shipped with the tokenizer folder)
        import strata_tokenizer as S
        tokens, merges = tiny_vocab()
        tk = Tokenizer(tokens, merges, pre="inkling")
        orig = S.Tokenizer.from_gguf
        try:
            S.Tokenizer.from_gguf = staticmethod(lambda path: tk)
            with tempfile.TemporaryDirectory() as d:
                tpl = pathlib.Path(d) / "t.jinja"
                tpl.write_text("{{ messages }}", encoding="utf-8")
                S.extract("unused.gguf", d, chat_template=tpl)
                cfg = json.loads((pathlib.Path(d) / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
                self.assertEqual((cfg["pre"], cfg["ignore_merges"], cfg["pre_pattern"]),
                                 ("inkling", True, MINIMAX_PATTERN))
                self.assertEqual((pathlib.Path(d) / "tokenizer" / "chat_template.jinja").read_text(encoding="utf-8"),
                                 "{{ messages }}")
        finally:
            S.Tokenizer.from_gguf = orig

    def test_unknown_still_refused(self):
        tokens, merges = tiny_vocab()
        with self.assertRaises(ValueError):
            Tokenizer(tokens, merges, pre="o200k")


if __name__ == "__main__":
    unittest.main()
