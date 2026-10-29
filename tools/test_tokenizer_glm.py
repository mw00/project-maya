"""tools/test_tokenizer_glm.py - the glm4/glm5 pre-tokenizer transcription, checked without a model.

The real GLM-5.3 tokenizer round trip needs the real GGUF (154,880 vocab), which nothing in this
repo ships.  What CAN be checked locally is the part that must be right BEFORE any vocab exists:

  * the CHATGLM4 regex is a verbatim transcription of llama-vocab.cpp L406-410 (the two
    load-bearing differences from qwen35: `\\p{N}{1,3}` digit grouping, and NO `\\p{M}` in the
    letter class);
  * the split is LOSSLESS (decode(encode(s)) == s at the piece level: the pieces concatenate back
    to the input exactly - the byte layer is lossless, so a dropped character is a split bug);
  * the digit grouping actually splits a 4-digit run 3+1 (qwen35 keeps it whole);
  * ignore_merges: a pre-token piece that is an exact vocabulary token is emitted whole, and a
    piece outside the vocabulary falls back to BPE (llama-vocab.cpp L633-637);
  * an unknown `pre` scheme is refused.

The vocab used here is the byte-unicode alphabet plus a handful of constructed tokens - enough to
exercise every branch, not a model.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_tokenizer import BYTE_TO_UNICODE, GLM_PATTERN, QWEN35_PATTERN, Tokenizer  # noqa: E402


def tiny_vocab() -> tuple[list[str], list[str]]:
    """A vocab that covers the byte layer (so BPE always bottoms out) plus whole-word tokens
    the ignore_merges branch can hit."""
    tokens = list(BYTE_TO_UNICODE.values())
    tokens += ["hello", "world", "hello world", "12345", " Head"]   # exact-hit candidates
    merges = []
    return tokens, merges


class GlmPatternTest(unittest.TestCase):
    def setUp(self):
        tokens, merges = tiny_vocab()
        self.tk = Tokenizer(tokens, merges, pre="glm5")

    def test_regex_is_the_chatglm4_transcription(self):
        # the two load-bearing differences from qwen35, as substrings
        self.assertIn(r"\p{N}{1,3}", GLM_PATTERN)                 # digit grouping
        self.assertNotIn(r"\p{M}", GLM_PATTERN)                   # no combining-mark class
        self.assertIn(r"\p{N}", QWEN35_PATTERN)
        self.assertIn(r"\p{M}", QWEN35_PATTERN)
        # the shared spine (contractions, letters, punctuation, whitespace) is the same shape
        self.assertIn(r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])", GLM_PATTERN)

    def test_pieces_are_lossless(self):
        corpus = [
            "hello world",
            "plain text 12345 and 9876543210",
            "UTF-8: héllo wörld, combining á and é",
            "emoji 👍🏽 and CJK 你好世界",
            "tabs\tand\nnewlines\r\n  spaces   ",
            "contractions: don't I'll we've I'm",
            "punctuation!!! ...？。，、",
        ]
        for text in corpus:
            pieces = self.tk._re.findall(text)
            self.assertEqual("".join(pieces), text, "the GLM split dropped or reordered a character")

    def test_digit_grouping_3_plus_1(self):
        # qwen35 splits digits ONE per piece (the merges reassemble them); glm splits a run into
        # groups of at most 3 - a different BPE input, therefore different tokens for long numbers
        pieces = self.tk._re.findall("9876543210")
        self.assertEqual(pieces, ["987", "654", "321", "0"])
        qwen = __import__("regex").compile(QWEN35_PATTERN)
        self.assertEqual(qwen.findall("9876543210"), list("9876543210"))

    def test_ignore_merges_emits_whole_vocab_hits(self):
        # "hello" is in the tiny vocab: the glm path emits it as ONE token (no byte-BPE)
        ids = self.tk.encode("hello")
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.tk.tokens[ids[0]], "hello")

    def test_ignore_merges_falls_back_to_bpe_off_vocab(self):
        # "helio" is NOT in the vocab: the piece falls back to byte-level BPE over the alphabet
        ids = self.tk.encode("helio")
        self.assertTrue(ids)
        self.assertEqual("".join(self.tk.token_bytes(i).decode("latin-1")
                                 for i in ids).encode("latin-1").decode("utf-8"), "helio")

    def test_qwen_path_has_no_ignore_merges(self):
        tokens, merges = tiny_vocab()
        tk = Tokenizer(tokens, merges, pre="qwen35")
        # the same text that is one whole token under glm splits byte-wise under qwen35
        self.assertNotEqual(len(tk.encode("hello")), 1)


class FromDir(unittest.TestCase):
    """#27: the server built its tokenizer from a pack's tokenizer/ without the pre-tokenizer, so a GLM pack split
    numbers the qwen35 way (one digit a piece).  from_dir reads it from tokenizer.json."""

    def write(self, meta):
        import json
        import tempfile
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d, True)
        tokens, _ = tiny_vocab()
        (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
        (d / "merges.txt").write_text("h e", encoding="utf-8")   # a pack's file always has merges
        (d / "token_type.json").write_text(json.dumps([1] * len(tokens)))
        if meta is not None:
            (d / "tokenizer.json").write_text(json.dumps(meta), encoding="utf-8")
        return d

    def test_glm_pack_groups_digits(self):
        tk = Tokenizer.from_dir(self.write({"pre": "glm4", "special_ids": {"tokenizer.ggml.eos_token_id": 3}}))
        self.assertEqual(tk.pre, "glm4")
        self.assertEqual(tk.special_ids, {"tokenizer.ggml.eos_token_id": 3})
        self.assertEqual(tk._re.findall("504 9876"), ["504", " ", "987", "6"])

    def test_without_tokenizer_json_stays_qwen35(self):
        self.assertEqual(Tokenizer.from_dir(self.write(None)).pre, "qwen35")


class PlainSpans(unittest.TestCase):
    """encode(..., plain=spans): a literal that starts inside a span is text, BPE'd with the text around it; every
    other literal (a CONTROL one with parse_special, a USER_DEFINED one always) is still its token."""

    def setUp(self):
        tokens = list(BYTE_TO_UNICODE.values()) + ["<think>", "<|user|>"]
        types = [1] * (len(tokens) - 2) + [4, 3]
        self.tk = Tokenizer(tokens, [], types, pre="glm5")
        self.think, self.user = len(tokens) - 2, len(tokens) - 1

    def test_spans(self):
        text = "<|user|>a<think>b<|user|>c<think>"
        self.assertEqual(self.tk.encode(text, parse_special=True).count(self.user), 2)
        got = self.tk.encode(text, parse_special=True, plain=[(9, 16), (17, 25)])   # the middle <think> and <|user|>
        self.assertEqual([t for t in got if t in (self.think, self.user)], [self.user, self.think])
        self.assertEqual(got[0], self.user)
        self.assertEqual(got[-1], self.think)
        self.assertEqual(b"".join(self.tk.token_bytes(t) for t in got[1:-1]), b"a<think>b<|user|>c")
        # unsorted spans, and no spans at all, as before
        self.assertEqual(self.tk.encode(text, parse_special=True, plain=[(17, 25), (9, 16)]), got)
        self.assertEqual(self.tk.encode(text, parse_special=True, plain=[]), self.tk.encode(text, parse_special=True))


class PreSchemeGuard(unittest.TestCase):
    def test_unknown_scheme_refused(self):
        with self.assertRaises(ValueError):
            Tokenizer(list(BYTE_TO_UNICODE.values()), [], pre="llama3")

    def test_glm_schemes_accepted(self):
        for pre in ("glm4", "glm5"):
            Tokenizer(list(BYTE_TO_UNICODE.values()), [], pre=pre)


if __name__ == "__main__":
    unittest.main()
