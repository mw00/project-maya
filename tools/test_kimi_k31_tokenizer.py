"""tools/test_kimi_k31_tokenizer.py - the `kimi-k2` transcription in strata_tokenizer, checked without the model.

The id-for-id check against Moonshot's tokenizer needs the release's tiktoken.model and tokenization_kimi.py, which
this repository does not ship: tools/k31/tokenizer_check.py runs it.  What is checked here is the part that must hold
before any vocabulary exists:

  * the pattern is tokenization_kimi.py's `pat_str`, and it is compiled so that `&&` intersects (regex V1);
  * Han splits apart from everything, its own Lm characters included;
  * kimi_k2_slices cuts where tokenization_kimi.py's loops cut (transcribed below as the oracle), and encode()
    encodes each slice alone, so a BPE merge or a special token across a cut does not form;
  * UTF-16 surrogates come out as tiktoken makes them, and `plain` spans follow;
  * the heap BPE for long pieces gives the merge loop's result;
  * the classes read as tiktoken's Unicode 16.0 tables do (KIMI_K2_UNICODE): a code point assigned since is in none of
    them, U+0295 is Ll, and this `regex` fits the table (tokenizer_check.py compares each class with tiktoken's).
"""
from __future__ import annotations

import pathlib
import random
import sys
import unittest
from unittest import mock

import regex

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import strata_tokenizer as st  # noqa: E402
from strata_tokenizer import BYTE_TO_UNICODE, KIMI_K2_PATTERN, Tokenizer  # noqa: E402


def hf_pieces(text: str, chunk: int, run: int) -> list[str]:
    """tokenization_kimi.py `_encode_text_piece` + `_split_whitespaces_or_nonwhitespaces`, as written there."""
    out = []
    for i in range(0, len(text), chunk):
        s = text[i:i + chunk]
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0
        for j in range(len(s)):
            is_now_space = s[j].isspace()
            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > run:
                    out.append(s[slice_start:j])
                    slice_start = j
                    current_slice_len = 1
        out.append(s[slice_start:])
    return out


def tiny(extra_tokens=(), merges=(), specials=()):
    tokens = list(BYTE_TO_UNICODE.values()) + list(extra_tokens) + list(specials)
    types = [1] * (len(tokens) - len(specials)) + [3] * len(specials)
    return Tokenizer(tokens, list(merges), types, pre="kimi-k2")


class Pattern(unittest.TestCase):
    def setUp(self):
        self.re = regex.compile(KIMI_K2_PATTERN, regex.V1)

    def test_is_pat_str(self):
        alts = KIMI_K2_PATTERN.split("|")
        self.assertEqual(alts[0], r"[\p{Han}]+")
        self.assertIn(r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+", alts[1])
        self.assertEqual(alts[-5:], [r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n]*", r"\s*[\r\n]+", r"\s+(?!\S)",
                                     r"\s+"])
        self.assertEqual(tiny()._re.pattern, KIMI_K2_PATTERN)
        self.assertTrue(tiny()._re.flags & regex.V1)

    def test_intersection_needs_v1(self):
        # V1: "a" is a letter that is not Han, "&&" punctuation.  V0 reads `&&` as two ampersands in the class.
        self.assertEqual(self.re.findall("a&&b"), ["a", "&&", "b"])
        self.assertNotEqual(regex.compile(KIMI_K2_PATTERN).findall("x&y\u4e00z"), self.re.findall("x&y\u4e00z"))

    def test_han_apart(self):
        self.assertEqual(self.re.findall("\u6f22\u3005\u5b57"), ["\u6f22\u3005\u5b57"])     # U+3005 is Han (Lm)
        self.assertEqual(self.re.findall("\u304b\u306a\u6f22\u5b57"), ["\u304b\u306a", "\u6f22\u5b57"])
        self.assertEqual(self.re.findall("Hello\u4e16\u754cworld"), ["Hello", "\u4e16\u754c", "world"])
        self.assertEqual(self.re.findall(" \u4e16"), [" ", "\u4e16"])            # no leading space on a Han run

    def test_lossless_and_digits(self):
        text = "don't I'LL 1234567 \u0661\u0662\u0663\u0664 \x1c\x1d  \n\n\r\n x\u0301 \U0001F44D\U0001F3FD <|sep|>"
        self.assertEqual("".join(self.re.findall(text)), text)
        self.assertEqual(self.re.findall("1234567"), ["123", "456", "7"])
        self.assertEqual(self.re.findall("I'LL"), ["I'LL"])


class Slices(unittest.TestCase):
    def test_against_tokenization_kimi(self):
        rng = random.Random(3)
        alphabet = ["a", "b", " ", "\n", "\x1c", "\u3000", "\u4e00", "<", "|", ">"]
        with mock.patch.object(st, "KIMI_K2_RUN", 5), mock.patch.object(st, "KIMI_K2_CHUNK", 23):
            for _ in range(2000):
                text = "".join(rng.choice(alphabet) * rng.randint(1, 9) for _ in range(rng.randint(0, 12)))
                got = [text[a:b] for a, b in st.kimi_k2_slices(text)]
                self.assertEqual(got, [p for p in hf_pieces(text, 23, 5) if p] if text else [], repr(text))

    def test_real_limits(self):
        text = "x" * 25_001 + " " * 3 + "y" * 25_000
        self.assertEqual([b - a for a, b in st.kimi_k2_slices(text)], [25_000, 25_004])
        self.assertEqual(len(st.kimi_k2_slices("z" * 400_001)), 17)          # 16 runs of 25,000, then the chunk cut

    def test_encode_cuts_merges_and_specials(self):
        tk = tiny(["ab"], ["a b"], ["<|sep|>"])
        ab, sep = tk.ids["ab"], tk.ids["<|sep|>"]
        self.assertIn(ab, tk.encode("aaaaaaab"))
        self.assertIn(sep, tk.encode("xx<|sep|>", parse_special=True))
        with mock.patch.object(st, "KIMI_K2_RUN", 7):
            self.assertNotIn(ab, tk.encode("aaaaaaab"))                         # cut between a and b
            self.assertNotIn(sep, tk.encode("xx<|sep|>", parse_special=True))   # a special across a cut is text
            self.assertIn(sep, tk.encode("x <|sep|>", parse_special=True))      # a run of 7: no cut
            # a plain span is clipped to each slice, and stays plain
            self.assertNotIn(sep, tk.encode("a <|sep|>", parse_special=True, plain=[(2, 9)]))
            self.assertEqual(tk.decode(tk.encode("aaaaaaab xx<|sep|>", parse_special=True)), "aaaaaaab xx<|sep|>")


class Surrogates(unittest.TestCase):
    def test_as_tiktoken(self):
        self.assertEqual(st._tiktoken_text("a\ud83d\ude00b")[0], "a\U0001F600b")
        self.assertEqual(st._tiktoken_text("a\ud800b\udfff")[0], "a\ufffdb\ufffd")
        fixed, spans = st._tiktoken_text("\ud83d\ude00<|sep|>\ud83d\ude00x", [(2, 9), (9, 12)])
        self.assertEqual(fixed[spans[0][0]:spans[0][1]], "<|sep|>")
        self.assertEqual(fixed[spans[1][0]:spans[1][1]], "\U0001F600x")
        tk = tiny()
        self.assertEqual(tk.decode(tk.encode("\ud800 ok \ud83d\ude00")), "\ufffd ok \U0001F600")


class HeapBpe(unittest.TestCase):
    def test_same_as_the_loop(self):
        merges = ["a a", "aa a", "b b", "aa b", "a b", "ab a", "c a", "aaa a", "bb c", "aab aa"]
        extra = sorted({m.replace(" ", "") for m in merges})
        tk = tiny(extra, merges)
        loop = Tokenizer(tk.tokens, merges, pre="qwen35")                      # the merge loop, never the heap
        rng = random.Random(5)
        for _ in range(400):
            w = "".join(rng.choice("aaabbc") for _ in range(rng.randint(65, 400)))
            self.assertEqual(tk._bpe(w), loop._bpe(w), w)


class Unicode16(unittest.TestCase):
    def test_tables_fit(self):
        self.assertEqual(st.kimi_k2_class_digest(st.kimi_k2_classes()), st.KIMI_K2_CLASS_DIGEST)
        with mock.patch.object(st, "KIMI_K2_CLASS_DIGEST", "0" * 16), mock.patch.object(st, "_KIMI_K2_FITS", []):
            with self.assertRaisesRegex(ValueError, "Unicode 16.0"):
                tiny()

    def test_every_class_listed(self):
        found = set(regex.findall(r"\\p\{\w+\}|\\s", KIMI_K2_PATTERN))
        self.assertTrue(found and all(any(f in c for c in st.KIMI_K2_CLASSES) for f in found), found)
        for c in st.KIMI_K2_CLASSES[:3]:
            self.assertIn(c, KIMI_K2_PATTERN)

    def test_stand_ins(self):
        tk = tiny()
        ext_j = "\U000323B0"                                   # CJK Extension J, Unicode 17.0: Cn to tiktoken
        self.assertEqual(tk._pieces(ext_j + "'s"), [ext_j + "'", "s"])
        self.assertEqual(tk._re.findall(ext_j + "'s"), [ext_j, "'s"])       # `regex`'s own tables: Han, then 's
        self.assertEqual(tk._pieces("\u0295Ab"), ["\u0295", "Ab"])           # Ll, as in 16.0 (Lo since 17.0)
        self.assertEqual(tk._pieces("\u4e2d\U000323B0\u4e2d"), ["\u4e2d", "\U000323B0", "\u4e2d"])
        text = "a\U000323B0b \u0295x \U00016EA0's"
        self.assertEqual("".join(tk._pieces(text)), text)
        self.assertEqual(tk.decode(tk.encode(text)), text)


class Scheme(unittest.TestCase):
    def test_accepted_and_named(self):
        self.assertEqual(tiny().pre, "kimi-k2")
        with self.assertRaisesRegex(ValueError, "kimi-k2"):
            Tokenizer(list(BYTE_TO_UNICODE.values()), [], pre="kimi")


if __name__ == "__main__":
    unittest.main()
