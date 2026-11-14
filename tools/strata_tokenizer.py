"""tools/strata_tokenizer.py - the tokenizer, as the reference implementation of record.

WHY THIS EXISTS AND WHERE IT SITS.  The GGUF carries the whole tokenizer in metadata: `tokenizer.ggml.model`
= `gpt2`, 248,320 `tokens`, 247,587 `merges`, `token_type`, `pre` = `qwen35`, and a Jinja `chat_template`.
None of that needs the 35 GB of weights, so it is extracted into the pack's `tokenizer/` directory
(docs/pack-format.md §6) and this module is what reads it back.  A C++ port follows and is tested against
this one.

THE DECISIVE PROPERTY IS THE ROUND TRIP.  Byte-level BPE maps each BYTE to a printable unicode character so
that any UTF-8 input is representable with no UNK token.  Every failure mode of that mapping - the wrong
offset for a byte, a merge applied in the wrong order, a pre-tokenizer split that drops a character - still
produces plausible token ids.  `decode(encode(s)) == s` is the check that sees them, and it is exact because
the byte layer is lossless.  It is asserted over a corpus chosen to hit the boundaries: multi-byte UTF-8,
emoji (4-byte), combining marks, whitespace runs, and C0 control bytes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import regex

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0
    controls and the high range that would be invisible) are shifted into 256+n so that no byte is
    unprintable.  The shift is the whole trick and getting its ORDER wrong is invisible in the output ids.
    """
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from the ORACLE rather than from the family resemblance:
# `.ref/llama.cpp/src/llama-vocab.cpp` L396, `case LLAMA_VOCAB_PRE_TYPE_QWEN35`.  The commented-out line
# above it is the `tokenizer.json` original, and llama.cpp's active version differs from it - it spells the
# contraction classes out instead of using `(?i:...)`, which is behaviourally the same.
#
# THE `\p{M}` IS THE WHOLE POINT OF THIS BEING TRANSCRIBED.  The QWEN3 pattern two cases earlier (L389) is
# the same shape with `\p{L}` where this has `[\p{L}\p{M}]` and without `\p{M}` in the punctuation negation.
# Writing the QWEN3 pattern for `qwen35` looks right, round-trips perfectly, and is wrong: a combining mark
# is `\p{M}`, so without it the mark gets swallowed into the following punctuation run and a token like `_j`
# never forms.  It cost 3 strings out of 1875 and only `tokenize_oracle_check` could see it.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `glm4`/`glm5` pre-tokenizer, transcribed from llama-vocab.cpp L406-410 at the vendored pin
# (LLAMA_VOCAB_PRE_TYPE_CHATGLM4).  Two differences from qwen35, both load-bearing:
#   * the digit run is `\p{N}{1,3}` - a 4-digit number splits 3+1, not as one whole run;
#   * the letter class is `\p{L}+` WITHOUT `\p{M}`, AND the punctuation class is
#     `[^\s\p{L}\p{N}]` without `\p{M}` - so a combining mark attaches to the FOLLOWING
#     punctuation run, the exact opposite of the qwen35 `\p{M}` subtlety above.
# AND the encoding semantic differs: llama.cpp sets `ignore_merges = true` for glm4/glm5, so a
# pre-token piece that is an exact vocabulary token is emitted WHOLE (no merges applied); only
# the leftovers go through BPE.  Transcribing the regex alone and keeping the merges would be
# the same class of plausible-but-wrong as the qwen3-for-qwen35 mistake above.
GLM_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


# The `deepseek3` pre-tokenizer (DeepSeek-V3 .. V4.1's tokenizer.json, llama.cpp's LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM):
# THREE Split stages with behavior "Isolated", each applied to every piece the one before produced - digit runs of at
# most 3, then runs of CJK ideographs / kana, then the word pattern - and the text between a stage's matches stays a
# piece of its own (a digit run passes the word pattern untouched that way).  Then the byte-level mapping; merges by
# rank as usual (no ignore_merges).
DEEPSEEK3_PATTERNS = (
    r"\p{N}{1,3}",
    r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)


# The `minimax` pre-tokenizer: MiniMax-M3.1's tokenizer.json verbatim - an NFC normalizer, then one Split (invert,
# "Removed") with this regex, GPT-4o's (o200k) - before the byte-level map.  The NFC runs on each plain segment,
# between the added tokens, as `tokenizers` does for added tokens that are not normalized.
MINIMAX_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `kimi-k2` pre-tokenizer (Kimi K2, K3 and K3.1, which share one 163,840-token vocabulary): tokenization_kimi.py's
# `TikTokenTokenizer.pat_str` verbatim, the pattern tiktoken runs.  It is Rust regex syntax, and `&&` in a class is an
# INTERSECTION there: `[\p{Lu}...\p{M}&&[^\p{Han}]]` is "a letter or mark that is not Han".  Python's `regex` reads the
# same text that way only with the V1 flag; without it `&&` is two literal ampersands, the class turns into something
# else and still splits plausibly.  Han comes first and on its own, so a run of ideographs is never glued to the kana,
# Latin or marks next to it, and Han's own Lm/Mc characters (U+3005, U+303B, U+16FF0) go with the ideographs.  Then the
# byte-level map and merges by rank (the GGUF's merges are the converter's replay of tiktoken's BPE, one per token, so
# a piece that is a whole token comes out whole, as tiktoken's shortcut has it).
KIMI_K2_PATTERN = "|".join((
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
))

# tokenization_kimi.py's `_encode_text_piece` cuts the text before tiktoken sees it, and every cut is a BPE and a
# special-token boundary: into chunks of 400,000 characters (TIKTOKEN_MAX_ENCODE_CHARS), then each chunk wherever a run
# of str.isspace() characters, or of the others, reaches 25,000 (MAX_NO_WHITESPACES_CHARS).  25,000 characters without
# a space is a minified JSON tool result or a base64 blob, so the cut is reached in real prompts, not only in tests.
KIMI_K2_CHUNK = 400_000
KIMI_K2_RUN = 25_000
# str.isspace(), the test those runs use (not the pattern's `\s`: U+001C..U+001F are space here and not there)
_PY_SPACE = regex.compile(r"[\t\n\x0b\x0c\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")
_SURROGATE = regex.compile(r"[\ud800-\udfff]")


def kimi_k2_slices(text: str) -> list[tuple[int, int]]:
    """The (start, end) pieces of `text` that tokenization_kimi.py hands tiktoken one at a time (see KIMI_K2_CHUNK)."""
    out: list[tuple[int, int]] = []
    for c0 in range(0, len(text), KIMI_K2_CHUNK):
        c1 = min(c0 + KIMI_K2_CHUNK, len(text))
        start = pos = c0
        runs = []
        for m in _PY_SPACE.finditer(text, c0, c1):
            runs += [(pos, m.start()), (m.start(), m.end())]
            pos = m.end()
        runs.append((pos, c1))
        for r0, r1 in runs:
            for cut in range(r0 + KIMI_K2_RUN, r1, KIMI_K2_RUN):
                out.append((start, cut))
                start = cut
        out.append((start, c1))
    return out


def _tiktoken_text(text: str, plain=()) -> tuple[str, list[tuple[int, int]]]:
    """tiktoken's answer to a string with UTF-16 surrogates (UTF-8 cannot carry them): a pair becomes its character, a
    lone half U+FFFD (`encode("utf-16", "surrogatepass").decode("utf-16", "replace")`).  `plain` follows the text."""
    if not _SURROGATE.search(text):
        return text, list(plain)
    fixed = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if not plain:
        return fixed, []
    import bisect
    gone = [m.start() + 1 for m in regex.finditer(r"[\ud800-\udbff][\udc00-\udfff]", text)]   # a pair: 2 -> 1
    return fixed, [(a - bisect.bisect_left(gone, a), b - bisect.bisect_left(gone, b)) for a, b in plain]


def kimi_k2_parts(text: str, plain=()):
    """(text, plain spans in it) for each piece tokenization_kimi.py hands tiktoken alone: kimi_k2_slices' cuts, then
    tiktoken's surrogates (_tiktoken_text).  Whatever encodes a kimi-k2 text takes it through here, as encode() does."""
    if len(text) <= KIMI_K2_RUN:                      # one piece (no run can reach the cut)
        yield _tiktoken_text(text, plain)
        return
    for a, b in kimi_k2_slices(text):
        yield _tiktoken_text(text[a:b], [(max(s, a) - a, min(e, b) - a) for s, e in plain if s < b and e > a])


# THE UNICODE VERSION IS PART OF THE PATTERN.  Its classes are Unicode properties, and each engine reads them from its
# own tables: tiktoken 0.14.0's Rust regex from Unicode 16.0's, the `regex` module (2026.9.29) from later ones.  To
# `regex`, 17,810 code points are assigned that 16.0 leaves unassigned (Cn), 17,480 of them letters, Han, digits or
# marks, and U+0295 is Lo where 16.0 has Ll; a text with one of them splits differently ("\U000323B0's": the `'s` joins
# the letter here and not there).  So kimi-k2 finds its pieces on a copy of the text with those stood in for - a later
# code point by U+FDD0 (a noncharacter, Cn in every version), U+0295 by "a" (an Ll the pattern has no other use for) -
# and cuts them from the text itself: only the splits follow 16.0, never the bytes.  tools/k31/tokenizer_check.py
# compares every class of the pattern with tiktoken's for every code point (--unicode-table prints this table from the
# two), and Tokenizer refuses a `regex` the table does not fit (tables older than 16.0, or newer than these): the
# classes as kimi-k2 reads them must hash to KIMI_K2_CLASS_DIGEST, tiktoken's.
KIMI_K2_UNICODE = "16.0"
KIMI_K2_AFTER = (       # (first, last): assigned after Unicode 16.0, as far as `regex` 2026.9.29 knows
    (0x558, 0x558), (0x58B, 0x58C), (0x5C8, 0x5C9), (0x88F, 0x88F), (0xB53, 0xB54), (0xC5C, 0xC5C), (0xCDC, 0xCDC),
    (0x1ACF, 0x1AF0), (0x208F, 0x208F), (0x209D, 0x209F), (0x20C1, 0x20C4), (0x2B96, 0x2B96), (0x2E60, 0x2E63),
    (0xA7CE, 0xA7CF), (0xA7D2, 0xA7D2), (0xA7D4, 0xA7D4), (0xA7DD, 0xA7DD), (0xA7E2, 0xA7E2), (0xA7F1, 0xA7F1),
    (0xAB6C, 0xAB6D), (0xFBC3, 0xFBD2), (0xFD90, 0xFD91), (0xFDC8, 0xFDCE), (0x107BB, 0x107BF), (0x10940, 0x10959),
    (0x10EC5, 0x10EC7), (0x10EC9, 0x10EEE), (0x10EF0, 0x10EFB), (0x11B0A, 0x11B0A), (0x11B60, 0x11B67),
    (0x11DB0, 0x11DDB), (0x11DE0, 0x11DE9), (0x11DF0, 0x11DF1), (0x1246F, 0x1246F), (0x12475, 0x1247F),
    (0x12550, 0x12686), (0x16EA0, 0x16EB8), (0x16EBB, 0x16ED3), (0x16FF2, 0x16FF6), (0x187F8, 0x187FF),
    (0x18CD6, 0x18CDA), (0x18D09, 0x18D20), (0x18D80, 0x18DF2), (0x18E00, 0x19191), (0x191A0, 0x191D2),
    (0x1B123, 0x1B128), (0x1B168, 0x1B168), (0x1CCFA, 0x1CCFC), (0x1CEBA, 0x1CED0), (0x1CED2, 0x1CED4),
    (0x1CEDD, 0x1CEFD), (0x1D127, 0x1D128), (0x1D1EB, 0x1D1FF), (0x1D250, 0x1D281), (0x1D6A6, 0x1D6A6),
    (0x1DB00, 0x1DB1C), (0x1DF1F, 0x1DF24), (0x1DF2B, 0x1DF81), (0x1DF90, 0x1DF96), (0x1DFCD, 0x1DFFF),
    (0x1E6C0, 0x1E6DE), (0x1E6E0, 0x1E6F5), (0x1E6FE, 0x1E6FF), (0x1F1AE, 0x1F1AE), (0x1F6D8, 0x1F6D9),
    (0x1F777, 0x1F77A), (0x1F7DA, 0x1F7DB), (0x1F7F1, 0x1F7FF), (0x1F8D0, 0x1F8D8), (0x1FA54, 0x1FA57),
    (0x1FA8A, 0x1FA8E), (0x1FAC8, 0x1FAC8), (0x1FACC, 0x1FACD), (0x1FADD, 0x1FADD), (0x1FAEA, 0x1FAEB),
    (0x1FAEF, 0x1FAEF), (0x1FAF9, 0x1FAFA), (0x1FBFA, 0x1FBFA), (0x2B73A, 0x2B73F), (0x2B81E, 0x2B81E),
    (0x2CEA2, 0x2CEAD), (0x323B0, 0x33479), (0x3D000, 0x3FC3F),
)
KIMI_K2_RECLASSED = {0x0295: "a"}          # a general category changed after 16.0: a stand-in of the 16.0 one (Ll)
# every class the pattern reads (`(?i:` for its contractions' case folding)
KIMI_K2_CLASSES = (r"[\p{Han}]", r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
                   r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]", r"\p{L}", r"\p{N}", r"\s", r"(?i:s|t|r|e|v|m|l|d)")
KIMI_K2_CLASS_DIGEST = "a73d722c9c0428d1"
_KIMI_K2_PIN = regex.compile("[%s]" % "".join(["\\U%08X-\\U%08X" % r for r in KIMI_K2_AFTER]
                                              + ["\\U%08X" % c for c in KIMI_K2_RECLASSED]))


def _kimi_k2_seen(text: str) -> str | None:
    """`text` as tiktoken's Unicode 16.0 tables read it, for finding the pieces (KIMI_K2_UNICODE); None: as it is."""
    if not _KIMI_K2_PIN.search(text):
        return None
    return _KIMI_K2_PIN.sub(lambda m: KIMI_K2_RECLASSED.get(ord(m.group()), "\ufdd0"), text)


def kimi_k2_classes(pinned: bool = True) -> list[list[tuple[int, int]]]:
    """Each of KIMI_K2_CLASSES as kimi-k2 reads it (pinned=False: as `regex` does), as (start, end) ranges of indices
    into every code point but the surrogates (index = code point below U+D800, code point - 0x800 above)."""
    every = "".join(map(chr, range(0xD800))) + "".join(map(chr, range(0xE000, 0x110000)))
    seen = (_kimi_k2_seen(every) or every) if pinned else every
    return [[m.span() for m in regex.finditer("(?:%s)+" % c, seen, regex.V1)] for c in KIMI_K2_CLASSES]


def kimi_k2_class_digest(classes: list[list[tuple[int, int]]]) -> str:
    import hashlib
    return hashlib.sha256(repr([[tuple(r) for r in c] for c in classes]).encode()).hexdigest()[:16]


_KIMI_K2_FITS: list[bool] = []


def _kimi_k2_tables_fit() -> None:
    """Refuse a `regex` that reads the pattern's classes otherwise than tiktoken's Unicode 16.0 tables, even with the
    stand-ins (once a process, well under a second)."""
    if _KIMI_K2_FITS:
        return
    got = kimi_k2_class_digest(kimi_k2_classes())
    if got != KIMI_K2_CLASS_DIGEST:
        raise ValueError("kimi-k2: regex %s reads the pattern's classes otherwise than tiktoken's Unicode %s tables "
                         "(digest %s, want %s), so some texts would split otherwise; KIMI_K2_AFTER is for other tables "
                         "(tools/k31/tokenizer_check.py --unicode-table prints them)"
                         % (regex.__version__, KIMI_K2_UNICODE, got, KIMI_K2_CLASS_DIGEST))
    _KIMI_K2_FITS.append(True)


class _IsolatedSplits:
    """findall() over a sequence of "Isolated" splits (see DEEPSEEK3_PATTERNS)."""

    def __init__(self, patterns):
        self.stages = [regex.compile(p) for p in patterns]

    def findall(self, text: str) -> list[str]:
        pieces = [text]
        for pat in self.stages:
            out = []
            for piece in pieces:
                pos = 0
                for m in pat.finditer(piece):
                    if m.start() > pos:
                        out.append(piece[pos:m.start()])
                    if m.end() > m.start():
                        out.append(m.group(0))
                    pos = m.end()
                if pos < len(piece):
                    out.append(piece[pos:])
            pieces = out
        return pieces


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None):
        self.tokens = tokens
        self.pre = pre
        # Each supported scheme names a transcription of a reference: the vendored llama.cpp's (qwen35, glm4/glm5), or
        # the release's own tokenizer (tokenizer.json for deepseek3 and minimax, tokenization_kimi.py for kimi-k2).  An
        # unknown scheme is REFUSED rather than silently tokenized with the wrong splits.
        if pre not in ("qwen35", "glm4", "glm5", "deepseek3", "minimax", "kimi-k2"):
            raise ValueError("the %r pre-tokenizer is not transcribed; known: qwen35, glm4, glm5, deepseek3, minimax, "
                             "kimi-k2" % pre)
        # glm4/glm5 set ignore_merges: an exact vocab hit emits the whole pre-token piece
        self.ignore_merges = pre in ("glm4", "glm5")
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # Merge rules as (left, right) -> rank.  A merge list that is not a total order over its pairs, or one
        # naming a token that is not in the vocabulary, would make BPE silently produce different ids than the
        # model was trained with, so both are checked here rather than discovered as a bad answer later.
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        self._re = (_IsolatedSplits(DEEPSEEK3_PATTERNS) if pre == "deepseek3" else
                    regex.compile(MINIMAX_PATTERN) if pre == "minimax" else
                    regex.compile(KIMI_K2_PATTERN, regex.V1) if pre == "kimi-k2" else
                    regex.compile(GLM_PATTERN if self.ignore_merges else QWEN35_PATTERN))
        self._nfc = pre == "minimax"
        if pre == "kimi-k2":
            _kimi_k2_tables_fit()

        # The literals matched directly instead of being run through BPE.  GGUF token types: 3 = CONTROL,
        # 4 = USER_DEFINED.  The two classes behave DIFFERENTLY and llama.cpp's own tokenizer settled which:
        #
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, `<tool_response>`, ...) is matched ALWAYS,
        #     with or without parse_special.
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ...) is matched ONLY when
        #     parse_special is set.
        #
        # Measured, not assumed: with parse_special=False the oracle still emitted `<tool_response>` as one
        # token and mine emitted four, and the three disagreements were exactly the type-4 cases while every
        # type-3 case agreed.  Treating both classes alike costs 3 strings in 1229 and silently changes a chat
        # prompt, because `<|im_end|>` decomposed into ordinary pieces is not the token the model expects.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        # Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.  `regex.escape` so a
        # token containing regex metacharacters (several do: `<|`, `[`, `(`) is matched literally.
        self._always_re = self._alt(always)
        self._special_re = self._alt(list(self.special_tokens))

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_dir(cls, path) -> "Tokenizer":
        """A pack's tokenizer/ directory (extract() below), with the pre-tokenizer its tokenizer.json names: built
        without it, a GLM pack was split the qwen35 way - every number one digit a token where GLM groups up to three
        (#27: numbers came back garbled in long prompts).  No tokenizer.json (never written by extract): qwen35."""
        t = pathlib.Path(path)
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for s, i in vocab.items():
            tokens[i] = s
        meta = json.loads((t / "tokenizer.json").read_text(encoding="utf-8")) if (t / "tokenizer.json").exists() else {}
        types = json.loads((t / "token_type.json").read_text()) if (t / "token_type.json").exists() else None
        return cls(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"), types,
                   meta.get("pre", "qwen35"), meta.get("special_ids") or None)

    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by LOWEST RANK first, repeatedly - not left to right.

        Applying merges in list order rather than rank order is the classic BPE bug: it produces a different
        segmentation and a plausible token count.
        """
        if len(word) > 64 and self.pre == "kimi-k2":
            return self._bpe_heap(word)
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def _bpe_heap(self, word: str) -> list[str]:
        """_bpe's merges in _bpe's order (lowest rank, then leftmost), from a heap: a long piece is quadratic the other
        way, and a model writes them (a 5,530-character rule of box-drawing lines, 16,590 bytes, took ~20 s)."""
        import heapq
        sym = list(word)
        n = len(sym)
        nxt = list(range(1, n + 1))
        prv = list(range(-1, n - 1))
        ver = [0] * n
        heap = [(r, i, 0, i + 1, 0) for i in range(n - 1)
                for r in (self.ranks.get((sym[i], sym[i + 1])),) if r is not None]
        heapq.heapify(heap)

        def push(i):
            j = nxt[i]
            if j < n:
                r = self.ranks.get((sym[i], sym[j]))
                if r is not None:
                    heapq.heappush(heap, (r, i, ver[i], j, ver[j]))

        while heap:
            r, i, vi, j, vj = heapq.heappop(heap)
            if ver[i] != vi or ver[j] != vj or nxt[i] != j:       # a side merged since: stale
                continue
            sym[i] += sym[j]
            ver[i] += 1
            ver[j] = -1                                          # j is gone
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prv[nxt[j]] = i
            if prv[i] >= 0:
                push(prv[i])
            push(i)
        out, i = [], 0
        while i < n:
            out.append(sym[i])
            i = nxt[i]
        return out

    def _pieces(self, text: str) -> list[str]:
        """The pre-tokenizer's pieces of `text`; kimi-k2's are found as tiktoken's tables read it (KIMI_K2_UNICODE)."""
        seen = _kimi_k2_seen(text) if self.pre == "kimi-k2" else None
        if seen is None:
            return self._re.findall(text)
        return [text[m.start():m.end()] for m in self._re.finditer(seen)]

    def _encode_plain(self, text: str) -> list[int]:
        # a piece always encodes to the same ids, so they are remembered (a prompt repeats most of its words, and an
        # agent re-sends the whole conversation every turn); the memo is dropped whole when it grows past ~300K pieces
        memo = self.__dict__.setdefault("_piece_memo", {})
        if len(memo) > 300_000:
            memo.clear()
        out: list[int] = []
        if self.__dict__.get("_nfc"):
            import unicodedata
            text = unicodedata.normalize("NFC", text)
        for piece in self._pieces(text):
            hit = memo.get(piece)
            if hit is not None:
                out.extend(hit)
                continue
            # The vocab is stored in the BYTE-LEVEL alphabet, so the ignore_merges whole-vocab
            # lookup (llama-vocab.cpp L633-637: an exact hit is emitted whole, only leftovers
            # reach BPE) must run on the byte-MAPPED piece - for ASCII the mapping is the
            # identity and the raw lookup agrees, for everything else it does not (x=Δy²: the
            # mapped piece 'Â²' is one token, the raw piece '²' is not in the vocab at all).
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            if self.ignore_merges and mapped in self.ids:
                ids = (self.ids[mapped],)
            else:
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                ids = tuple(got)
            memo[piece] = ids
            out.extend(ids)
        return out

    def _encode_segment(self, text: str) -> list[int]:
        """_encode_plain, remembering long segments (the text between two special tokens: a message, a tool result):
        BPE never merges across a special token, so a re-sent conversation's earlier messages encode to the same ids
        and come from here (LRU, up to ~1M tokens)."""
        if len(text) < 2048:
            return self._encode_plain(text)
        cache = self.__dict__.get("_seg_cache")
        if cache is None:
            import collections
            import threading
            cache = self.__dict__["_seg_cache"] = collections.OrderedDict()
            self.__dict__["_seg_lock"] = threading.Lock()
            self.__dict__["_seg_tokens"] = 0
        lock = self.__dict__["_seg_lock"]
        with lock:
            hit = cache.get(text)
            if hit is not None:
                cache.move_to_end(text)
                return list(hit)
        ids = self._encode_plain(text)
        with lock:
            if text not in cache:
                cache[text] = tuple(ids)
                self.__dict__["_seg_tokens"] += len(ids)
                while self.__dict__["_seg_tokens"] > 1_000_000 and len(cache) > 1:
                    _, old = cache.popitem(last=False)
                    self.__dict__["_seg_tokens"] -= len(old)
        return ids

    def _encode_matching(self, text: str, pat, plain=()) -> list[int]:
        """Encode `text`, emitting any literal `pat` matches as single tokens and BPE-ing the rest.

        KEEP IN STEP: serve/server.py's PromptMemo._encode (MAYA_PROMPT_MEMO) runs a copy of this loop over a prompt's
        tail, using `_special_re`, `special_tokens` and `_encode_segment`; a change here must be made there too
        (serve/test_prompt_memo.py checks that both give the same ids).  NOT IN STEP FOR kimi-k2 YET: encode() runs this
        loop on each of kimi_k2_parts (tokenization_kimi.py's cuts, tiktoken's surrogates) and PromptMemo on the
        uncut text, so a prompt with a 25,000-character run or a UTF-16 surrogate gets other ids there.  Before K3.1
        is served, PromptMemo must take its text through kimi_k2_parts too, or stay off for kimi-k2.

        The split happens on the RAW text, before the byte mapping, because a special token's string is a
        literal to match rather than bytes to decompose.  Everything between the matches is tokenized
        normally - which is why a near-miss like `<|im_star` still costs ordinary tokens.  A match that starts
        inside one of the `plain` (start, end) spans is not a token: it is left to the text around it.
        """
        if pat is None:
            return self._encode_segment(text)
        out: list[int] = []
        pos = 0
        spans = iter(sorted(plain))
        upcoming = next(spans, None)
        plain_end = 0
        for m in pat.finditer(text):
            start = m.start()
            while upcoming is not None and upcoming[0] <= start:   # matches come in text order: each span once
                plain_end = max(plain_end, upcoming[1])
                upcoming = next(spans, None)
            if start < plain_end:
                continue
            if start > pos:
                out.extend(self._encode_segment(text[pos:start]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_segment(text[pos:]))
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`.

        `parse_special` controls only the type-3 CONTROL literals such as `<|im_end|>`; the type-4
        USER_DEFINED ones such as `<think>` are matched either way.  See the note in `__init__`.
        `plain`: (start, end) spans of `text` that are ordinary text even where they spell a literal - a `</think>`
        or `<|user|>` quoted inside a message - and are tokenized with the text around them.
        `kimi-k2` cuts the text first, as tokenization_kimi.py does (kimi_k2_parts), and encodes each piece alone.
        """
        pat = self._special_re if parse_special else self._always_re
        if self.pre != "kimi-k2":
            return self._encode_matching(text, pat, plain)
        out: list[int] = []
        for piece, spans in kimi_k2_parts(text, plain):
            out.extend(self._encode_matching(piece, pat, spans))
        return out

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens)."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            if self.token_types and self.token_types[i] in (3, 4):
                # a control / user-defined token is its text as is (llama.cpp's token_to_piece): GLM's are ASCII,
                # where the byte alphabet is the identity, DeepSeek's (<｜User｜>, ｜DSML｜) are not
                b = cache[i] = self.tokens[i].encode("utf-8")
                return b
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_gguf(gguf_path)
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": False,
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        "ignore_merges": tk.ignore_merges,
        "pre_pattern": GLM_PATTERN if tk.ignore_merges else QWEN35_PATTERN,
        "pre_pattern_source": (".ref/llama.cpp src/llama-vocab.cpp L406 (LLAMA_VOCAB_PRE_TYPE_CHATGLM4; "
                               "glm4/glm5 set ignore_merges)" if tk.ignore_merges else
                               ".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)"),
    }
    if tk.pre == "kimi-k2":
        cfg.update(pre_pattern=KIMI_K2_PATTERN,
                   pre_pattern_source="tokenization_kimi.py TikTokenTokenizer.pat_str (Rust regex syntax: `&&` in a "
                                      "class is an intersection)",
                   slices={"chunk": KIMI_K2_CHUNK, "run": KIMI_K2_RUN, "space": "str.isspace"},
                   unicode={"version": KIMI_K2_UNICODE, "unassigned": [list(r) for r in KIMI_K2_AFTER],
                            "stand_ins": {"%04X" % c: t for c, t in KIMI_K2_RECLASSED.items()}})
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    tpl = GGUFFile(pathlib.Path(gguf_path)).metadata.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
