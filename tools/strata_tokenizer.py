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
        # Each supported scheme names a transcription of the vendored llama.cpp reference; an
        # unknown scheme is REFUSED rather than silently tokenized with the wrong splits.
        if pre not in ("qwen35", "glm4", "glm5", "deepseek3"):
            raise ValueError("the %r pre-tokenizer is not transcribed; known: qwen35, glm4, glm5, deepseek3" % pre)
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
                    regex.compile(GLM_PATTERN if self.ignore_merges else QWEN35_PATTERN))

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

    def _encode_plain(self, text: str) -> list[int]:
        # a piece always encodes to the same ids, so they are remembered (a prompt repeats most of its words, and an
        # agent re-sends the whole conversation every turn); the memo is dropped whole when it grows past ~300K pieces
        memo = self.__dict__.setdefault("_piece_memo", {})
        if len(memo) > 300_000:
            memo.clear()
        out: list[int] = []
        for piece in self._re.findall(text):
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
        """
        return self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)

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
