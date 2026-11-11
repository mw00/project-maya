#!/usr/bin/env python3
"""tools/k31/tokenizer_check.py - the server's tokenizer (pre "kimi-k2") against Moonshot's own, id for id.

    python -I tools/k31/tokenizer_check.py --model DIR --gguf K31.gguf \
        [--jsonl FILE ...] [--tokens NPZ ...] [--code DIR ...] [--sweep] [--min-ids 1000000]
    python -I tools/k31/tokenizer_check.py --model DIR --unicode-table

DIR is the release's tokenizer: tiktoken.model, tokenizer_config.json, tokenization_kimi.py and encoding_k3.py.  Those
files are imported (they are not part of this repository), so run this with `python -I`.  Theirs is
`TikTokenTokenizer.from_pretrained(DIR)`; ours is tools/strata_tokenizer.py's Tokenizer on the GGUF's vocabulary and
merges, the ones a pack ships.  Without --gguf, ours is built from tiktoken.model the way the converter builds it (one
merge per token: tiktoken's BPE replayed up to the token's rank); with --gguf, the GGUF must equal that rebuild.

What is compared:
  vocab     every id's bytes (tiktoken's decode_single_token_bytes against ours), and the special tokens: none may be a
            prefix of another, since tiktoken tries them in hash order and we try the longest first;
  unicode   every class the pattern reads (strata's KIMI_K2_CLASSES), code point by code point over all of them but the
            surrogates, as tiktoken's own regex reads it and as ours does with its Unicode 16.0 stand-ins, and the
            digest strata checks its `regex` against (KIMI_K2_CLASS_DIGEST) against tiktoken's;
  texts     every text encoded both ways: specials parsed (their encode(t), ours parse_special=True) and as plain text
            (their allow_special_tokens=False, ours parse_special=False); then both decodes of the ids, and the
            round trip to the text itself;
  chat      every --jsonl conversation through their apply_chat_template(tokenize=True) against ours, segment by
            segment from encoding_k3.build_chat_segments (the same segments, so only the tokenizer differs), with
            thinking on/off, the four efforts and the tools the record declares.
Texts, by kind: tools/tokenizer_corpus.txt and its lines; the Kimi cases below; the messages, reasoning, tool
arguments and raw text of each --jsonl record; each --tokens sequence (calibration ids) decoded by them; source files
under each --code directory; and generated text, seeded: CJK, other scripts, emoji, tool JSON, the 256 special
tokens and near misses, whitespace and controls, numbers, UTF-16 surrogates, and the long cases (a text past
tokenization_kimi.py's 400,000-character chunk, runs past its 25,000-character cut, also where str.isspace() and the
pattern's whitespace disagree, a special token across that cut).  --sweep adds every code point that Python's
unicodedata or the `regex` module calls assigned (their tables are of two Unicode versions, tiktoken's of a third), in
nine contexts each.

Exit 0 only when nothing differs and at least --min-ids ids were compared: their side, specials parsed, each text once
(the chats count, though they carry the messages again; the sweep, the chats re-encoded whole and the long mix of the
generated texts are compared and not counted).  --unicode-table prints strata's KIMI_K2_AFTER and the code points whose
class changed, from tiktoken's tables and this `regex`'s, for when either is upgraded.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import pathlib
import random
import sys
import time
import unicodedata

import regex

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from strata_tokenizer import (BYTE_TO_UNICODE, KIMI_K2_CHUNK, KIMI_K2_CLASS_DIGEST, KIMI_K2_CLASSES,  # noqa: E402
                              KIMI_K2_RUN, Tokenizer, kimi_k2_class_digest, kimi_k2_classes)

CASES = [
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01Hello\u4e16\u754cworld "
    "\u6f22\u5b57\u304b\u306a\u30ab\u30ca\u3005\u3005 \u303b\u3007",
    "\U00016ff0\u4e00 \u4e00\u0301 a\u4e00b \u4e00'\u5b57's \u3400\u4dbf \U00020000\U0002a6df \uf900\ufa6d",
    "a&&b A&&B &&& [&&] \u00c0&&\u00e0",                                 # the `&&` is an intersection, not text
    "don't I'LL we'VE it'\u017f O'Neil rock'n'roll ''s x's's 'S 'D 'M HELLO'S \u00c9T\u00c9'LL",
    "x=\u0394y\u00b2 12345 1234567 \u0661\u0662\u0663\u0664 \uff10\uff11\uff12\uff13 "
    "\u2167\u216b \u2460\u2461 \u00bd \u0967\u0968",
    "\x1c\x1d\x1e\x1f spaces\u00a0\u3000x  \n\n\r\n  y\x0b\x0c\x85\u2028\u2029\u200b\ufeff\u180e\u2060end",
    "\U0001F469\u200D\U0001F4BB \U0001F1EF\U0001F1F5 \U0001F44D\U0001F3FD \u2764\ufe0f 1\ufe0f\u20e3 "
    "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F \U0001F468\u200D\U0001F469\u200D\U0001F467",
    "<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>",
    "<|end_of_msg|<|end_of_msg|>>[EOS][EOS[BOS]] <|reserved_token_163592|><|reserved_token_9|>[PAD][UNK]",
    "e\u0301\u0301 combining \u0915\u093f \u0e01\u0e34 \u05e9\u05c1\u05b8 \u0628\u0650\u0633\u0652 \u1100\u1161\u11a8",
    "\u2500" * 300 + "\n" + "=" * 90 + "\n" + "|---|:--:|---:|\n" + " " * 70 + "x",
    "def f(x):\n\treturn {'a': [1, 2, 3]}  # comment\r\n    if x>=10//3: pass\n\n\n",
]
SCRIPTS = {   # (first, last) code point ranges sampled for generated text
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAD9)],
    "kana": [(0x3041, 0x3096), (0x30A1, 0x30FA), (0xFF66, 0xFF9D), (0x31F0, 0x31FF)],
    "hangul": [(0xAC00, 0xD7A3), (0x1100, 0x1112), (0x1161, 0x1175)],
    "cjk_punct": [(0x3000, 0x303F), (0xFF01, 0xFF5E), (0x3200, 0x32FF)],
    "latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)],
    "greek_cyr": [(0x391, 0x3C9), (0x400, 0x4FF)],
    "rtl": [(0x5D0, 0x5EA), (0x591, 0x5C7), (0x621, 0x64A), (0x64B, 0x652), (0x660, 0x669)],
    "indic_thai": [(0x900, 0x97F), (0x980, 0x9FF), (0xB80, 0xBFF), (0xE01, 0xE5B)],
    "other": [(0x10A0, 0x10FF), (0x531, 0x587), (0x1200, 0x137C), (0x13A0, 0x13F5)],
    "marks": [(0x300, 0x36F), (0x1AB0, 0x1ACE), (0x20D0, 0x20F0), (0xFE20, 0xFE2F)],
    "symbols": [(0x2010, 0x2064), (0x20A0, 0x20C0), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2500, 0x257F),
                (0x2460, 0x24FF), (0x2150, 0x218B), (0xA0, 0xBF)],
    "emoji_cp": [(0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF), (0x2600, 0x27BF)],
}
EMOJI = ["\U0001F469\u200D\U0001F4BB", "\U0001F1EF\U0001F1F5", "\U0001F1FA\U0001F1F8", "\U0001F44D\U0001F3FD",
         "\u2764\ufe0f", "1\ufe0f\u20e3", "#\ufe0f\u20e3", "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466",
         "\U0001F3F3\ufe0f\u200D\U0001F308", "\U0001F9D1\U0001F3FF\u200D\U0001F9B0", "\u263a", "\u263a\ufe0f",
         "\U0001F600", "\U0001F680", "\U0001F525\U0001F525\U0001F525", "\U0001F926\U0001F3FB\u200D\u2642\ufe0f",
         "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F", "\u00a9\ufe0f", "\U0001F004"]
WORDS = ("the of and to in is that for it with as was on be at by this had not are but from or have an they which "
         "one you were all her she there would their we him been has when who will more no if out so said what up its "
         "about into than them can only other new some could time these two may then do first any my now such like "
         "a\u00f1o ni\u00f1o canci\u00f3n m\u00e1s est\u00e1 \u00e9l qu\u00e9 tambi\u00e9n Stra\u00dfe na\u00efve "
         "caf\u00e9 r\u00e9sum\u00e9 \u00fcber Gr\u00f6\u00dfe co\u00f6perate Z\u00fcrich S\u00e3o "
         "Lodz \u0141\u00f3d\u017a "
         "camelCase snake_case SCREAMING_CASE HTTPServer iPhone McDonald's kebab-case __init__ self.x foo.bar() "
         "IDs URLs JSON UTF-8 x86_64 v1.1.55 3.14159 1e-20 0x83DCD12D 2026-11-05T19:20:00Z").split()
WS = [" ", " ", " ", "  ", "\t", "\n", "\r\n", "\r", "\n\n", "\n\n\n", "\x0b", "\x0c", "\x85", "\xa0", "\u2028",
      "\u2029", "\u3000", "\u200b", "\ufeff", "\x1c", "\x1f", " \n ", "\t\t\n", "    ", "\n    ", " \r\n\r\n"]
PUNCT = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + ["...", "--", "->", "=>", "::", "//", "/*", "*/", "\u2014",
                                                     "\u2026", "\u201c", "\u201d", "\u2018", "\u2019", "\u00bf",
                                                     "\u00a1", "\u3001", "\u3002", "\uff0c", "\u300c", "\u300d"]


# ------------------------------------------------------------------ the two tokenizers
def load_theirs(model: pathlib.Path):
    sys.path.append(str(model))       # appended: the release's files never shadow an installed module
    import transformers
    transformers.logging.set_verbosity_error()
    mod = importlib.import_module("tokenization_kimi")
    return mod.TikTokenTokenizer.from_pretrained(str(model)), importlib.import_module("encoding_k3")


def rebuild(hf) -> tuple[list[str], list[str], list[int]]:
    """The GGUF tokenizer the converter writes (conversion/kimi_linear.py set_vocab, `kimi-k2`), from tiktoken.model."""
    ranks = hf.model._mergeable_ranks

    def bpe(token: bytes, max_rank: int) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best_rank is None or best_rank >= max_rank:
                return parts
            parts[best:best + 2] = [parts[best] + parts[best + 1]]

    def s(b: bytes) -> str:
        return "".join(BYTE_TO_UNICODE[x] for x in b)

    n = hf.vocab_size
    tokens, types, merges = ["[PAD%d]" % i for i in range(n)], [5] * n, []    # 5 = UNUSED (gguf.TokenType)
    for token, rank in ranks.items():
        tokens[rank], types[rank] = s(token), 1
        if len(token) > 1:
            m = bpe(token, rank)
            if len(m) == 2:
                merges.append(s(m[0]) + " " + s(m[1]))
    for name, i in hf.special_tokens.items():
        tokens[i], types[i] = name, 3
    return tokens, merges, types


def load_ours(hf, gguf) -> tuple[Tokenizer, list[str]]:
    notes = []
    tokens, merges, types = rebuild(hf)
    if gguf is None:
        return Tokenizer(tokens, merges, types, "kimi-k2"), ["ours: rebuilt from tiktoken.model (no --gguf)"]
    from gguf_reader import GGUFFile
    md = GGUFFile(pathlib.Path(gguf)).metadata
    same = {"tokens": list(md["tokenizer.ggml.tokens"]) == tokens,
            "merges": list(md["tokenizer.ggml.merges"]) == merges,
            "types": list(md["tokenizer.ggml.token_type"]) == types, "pre": md.get("tokenizer.ggml.pre") == "kimi-k2"}
    notes.append("ours: %s; GGUF = rebuild from tiktoken.model: %s (%d tokens, %d merges)"
                 % (pathlib.Path(gguf).name, ", ".join("%s %s" % (k, "yes" if v else "NO") for k, v in same.items()),
                    len(tokens), len(merges)))
    return Tokenizer.from_gguf(gguf), notes + ([] if all(same.values()) else ["FAIL: the GGUF's tokenizer differs"])


def every_index_cp() -> list[int]:
    """The code points kimi_k2_classes indexes: all but the surrogates."""
    return list(range(0xD800)) + list(range(0xE000, 0x110000))


def spans(members) -> list[tuple[int, int]]:
    """Sorted indices -> (start, end) ranges, as kimi_k2_classes gives them."""
    out: list[list[int]] = []
    for i in members:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(r) for r in out]


def theirs_classes(hf, classes=KIMI_K2_CLASSES) -> list[list[tuple[int, int]]]:
    """Each class as tiktoken's own regex reads it, in kimi_k2_classes' form: the class alone is the pattern of an
    Encoding on the same ranks, over every code point but the surrogates, each followed by U+0000.  tiktoken encodes
    only what its pattern matches, so a code point gives ids exactly when it is in the class (U+0000 on its own)."""
    import tiktoken
    ranks = hf.model._mergeable_ranks
    cps = every_index_cp()
    text = "".join(chr(c) + "\x00" for c in cps[1:])
    sep = ranks[b"\x00"]
    out = []
    for cls in classes:
        enc = tiktoken.Encoding("k2-class", pat_str=r"(?:%s)|\x00" % cls, mergeable_ranks=ranks, special_tokens={})
        alone = tiktoken.Encoding("k2-class0", pat_str=cls, mergeable_ranks=ranks, special_tokens={})
        members, k, got = [0] if alone.encode_ordinary("\x00") else [], 1, 0
        for x in enc.encode_ordinary(text):
            if x != sep:
                got += 1
                continue
            if got:
                members.append(k)
            k, got = k + 1, 0
        if k != len(cps):
            raise RuntimeError("the class probe lost its place: %d of %d code points" % (k, len(cps)))
        out.append(spans(members))
    return out


def check_unicode(hf) -> int:
    """The pattern's classes, ours (with strata's stand-ins) against tiktoken's: the code points that differ."""
    import tiktoken
    cps = every_index_cp()
    theirs, ours = theirs_classes(hf), kimi_k2_classes()
    bad = 0
    for cls, t, o in zip(KIMI_K2_CLASSES, theirs, ours):
        ts, os_ = ({i for a, b in r for i in range(a, b)} for r in (t, o))
        if ts != os_:
            bad += len(ts ^ os_)
            print("UNICODE %s: %d code points differ, e.g. %s" % (cls, len(ts ^ os_), ", ".join(
                "U+%04X (%s)" % (cps[i], "theirs" if i in ts else "ours") for i in sorted(ts ^ os_)[:6])))
    digest = kimi_k2_class_digest(theirs)
    print("unicode: %d classes over %d code points, ours (regex %s, stand-ins) = tiktoken %s's: %s; digest %s, "
          "strata's %s" % (len(KIMI_K2_CLASSES), len(cps), regex.__version__, tiktoken.__version__,
                           "yes" if not bad else "NO", digest, KIMI_K2_CLASS_DIGEST))
    return bad + (digest != KIMI_K2_CLASS_DIGEST)


def unicode_table(hf) -> None:
    """strata's KIMI_K2_AFTER from the two tables (assigned to `regex`, not to tiktoken), and the code points both
    assign whose class differs (each needs a stand-in in KIMI_K2_RECLASSED)."""
    cps = every_index_cp()
    t_cls, r_cls = theirs_classes(hf, KIMI_K2_CLASSES + (r"\P{Cn}",)), kimi_k2_classes(pinned=False)
    every = "".join(map(chr, cps))
    r_cls.append([m.span() for m in regex.finditer(r"\P{Cn}+", every)])
    sets = [[{i for a, b in r for i in range(a, b)} for r in c] for c in (t_cls, r_cls)]
    after = sorted(sets[1][-1] - sets[0][-1])
    print("KIMI_K2_AFTER (%d code points): %s" % (len(after), ", ".join(
        "(0x%X, 0x%X)" % (cps[a], cps[b - 1]) for a, b in spans(after))))
    both = sets[0][-1] & sets[1][-1]
    moved = sorted({i for t, r in zip(sets[0][:-1], sets[1][:-1]) for i in (t ^ r) & both})
    for i in moved:
        print("class changed: U+%04X %s; tiktoken's classes %s, regex's %s" % (
            cps[i], unicodedata.name(chr(cps[i]), "?"), [c for c, t in zip(KIMI_K2_CLASSES, sets[0]) if i in t],
            [c for c, r in zip(KIMI_K2_CLASSES, sets[1]) if i in r]))
    print("tiktoken's digest (KIMI_K2_CLASS_DIGEST): %s" % kimi_k2_class_digest(t_cls[:-1]))


# ------------------------------------------------------------------ texts
def file_texts(paths) -> list[str]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:     # CRLF kept: the pattern treats it apart
            out.append(f.read())
    return out


def jsonl_records(paths) -> list[dict]:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return recs


def record_texts(recs) -> list[str]:
    out = []
    for r in recs:
        if isinstance(r.get("raw"), str):
            out.append(r["raw"])
        if r.get("tools"):
            out.append(json.dumps(r["tools"], ensure_ascii=False))
        for m in r.get("messages") or []:
            for k in ("content", "reasoning_content"):
                if isinstance(m.get(k), str) and m[k]:
                    out.append(m[k])
            for c in m.get("tool_calls") or []:
                out.append(json.dumps(c, ensure_ascii=False))
    return out


def token_texts(hf, paths) -> list[str]:
    import numpy as np
    out = []
    for p in paths:
        z = np.load(p)
        flat, offs = z["flat"], z["seq_offsets"]
        out += [hf.decode([int(x) for x in flat[offs[i]:offs[i + 1]]]) for i in range(len(offs) - 1)]
    return out


def code_texts(dirs, cap: int) -> list[str]:
    exts = {".py", ".cu", ".cuh", ".cpp", ".hpp", ".h", ".c", ".md", ".json", ".sh", ".txt", ".yml", ".yaml",
            ".jinja", ".cmake", ".bat", ".ps1", ".metal", ".toml", ".js", ".ts", ".html", ".css"}
    out, total = [], 0
    for d in dirs:
        for p in sorted(pathlib.Path(d).rglob("*")):
            if total >= cap:
                return out
            if (p.is_file() and (p.suffix in exts or p.name == "CMakeLists.txt") and p.stat().st_size < 300_000
                    and ".git" not in p.parts):
                try:
                    t = p.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                out.append(t)
                total += len(t)
    return out


def generated(specials: list[str], seed: int = 31) -> dict[str, list[str]]:
    rng = random.Random(seed)

    def ch(kind):
        lo, hi = rng.choice(SCRIPTS[kind])
        while True:
            c = chr(rng.randint(lo, hi))
            if unicodedata.category(c) not in ("Cn", "Cs"):
                return c

    def run(kind, a=1, b=8):
        return "".join(ch(kind) for _ in range(rng.randint(a, b)))

    def word():
        w = rng.choice(WORDS)
        return rng.choice([w, w, w.upper(), w.capitalize(), w + "'s", w + "'LL", w + "'ve"])

    def mix(kinds, n):
        parts = []
        while sum(map(len, parts)) < n:
            k = rng.random()
            parts.append(run(rng.choice(kinds)) if k < 0.45 else word() if k < 0.7 else
                         rng.choice(WS) if k < 0.85 else rng.choice(PUNCT) if k < 0.95 else rng.choice(EMOJI))
        return "".join(parts)

    def jvalue(depth=0):
        k = rng.random()
        if depth < 4 and k < 0.3:
            return {rng.choice([word(), run("latin", 3, 10), run("han", 1, 3), "_" + word()]): jvalue(depth + 1)
                    for _ in range(rng.randint(1, 6))}
        if depth < 4 and k < 0.45:
            return [jvalue(depth + 1) for _ in range(rng.randint(0, 5))]
        return rng.choice([rng.randint(-10 ** 9, 10 ** 12), rng.random() * 10 ** rng.randint(-8, 8), True, False,
                           None, mix(["latin", "han", "symbols"], rng.randint(1, 60)), word(), "", "\\n\"quoted\"\t"])

    def jdump(v):
        return rng.choice([lambda: json.dumps(v, ensure_ascii=False), lambda: json.dumps(v),
                           lambda: json.dumps(v, ensure_ascii=False, indent=2),
                           lambda: json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)])()

    def tool():
        props = {word().replace("'", ""): {"type": rng.choice(["string", "integer", "number", "boolean", "array"]),
                                          "description": mix(["latin"], rng.randint(10, 80))}
                 for _ in range(rng.randint(1, 5))}
        return {"type": "function", "function": {"name": "_".join(rng.choice(WORDS) for _ in range(2)),
                                                 "description": mix(["latin", "han"], 60),
                                                 "parameters": {"type": "object", "properties": props,
                                                                "required": list(props)[:1]}}}

    out: dict[str, list[str]] = {}
    out["cjk"] = [mix(["han", "han", "kana", "hangul", "cjk_punct", "latin"], rng.randint(20, 1500))
                  for _ in range(500)]
    out["scripts"] = [mix(["latin", "greek_cyr", "rtl", "indic_thai", "other", "marks", "symbols"],
                          rng.randint(20, 1500)) for _ in range(500)]
    out["emoji"] = ["".join(rng.choice(EMOJI + [" ", "\n", word(), ch("emoji_cp"), ch("emoji_cp") + "\ufe0f",
                                                ch("emoji_cp") + "\u200d" + ch("emoji_cp")])
                            for _ in range(rng.randint(5, 200))) for _ in range(400)]
    out["tool_json"] = ([jdump(jvalue()) for _ in range(700)] + [jdump([tool() for _ in range(rng.randint(1, 6))])
                                                                 for _ in range(200)])
    near = ["<|", "|>", "<|end_of_msg|", "end_of_msg|>", "[EOS", "EOS]", "<|open|", "<<|sep|>>", "[[BOS]]",
            "<|reserved_token_|>", "<|Open|>", "< |sep| >", "<|sep|>"[::-1]]
    out["specials"] = ["".join(rng.choice([rng.choice(specials), rng.choice(near), word(), rng.choice(WS),
                                           rng.choice(PUNCT), run("han", 1, 3), rng.choice(specials) + word()])
                               for _ in range(rng.randint(2, 120))) for _ in range(600)]
    out["whitespace"] = ["".join(rng.choice([rng.choice(WS) * rng.randint(1, 9), word(), rng.choice(PUNCT),
                                             chr(rng.randint(0, 0x1f)), "\x7f"]) for _ in range(rng.randint(2, 200)))
                         for _ in range(400)]
    digits = "0123456789\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669\u0966\u0967\u0968\uff10" \
             "\uff11\uff12\u00b2\u00b3\u00b9\u00bd\u2160\u2161\u216b\u2460\u2461\u09e6\u09e7\u0e50\u0e51"
    out["numbers"] = [" ".join("".join(rng.choice(digits if rng.random() < 0.3 else "0123456789")
                                       for _ in range(rng.randint(1, 14))) + rng.choice(["", ".", ",", "e-", "%", "x"])
                               for _ in range(rng.randint(1, 80))) for _ in range(300)]
    sur = ["\ud800", "\udfff", "\udbff", "\udc00", "\ud83d\ude00", "\ude00\ud83d", "\ud83d"]
    out["surrogates"] = ["".join(rng.choice([rng.choice(sur), word(), " ", rng.choice(specials)])
                                 for _ in range(rng.randint(1, 30))) for _ in range(60)]
    # the long cases: tokenization_kimi.py's cuts (KIMI_K2_CHUNK characters, KIMI_K2_RUN of one class)
    nospace = json.dumps([jvalue() for _ in range(900)], ensure_ascii=False, separators=(",", ":"))
    nospace = "".join(c for c in nospace if not c.isspace())
    b64 = base64.b64encode(rng.randbytes(70_000)).decode()
    out["long"] = [
        nospace[:2 * KIMI_K2_RUN + 17], b64, " \x1c" * (KIMI_K2_RUN + 7), "\n" + run("han", 30_000, 30_000),
        # U+001C..U+001F are str.isspace() and not `\s`: a cut only str.isspace() runs make, and one a place apart
        "a" * (KIMI_K2_RUN - 10) + "\x1c" + "a" * 20, "\x1c" + " " * (KIMI_K2_RUN + 50),
        "x" * (KIMI_K2_RUN - 7) + "<|end_of_msg|>" + "y" * 50,           # a special token across the cut
        "z" * (KIMI_K2_RUN - 2) + "<|open|>" + nospace[:KIMI_K2_RUN],
        " " * 3 + b64[:KIMI_K2_RUN * 2] + "\n" * 4 + b64[:KIMI_K2_RUN + 1],
    ]
    out["long_mix"] = ["".join(rng.choice(out["cjk"] + out["scripts"] + out["tool_json"] + out["specials"])
                               for _ in range(1400))[:int(KIMI_K2_CHUNK * 2.3)]]     # past two chunk cuts
    return out


def sweep_texts() -> list[str]:
    """Every code point unicodedata or `regex` calls assigned, private use only in the BMP, in nine contexts, 256 a
    text.  unicodedata's alone (Unicode 15.0 here) never reaches what was assigned since, where the stand-ins work."""
    assigned = regex.compile(r"\P{Cn}")
    cps = [c for c in range(0xF0000) if not 0xD800 <= c <= 0xDFFF
           and (unicodedata.category(chr(c)) != "Cn" or assigned.match(chr(c)))]
    out = []
    for i in range(0, len(cps), 256):
        lines = []
        for cp in cps[i:i + 256]:
            c = chr(cp)
            lines += [c, "a" + c + "b", " " + c + c, "'" + c, c + "'s", "1" + c + "2", "\u4e2d" + c, "A" + c + "a",
                      "<|sep|>" + c]
        out.append("\n".join(lines))
    return out


# ------------------------------------------------------------------ the comparison
class Tally:
    def __init__(self, tk):
        self.tk, self.rows, self.shown, self.bad = tk, {}, 0, 0

    def row(self, kind):
        return self.rows.setdefault(kind, {"texts": 0, "ids": 0, "ids_plain": 0, "differ": 0, "decode": 0})

    def differ(self, kind, what, text, got, want):
        self.row(kind)["differ"] += 1
        self.bad += 1
        if self.shown < 8:
            self.shown += 1
            i = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            print("DIFFER %s/%s at id %d of %d/%d (ours/theirs), text %r" % (kind, what, i, len(got), len(want),
                                                                          text[:80]))
            for who, ids in (("ours  ", got), ("theirs", want)):
                near = ids[max(0, i - 2):i + 4]
                print("  %s %s %r" % (who, near, [self.tk.token_bytes(x) for x in near]))


def check_text(t: Tally, hf, kind: str, text: str) -> None:
    r = t.row(kind)
    r["texts"] += 1
    for special in (True, False):
        want = hf.encode(text, allow_special_tokens=special)
        try:
            got = t.tk.encode(text, parse_special=special)
        except Exception as e:                      # noqa: BLE001 - counted, and the run goes on
            print("RAISED %s/%s: %s: %s on %r" % (kind, special, type(e).__name__, str(e)[:80], text[:60]))
            got = []
        r["ids" if special else "ids_plain"] += len(want)
        if got != want:
            t.differ(kind, "specials" if special else "plain", text, got, want)
            continue
        back = t.tk.decode(got)
        if back != hf.decode(want) or (back != text and kind != "surrogates"):
            r["decode"] += 1
            t.bad += 1
            if t.shown < 8:
                t.shown += 1
                print("DECODE %s: %r -> %r" % (kind, text[:60], back[:60]))


def check_chat(t: Tally, hf, enc, recs, rng, rendered: list) -> tuple[int, int]:
    r = t.row("chat")
    whole_differs = refused = 0
    for rec in recs:
        msgs = rec.get("messages")
        if not msgs:
            continue
        thinking, gen = rng.random() < 0.75, rng.random() < 0.8
        effort = rng.choice(["low", "medium", "high", "max"])    # theirs defaults to "max"; used only when thinking
        try:
            want = hf.apply_chat_template(msgs, tools=rec.get("tools"), tokenize=True, add_generation_prompt=gen,
                                          thinking=thinking, thinking_effort=effort)
            segs = enc.build_chat_segments(msgs, tools=rec.get("tools"), add_generation_prompt=gen, thinking=thinking,
                                           thinking_effort=effort)
        except Exception as e:                      # noqa: BLE001 - a record their template refuses is not ours
            refused += 1
            print("refused by their template: %s" % str(e)[:100])
            continue
        got = []
        for s in segs:
            got += t.tk.encode(s.text, parse_special=s.allow_special)
        r["texts"] += 1
        r["ids"] += len(want)
        if got != want:
            t.differ("chat", "segments", "".join(s.text for s in segs), got, want)
        whole = "".join(s.text for s in segs)
        rendered.append(whole)
        if t.tk.encode(whole, parse_special=True) != want:
            whole_differs += 1                      # information: the rendered text in one encode is not the same
    return whole_differs, refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tiktoken.model, tokenizer_config.json, tokenization_kimi.py, "
                    "encoding_k3.py")
    ap.add_argument("--gguf", help="the GGUF a pack is built from (only its header is read)")
    ap.add_argument("--jsonl", nargs="*", default=[], help="chat records: messages (+ tools, raw)")
    ap.add_argument("--tokens", nargs="*", default=[], help="npz with flat + seq_offsets (calibration ids)")
    ap.add_argument("--code", nargs="*", default=[], help="source directories")
    ap.add_argument("--code-chars", type=int, default=3_000_000)
    ap.add_argument("--sweep", action="store_true", help="every code point unicodedata or regex calls assigned, "
                    "nine contexts each")
    ap.add_argument("--unicode-table", action="store_true", help="print strata's KIMI_K2_AFTER and the reclassed "
                    "code points from tiktoken's tables and this regex's, and stop")
    ap.add_argument("--min-ids", type=int, default=1_000_000)
    a = ap.parse_args()
    t0 = time.time()
    hf, enc = load_theirs(pathlib.Path(a.model).expanduser())
    if a.unicode_table:
        unicode_table(hf)
        return 0
    tk, notes = load_ours(hf, a.gguf)
    for n in notes:
        print(n)
    bad = sum(n.startswith("FAIL") for n in notes)

    # the vocabulary, id by id, and the special tokens' shape
    vb = [i for i in range(hf.vocab_size) if tk.token_bytes(i) != hf.model.decode_single_token_bytes(i)]
    names = sorted(hf.special_tokens, key=len)
    prefixed = [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if y.startswith(x)]
    specials_ok = sorted(tk.special_tokens.items(), key=lambda kv: kv[1]) == sorted(hf.special_tokens.items(),
                                                                                       key=lambda kv: kv[1])
    print("vocab: %d ids, %d differ in bytes; specials: %d, the same set and ids: %s, one a prefix of another: %d"
          % (hf.vocab_size, len(vb), len(names), "yes" if specials_ok else "NO", len(prefixed)))
    bad += len(vb) + (not specials_ok) + len(prefixed)
    bad += check_unicode(hf)

    recs = jsonl_records(a.jsonl)
    with open(HERE.parent / "tokenizer_corpus.txt", encoding="utf-8", newline="") as f:
        base = f.read()
    kinds = {"boundary": [base] + base.splitlines(keepends=True) + CASES + ["".join(CASES)],
             "messages": record_texts(recs), "calib_ids": token_texts(hf, a.tokens),
             "code": code_texts(a.code, a.code_chars)}
    kinds.update(generated(sorted(hf.special_tokens)))
    if a.sweep:
        kinds["sweep"] = sweep_texts()
    digest = hashlib.sha256()
    for texts in kinds.values():
        for x in texts:
            digest.update(x.encode("utf-8", "surrogatepass"))
    print("corpus: %d texts, %d characters, sha256 %s" % (sum(map(len, kinds.values())),
                                                           sum(len(x) for v in kinds.values() for x in v),
                                                           digest.hexdigest()[:16]))
    tally = Tally(tk)
    for kind, texts in kinds.items():
        for x in texts:
            check_text(tally, hf, kind, x)
    rendered: list[str] = []
    whole_differs, refused = check_chat(tally, hf, enc, recs, random.Random(7), rendered)
    for x in rendered:                              # the rendered chats as texts too: specials in their real places
        check_text(tally, hf, "chat_text", x)

    print("%-11s %7s %10s %10s %7s %7s" % ("kind", "texts", "ids", "ids plain", "differ", "decode"))
    for kind, r in tally.rows.items():
        print("%-11s %7d %10d %10d %7d %7d" % (kind, r["texts"], r["ids"], r["ids_plain"], r["differ"], r["decode"]))
    counted = sum(r["ids"] for k, r in tally.rows.items() if k not in ("sweep", "chat_text", "long_mix"))
    if tally.rows.get("chat", {}).get("texts"):
        print("chat: %d conversations, %d refused by their template; the rendered text in ONE encode (specials "
              "parsed) differs from their segment-by-segment ids in %d (information, not a failure)"
              % (tally.rows["chat"]["texts"], refused, whole_differs))
    bad += tally.bad
    short = counted < a.min_ids
    print("%s: %d ids compared (specials parsed, each text once: sweep, chat_text and long_mix apart), %d problems, "
          "%.0f s"
          % ("PASS" if not bad and not short else "FAIL", counted, bad, time.time() - t0)
          + (" - fewer than --min-ids %d" % a.min_ids if short else ""))
    return 0 if not bad and not short else 1


if __name__ == "__main__":
    sys.exit(main())
