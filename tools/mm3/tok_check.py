"""tools/mm3/tok_check.py - the server's tokenizer (pre "minimax") against `tokenizers` on MiniMax's tokenizer.json, id
for id.

    python tools/mm3/tok_check.py --model ~/models/MiniMax-M3.1-Flash-EXL3-3.5bpw [--corpus FILE]

Builds the pack's tokenizer/ (tools/mm3/pack.py's write_tokenizer) in a temporary directory, then encodes
tools/tokenizer_corpus.txt, the MiniMax cases below and FILE line by line and whole, with the added tokens parsed
(`tokenizers` always parses them), and reports the first difference of each disagreeing text with the round trip.
Exit 0 only when every text agrees and decodes back to itself.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pack  # noqa: E402
from strata_tokenizer import Tokenizer  # noqa: E402

CASES = [
    "e\u0301t\u00e9 caf\u00e9 cafe\u0301 \u212b \u0041\u030a",                   # NFC composes these
    "\u1100\u1161\u11a8 \uac01 \u1112\u1161\u11ab\u1100\u1173\u11af",          # Hangul jamo -> syllables
    "]~!b[]~b]system\nYou are a helpful assistant.[e~[\n]~b]user\nhi[e~[\n]~b]ai\n<mm:think>ok</mm:think>Hello",
    "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[<city>Paris]<]minimax[>[</city>"
    "]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    "]~b]tool\n<response>{\"temp\": 21.5}</response>[e~[",
    "\n\n<effort>xhigh</effort>",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002 \u3053\u3093\u306b\u3061\u306f\u4e16\u754c "
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645 "
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\U0001F469\u200D\U0001F4BB \U0001F1EF\U0001F1F5 \U0001F44D\U0001F3FD emoji \u2764\ufe0f",
    "def f(x):\n\treturn {'a': [1, 2, 3]}  # comment\r\n    if x>=10//3: pass\n\n\n",
    "path/to/file.py:123  https://example.com/a/b?c=d&e=f  C:\\Users\\me  user@example.com",
    "camelCaseIdentifier snake_case_name SCREAMING_CASE HTTPServer iPhone McDonald's O'Neil",
    "   leading  \t\ttabs and trailing spaces   \n \n\r\n\r\r\n",
    "1 12 123 1234 12345 3.14159 1,000,000 2026-10-24T16:00:00Z 0x83DCD12D",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a directory with tokenizer.json, special_tokens_map.json, "
                    "chat_template.jinja (the pack, or its dspark/)")
    ap.add_argument("--corpus", help="more text, line by line and whole")
    a = ap.parse_args()
    from tokenizers import Tokenizer as HF
    model = pathlib.Path(a.model).expanduser()
    hf = HF.from_file(str(model / "tokenizer.json"))
    with tempfile.TemporaryDirectory() as d:
        pack.write_tokenizer(pathlib.Path(d), model)
        tk = Tokenizer.from_dir(d)
    with open(HERE.parent / "tokenizer_corpus.txt", encoding="utf-8", newline="") as f:
        base = f.read()
    texts = [base] + base.splitlines(keepends=True) + CASES + ["".join(CASES)]
    if a.corpus:
        with open(a.corpus, encoding="utf-8", newline="") as f:
            extra = f.read()
        texts += [extra] + extra.splitlines(keepends=True)
    bad = 0
    n_ids = 0
    for t in texts:
        want = hf.encode(t, add_special_tokens=False).ids
        got = tk.encode(t, parse_special=True)
        n_ids += len(want)
        if got != want:
            bad += 1
            i = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            print("DIFFER at id %d of %r:\n  ours   %s\n  theirs %s" % (i, t[:120], got[max(0, i - 3):i + 5],
                                                                     want[max(0, i - 3):i + 5]))
            print("         %r vs %r" % ([tk.token_bytes(x) for x in got[max(0, i - 1):i + 3]],
                                         [tk.token_bytes(x) for x in want[max(0, i - 1):i + 3]]))
        import unicodedata
        if tk.decode(got) != unicodedata.normalize("NFC", t) and tk.decode(got) != t:
            bad += 1
            print("ROUND TRIP %r -> %r" % (t[:80], tk.decode(got)[:80]))
    print("%d texts, %d ids: %s" % (len(texts), n_ids, "all agree" if not bad else "%d problems" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
