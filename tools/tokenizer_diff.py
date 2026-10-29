"""tools/tokenizer_diff.py - our tokenizer vs llama-tokenize, id for id (second-opinion item 5).

    python tools/tokenizer_diff.py <vocab.gguf> <llama_ids.txt> [corpus]

<llama_ids.txt> is `llama-tokenize --ids-only <corpus>` output: one id per line, the whole file
tokenized as one text with specials parsed.  Our side loads the SAME vocab-only GGUF, encodes the
corpus with parse_special=True, and diffs the two sequences, reporting the first divergence with
context and byte-level token text.  Exit 0 only on exact agreement.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer  # noqa: E402


def main() -> int:
    vocab_gguf, llama_ids = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    corpus_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else \
        pathlib.Path(__file__).resolve().parent / "tokenizer_corpus.txt"
    # preserve CRLF: the pre-tokenizer treats it differently
    with open(corpus_path, encoding="utf-8", newline="") as f:
        corpus = f.read()

    tk = Tokenizer.from_gguf(vocab_gguf)
    ours = tk.encode(corpus, parse_special=True)
    text = llama_ids.read_text().strip()
    theirs = json.loads(text) if text.startswith("[") else [int(x) for x in text.split()]

    if ours == theirs:
        print("tokenizers agree: %d ids, %d pieces" % (len(ours), len(set(ours))))
        return 0

    n = min(len(ours), len(theirs))
    at = next((i for i in range(n) if ours[i] != theirs[i]), n)
    show = 12
    print("DISAGREE at flat index %d of %d/%d (ours/theirs)" % (at, len(ours), len(theirs)))
    print("  ours  [...%s]" % " | ".join(repr(tk.tokens[i]) for i in ours[max(0, at - 3):at + show]))
    print("  llama [...%s]" % " | ".join(repr(tk.tokens[i]) for i in theirs[max(0, at - 3):at + show]))
    # how far until the sequences re-synchronize?
    probe = ours[at:at + 40]
    for back in range(0, 40):
        if theirs[at + back:at + back + 4] == probe[0:4]:
            print("  resync after %d id(s)" % back)
            break
    print("FAIL: %d vs %d ids (%+d)" % (len(ours), len(theirs), len(ours) - len(theirs)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
