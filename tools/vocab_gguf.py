"""tools/vocab_gguf.py - a vocab-only GGUF from a HF tokenizer.json, for the llama-tokenize diff.

Second-opinion item 5's closing move: the glm4/glm5 pre-tokenizer transcription was verified
against the source, but the ID-level behavior of the whole tokenizer (byte mapping, ignore_merges,
merge order) has never been diffed against llama.cpp on real text.  This writes the vocabulary of
a HF `tokenizer.json` (BPE) as a GGUF with NO tensors - the same artifact llama.cpp's
`convert_hf_to_gguf.py --vocab-only` produces - so `llama-tokenize` can run it, and the same
vocabulary feeds tools/strata_tokenizer.py on our side.  Same ids or the diff fails.

    python tools/vocab_gguf.py --tokenizer tokenizer.json --out vocab.gguf [--arch glm5-next]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

GGUF_V3 = 3
STRING, U32, F32, BOOL, I32, ARRAY = 8, 4, 0, 7, 5, 9
ALIGN = 32


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(v): return struct.pack("<I", U32) + struct.pack("<I", v)
def kv_f32(v): return struct.pack("<I", F32) + struct.pack("<f", v)
def kv_i32(v): return struct.pack("<I", I32) + struct.pack("<i", v)
def kv_str(v): return struct.pack("<I", STRING) + _s(v)
def kv_arr(elem_tag: int, elems: list[bytes]) -> bytes:
    return struct.pack("<I", ARRAY) + struct.pack("<I", elem_tag) + struct.pack("<Q", len(elems)) + b"".join(elems)
def arr_str(vs): return kv_arr(STRING, [_s(v) for v in vs])
def arr_f32(vs): return kv_arr(F32, [struct.pack("<f", v) for v in vs])
def arr_i32(vs): return kv_arr(I32, [struct.pack("<i", v) for v in vs])


TOKEN_TYPE_NORMAL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED = 1, 3, 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="glm5-next")
    ap.add_argument("--pre", default="glm5")
    a = ap.parse_args()

    tk = json.load(open(a.tokenizer, encoding="utf-8"))
    if tk["model"]["type"] != "BPE":
        print("model type %r; this writer handles BPE only" % tk["model"]["type"])
        return 1
    vocab = tk["model"]["vocab"]                       # token -> id
    # HF stores merges either as "a b" strings or as ["a", "b"] pairs; GGUF wants the strings
    merges = [" ".join(m) if isinstance(m, list) else m for m in tk["model"]["merges"]]

    tokens: dict[int, str] = {i: t for t, i in vocab.items()}
    types: dict[int, int] = {i: TOKEN_TYPE_NORMAL for i in tokens}
    for at in tk.get("added_tokens", []):
        i, sp = at["id"], bool(at.get("special"))
        tokens[i] = at["content"]
        types[i] = TOKEN_TYPE_CONTROL if sp else TOKEN_TYPE_USER_DEFINED
    n_vocab = max(tokens) + 1
    missing = [i for i in range(n_vocab) if i not in tokens]
    if missing:
        print("token ids not contiguous: %d gaps (first %r)" % (len(missing), missing[:3]))
        return 1

    md = [
        ("general.architecture", kv_str(a.arch)),
        ("general.name", kv_str("vocab-only")),
        ("general.alignment", kv_u32(ALIGN)),
        (a.arch + ".vocab_size", kv_u32(n_vocab)),
        ("tokenizer.ggml.model", kv_str("gpt2")),
        ("tokenizer.ggml.pre", kv_str(a.pre)),
        ("tokenizer.ggml.tokens", arr_str([tokens[i] for i in range(n_vocab)])),
        ("tokenizer.ggml.scores", arr_f32([0.0] * n_vocab)),
        ("tokenizer.ggml.token_type", arr_i32([types[i] for i in range(n_vocab)])),
        ("tokenizer.ggml.merges", arr_str(merges)),
    ]

    out = pathlib.Path(a.out)
    # PREFER llama.cpp's own writer: the hand-rolled writer below desyncs llama's GGUF reader on
    # large string arrays (the merges KV reads back as an empty key at ~1000+ entries) - the
    # root cause was not chased, because the gguf-py path is the canonical one.  The fallback
    # stays for environments without gguf-py and is fine for SMALL vocabs (the seed-1234 synth
    # loads cleanly either way).
    try:
        from _paths import add_gguf_py
        add_gguf_py()
        import gguf as G
    except (ImportError, SystemExit):
        G = None
    if G is not None:
        w = G.GGUFWriter(str(out), a.arch)
        w.add_name("vocab-only")
        w.add_uint32(a.arch + ".vocab_size", n_vocab)
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre(a.pre)
        w.add_token_list([tokens[i] for i in range(n_vocab)])
        w.add_token_scores([0.0] * n_vocab)
        w.add_token_types([types[i] for i in range(n_vocab)])
        w.add_token_merges(merges)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        w.close()
        writer = "gguf-py"
    else:
        header = b"GGUF" + struct.pack("<IQQ", GGUF_V3, 0, len(md))
        kvs = b"".join(_s(k) + v for k, v in md)
        pad = (-(len(header) + len(kvs))) % ALIGN
        out.write_bytes(header + kvs + b"\0" * pad)
        writer = "builtin (small vocabs only)"
    print("vocab gguf [%s]: %d tokens, %d merges, %d bytes -> %s"
          % (writer, n_vocab, len(merges), out.stat().st_size, out.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
