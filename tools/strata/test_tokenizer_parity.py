"""tools/strata/test_tokenizer_parity.py - tokenizer_parity.py logic (B12.8).

A small byte-level BPE vocabulary with the glm4 pre-tokenizer and GLM specials stands in for the real one; the
"llama.cpp side" is a saved llama_tokenize_lines output.  When a built llama_tokenize_lines and llama.cpp's
models/ggml-vocab-qwen35.gguf are available (env LLAMA_TOKENIZE_LINES, LLAMA_VOCAB_GGUF), the end-to-end test runs
the real llama.cpp tokenizer against tools/strata_tokenizer.py on the whole corpus.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import tokenizer_parity as tp  # noqa: E402
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer  # noqa: E402

SPECIALS = [("<|endoftext|>", 3), ("[gMASK]", 3), ("<sop>", 3), ("<|system|>", 3), ("<|user|>", 3),
            ("<|assistant|>", 3), ("<|observation|>", 3), ("<think>", 4), ("</think>", 4), ("<tool_call>", 4),
            ("</tool_call>", 4), ("<arg_key>", 4), ("</arg_key>", 4), ("<arg_value>", 4), ("</arg_value>", 4)]


def small_tokenizer_dir(d: pathlib.Path) -> pathlib.Path:
    """vocab.json / merges.txt / token_type.json / tokenizer.json in the pack layout (strata_tokenizer.extract)."""
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)]
    merges = []
    for word in ("the", "and", "ing", " the", " a", "er", "token"):
        u = "".join(BYTE_TO_UNICODE[b] for b in word.encode())
        cur = u[0]
        for ch in u[1:]:
            merges.append("%s %s" % (cur, ch))
            cur = cur + ch
            if cur not in tokens:
                tokens.append(cur)
    types = [1] * len(tokens)
    for s, ty in SPECIALS:
        tokens.append(s)
        types.append(ty)
    t = d / "tokenizer"
    t.mkdir(parents=True)
    (t / "vocab.json").write_text(json.dumps({s: i for i, s in enumerate(tokens)}, ensure_ascii=False))
    (t / "merges.txt").write_text("\n".join(merges))
    (t / "token_type.json").write_text(json.dumps(types))
    (t / "tokenizer.json").write_text(json.dumps({"pre": "glm4"}))
    return t


class ParityTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.td.name)
        self.tokdir = small_tokenizer_dir(self.d)
        self.tok = Tokenizer.from_dir(self.tokdir)
        self.corpus = self.d / "corpus"
        self.corpus.mkdir()
        (self.corpus / "a.txt").write_text("the token and the thing\nline two\r\nlast line no newline", newline="")
        (self.corpus / "b.jsonl").write_text(json.dumps("<|user|>hi<|assistant|><think></think>ok") + "\n" +
                                             json.dumps({"text": "multi\nline\n\n"}) + "\n" + json.dumps("") + "\n")

    def tearDown(self):
        self.td.cleanup()

    def theirs_file(self, records, mutate=None) -> pathlib.Path:
        lines = []
        for i, (_, text) in enumerate(records):
            ids = self.tok.encode(text, parse_special=True)
            if mutate and i == mutate:
                ids = ids[:-1] + [ids[-1] + 1] if ids else [5]
            lines.append(" ".join(map(str, ids)) + "\t" + text.encode("utf-8").hex())
        p = self.d / "theirs.txt"
        p.write_text("\n".join(lines) + "\n")
        return p

    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = tp.main(["--tokenizer", str(self.tokdir), "--corpus", str(self.corpus), *args])
        return rc, out.getvalue()

    def test_records(self):
        rec = tp.load_records(self.corpus)
        self.assertEqual([r[0] for r in rec], ["a.txt:1", "a.txt:2", "a.txt:3", "b.jsonl:1", "b.jsonl:2", "b.jsonl:3"])
        self.assertEqual(rec[1][1], "line two\r\n")             # CRLF kept
        self.assertEqual(rec[4][1], "multi\nline\n\n")
        self.assertEqual(rec[5][1], "")

    def test_identical_passes(self):
        rec = tp.load_records(self.corpus)
        rc, md = self.run_main("--theirs", str(self.theirs_file(rec)))
        self.assertEqual(rc, 0, md)
        self.assertIn("**PASS**", md)
        self.assertIn("| Identical ids | 6 (100.000%) |", md)

    def test_one_divergence_fails_and_is_reported(self):
        rec = tp.load_records(self.corpus)
        rc, md = self.run_main("--theirs", str(self.theirs_file(rec, mutate=3)))
        self.assertEqual(rc, 1)
        self.assertIn("**FAIL**", md)
        self.assertIn("| b.jsonl:1 |", md)

    def test_their_bad_round_trip_fails(self):
        rec = tp.load_records(self.corpus)
        p = self.theirs_file(rec)
        lines = p.read_text().splitlines()
        ids, _ = lines[0].split("\t")
        lines[0] = ids + "\t" + "XX".encode().hex()
        p.write_text("\n".join(lines) + "\n")
        rc, md = self.run_main("--theirs", str(p))
        self.assertEqual(rc, 1)
        self.assertIn("NO (1 records)", md)

    def test_specials_are_single_tokens(self):
        ids = self.tok.encode("<|user|>hi<think>", parse_special=True)
        self.assertEqual(ids[0], self.tok.ids["<|user|>"])
        self.assertEqual(ids[-1], self.tok.ids["<think>"])

    def test_write_records_format(self):
        out = self.d / "rec.bin"
        rc, _ = self.run_main("--write-records", str(out))
        self.assertEqual(rc, 0)
        data = out.read_bytes()
        recs, pos = [], 0
        while pos < len(data):
            nl = data.index(b"\n", pos)
            n = int(data[pos:nl])
            recs.append(data[nl + 1:nl + 1 + n].decode())
            pos = nl + 1 + n
        self.assertEqual(recs, [t for _, t in tp.load_records(self.corpus)])

    def test_render_template_records(self):
        tpl = self.d / "t.jinja"
        tpl.write_text("[gMASK]<sop>{% if tools %}<|system|>tools{% endif %}{% for m in messages %}<|{{ m.role }}|>"
                       "{{ m.content }}{% endfor %}{% if reasoning_effort in ['low', 'high'] %}E{% endif %}"
                       "{% if add_generation_prompt %}<|assistant|><think>{% endif %}")
        rec = tp.render_records(str(tpl))
        self.assertEqual(len(rec), len(tp.CONVERSATIONS))
        self.assertTrue(all(t.startswith("[gMASK]<sop>") for _, t in rec))
        self.assertTrue(rec[-1][1].endswith("<think></think>"))     # enable_thinking False closes the block

    def test_committed_corpus_round_trips(self):
        """Every committed corpus record survives our byte-level round trip (no llama.cpp needed)."""
        rec = tp.load_records(tp.CORPUS)
        self.assertGreater(len(rec), 300)
        for rid, text in rec:
            ids = self.tok.encode(text, parse_special=True)
            self.assertEqual(b"".join(self.tok.token_bytes(i) for i in ids), text.encode("utf-8"), rid)


LINES = os.environ.get("LLAMA_TOKENIZE_LINES")
VOCAB = os.environ.get("LLAMA_VOCAB_GGUF")


@unittest.skipUnless(LINES and VOCAB and os.path.exists(LINES) and os.path.exists(VOCAB),
                     "set LLAMA_TOKENIZE_LINES and LLAMA_VOCAB_GGUF (e.g. llama.cpp models/ggml-vocab-qwen35.gguf)")
class EndToEndTest(unittest.TestCase):
    def test_corpus_against_llama_cpp(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = tp.main(["--gguf", VOCAB, "--llama-lines", LINES])
        self.assertEqual(rc, 0, out.getvalue())


if __name__ == "__main__":
    unittest.main()
