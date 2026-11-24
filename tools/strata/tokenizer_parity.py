#!/usr/bin/env python3
"""tools/strata/tokenizer_parity.py - pack tokenizer vs llama.cpp tokenizer, record by record (BUILD-PLAN B12.8).

The engine's prompts are tokenized by the pack tokenizer (tools/strata_tokenizer.py: glm4/glm5 pre-tokenizer with
ignore_merges); the parity oracle (llama.cpp aa94f208) tokenizes with its own.  Acceptance: ids identical on 100% of
the records, and the detokenized round trip byte-identical (ours: decode(encode(x)) == x; llama.cpp's: its
detokenization of its ids == x).  M2 criterion (i) is this check on the chosen GGUF.

Records come from the corpus directory (default data/glm-dsa/tokenizer-corpus/): every line of a *.txt file is a
record (its newline included, so newline handling is tested), every line of a *.jsonl file is one JSON string
record (CR, multi-line text, rendered chat prompts).  `--render-template` adds records rendered from a chat
template (the GGUF's own, extracted into the pack as tokenizer/chat_template.jinja) over fixed conversations.

llama.cpp side, one of:
  --llama-lines BIN   tools/strata/llama_tokenize_lines.cpp (vocabulary loaded once; fast; also detokenizes)
  --llama-tokenize BIN  stock llama-tokenize, one process per record (slow: use --limit)
  --theirs FILE       a saved output of llama_tokenize_lines

    python3 tools/strata/tokenizer_parity.py --gguf $M --llama-lines build/llama_tokenize_lines \\
        --render-template <pack>/tokenizer/chat_template.jinja --report docs/strata/volta3/tokenizer-parity.md
    python3 tools/strata/tokenizer_parity.py --tokenizer <pack>/tokenizer --model $M --write-records /tmp/rec.bin

Exit 0 only when every record agrees and both round trips are exact.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "glm-dsa" / "tokenizer-corpus"
sys.path.insert(0, str(ROOT / "tools"))

# fixed conversations for --render-template (system prompt, tools, multi-turn, tool call and result, Chinese)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}, "unit": {"type": "string",
                                                                                    "enum": ["c", "f"]}},
                                               "required": ["city"]}}}]
CONVERSATIONS = [
    ({"messages": [{"role": "user", "content": "Hello! What is 17 * 23?"}]}, {}),
    ({"messages": [{"role": "system", "content": "You are a careful assistant. Answer briefly."},
                   {"role": "user", "content": "Name three rivers in Europe."},
                   {"role": "assistant", "content": "The Danube, the Rhine and the Loire."},
                   {"role": "user", "content": "And one in Asia?"}]}, {"reasoning_effort": "high"}),
    ({"messages": [{"role": "user", "content": "\u8bf7\u7528\u4e2d\u6587\u89e3\u91ca\u4e00\u4e0b\u4ec0\u4e48\u662f"
                                               "\u7f13\u5b58\u3002"}]}, {"reasoning_effort": "low"}),
    ({"messages": [{"role": "user", "content": "What's the weather in Lisbon?"},
                   {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                       "name": "get_weather", "arguments": {"city": "Lisbon", "unit": "c"}}}]},
                   {"role": "tool", "content": "{\"temp\": 21, \"sky\": \"clear\"}"}], "tools": TOOLS}, {}),
    ({"messages": [{"role": "user", "content": "def f(x):\n    return x ** 2\n\nWhat does f(12) return?"}]},
     {"enable_thinking": False}),
]


# ------------------------------------------------------------------ records
def load_records(corpus: pathlib.Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for f in sorted(corpus.glob("*.txt")):
        with open(f, encoding="utf-8", newline="") as fh:
            text = fh.read()
        # split at "\n" only (str.splitlines would also split at \r, \x0b, \x1c-\x1e, \x85, \u2028, \u2029)
        for i, line in enumerate(re.findall(r"[^\n]*\n|[^\n]+$", text)):
            out.append(("%s:%d" % (f.name, i + 1), line))
    for f in sorted(corpus.glob("*.jsonl")):
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n")):
            if line.strip():
                v = json.loads(line)
                out.append(("%s:%d" % (f.name, i + 1), v if isinstance(v, str) else v["text"]))
    return out


def render_records(template_path: str) -> list[tuple[str, str]]:
    sys.path.insert(0, str(ROOT))
    from serve.frontend import ChatTemplate
    tpl = ChatTemplate(template_path)
    out = []
    for i, (conv, kw) in enumerate(CONVERSATIONS):
        text = tpl.render(conv["messages"], tools=conv.get("tools"), add_generation_prompt=True, **dict(kw))
        out.append(("template:%d" % (i + 1), text))
    return out


def write_records(records: list[tuple[str, str]], path: pathlib.Path) -> None:
    with open(path, "wb") as f:
        for _, text in records:
            b = text.encode("utf-8")
            f.write(b"%d\n" % len(b))
            f.write(b)


# ------------------------------------------------------------------ the two sides
def ours(tokenizer, records) -> list[list[int]]:
    return [tokenizer.encode(text, parse_special=True) for _, text in records]


def parse_theirs(text: str) -> list[tuple[list[int], bytes | None]]:
    rows = []
    for line in text.splitlines():
        ids_s, _, hx = line.partition("\t")
        rows.append(([int(x) for x in ids_s.split()], bytes.fromhex(hx) if "\t" in line else None))
    return rows


def theirs_lines(binary: str, model: str, records) -> list[tuple[list[int], bytes | None]]:
    with tempfile.TemporaryDirectory() as td:
        rec = pathlib.Path(td) / "records.bin"
        write_records(records, rec)
        r = subprocess.run([binary, model, str(rec)], capture_output=True, check=False)
        if r.returncode != 0:
            raise RuntimeError("%s failed: %s" % (binary, r.stderr.decode(errors="replace")[-2000:]))
        return parse_theirs(r.stdout.decode("ascii"))


def theirs_tokenize(binary: str, model: str, records) -> list[tuple[list[int], bytes | None]]:
    """Stock llama-tokenize, one process per record: --ids --no-bos --no-escape, specials parsed (its default)."""
    out = []
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "rec.txt"
        for _, text in records:
            p.write_bytes(text.encode("utf-8"))
            r = subprocess.run([binary, "-m", model, "-f", str(p), "--ids", "--no-bos", "--no-escape", "--log-disable"],
                               capture_output=True, check=False)
            if r.returncode != 0:
                raise RuntimeError("%s failed: %s" % (binary, r.stderr.decode(errors="replace")[-2000:]))
            line = [l for l in r.stdout.decode().splitlines() if l.startswith("[")]
            out.append((json.loads(line[-1]) if line else [], None))
    return out


# ------------------------------------------------------------------ comparison
def compare(records, mine, theirs, tokenizer, show: int = 20) -> dict:
    if len(theirs) != len(records):
        raise RuntimeError("llama.cpp side has %d records, corpus %d" % (len(theirs), len(records)))
    diffs, rt_ours, rt_theirs = [], [], []
    for (rid, text), a, (b, detok) in zip(records, mine, theirs):
        if a != b:
            at = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            diffs.append({"record": rid, "at": at, "ours": a[max(0, at - 3):at + 5], "theirs": b[max(0, at - 3):at + 5],
                          "ours_n": len(a), "theirs_n": len(b), "text": text[:120]})
        if b"".join(tokenizer.token_bytes(i) for i in a) != text.encode("utf-8"):
            rt_ours.append(rid)
        if detok is not None and detok != text.encode("utf-8"):
            rt_theirs.append(rid)
    n = len(records)
    return {"records": n, "identical": n - len(diffs), "identical_pct": 100.0 * (n - len(diffs)) / max(1, n),
            "diffs": diffs[:show], "n_diffs": len(diffs), "roundtrip_ours_bad": rt_ours[:show],
            "n_roundtrip_ours_bad": len(rt_ours), "roundtrip_theirs_bad": rt_theirs[:show],
            "n_roundtrip_theirs_bad": len(rt_theirs), "theirs_detok": any(d is not None for _, d in theirs),
            "tokens": sum(len(a) for a in mine)}


def report_md(r: dict, meta: dict) -> str:
    ok = r["n_diffs"] == 0 and r["n_roundtrip_ours_bad"] == 0 and r["n_roundtrip_theirs_bad"] == 0
    out = ["# Tokenizer parity (B12.8)", "",
           "Pack tokenizer (`tools/strata_tokenizer.py`, pre `%s`) vs llama.cpp (%s) on `%s`; corpus `%s`." % (
               meta["pre"], meta["oracle"], meta["model"], meta["corpus"]), "",
           "| Check | Result |", "|---|---|",
           "| Records | %d (%d tokens) |" % (r["records"], r["tokens"]),
           "| Identical ids | %d (%.3f%%) |" % (r["identical"], r["identical_pct"]),
           "| Our round trip exact | %s |" % ("yes" if not r["n_roundtrip_ours_bad"] else "NO (%d records)" % r["n_roundtrip_ours_bad"]),
           "| llama.cpp round trip exact | %s |" % (("yes" if not r["n_roundtrip_theirs_bad"] else
                                                    "NO (%d records)" % r["n_roundtrip_theirs_bad"])
                                                   if r["theirs_detok"] else "not measured (llama-tokenize mode)"),
           "| **Verdict** | **%s** |" % ("PASS" if ok else "FAIL")]
    if r["diffs"]:
        out += ["", "## First divergences", "", "| Record | Index | Ours | llama.cpp | Text |", "|---|---|---|---|---|"]
        for d in r["diffs"]:
            out.append("| %s | %d | %s | %s | `%s` |" % (d["record"], d["at"], d["ours"], d["theirs"],
                                                       d["text"].replace("|", "\\|").replace("\n", "\\n")))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="GGUF whose tokenizer metadata both sides use (the chosen model's first shard)")
    src.add_argument("--tokenizer", help="pack tokenizer/ directory (tools/strata_tokenizer.py extract output)")
    ap.add_argument("--model", help="GGUF for the llama.cpp side (default: --gguf)")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--render-template", help="chat template to render fixed conversations with (extra records)")
    side = ap.add_mutually_exclusive_group()
    side.add_argument("--llama-lines", help="tools/strata/llama_tokenize_lines binary")
    side.add_argument("--llama-tokenize", help="stock llama-tokenize binary (one process per record)")
    side.add_argument("--theirs", help="saved llama_tokenize_lines output")
    ap.add_argument("--write-records", help="write the length-prefixed record file and exit")
    ap.add_argument("--limit", type=int, help="first N records only")
    ap.add_argument("--report", help="also write the markdown report here")
    a = ap.parse_args(argv)
    from strata_tokenizer import Tokenizer
    try:
        tok = Tokenizer.from_gguf(a.gguf) if a.gguf else Tokenizer.from_dir(a.tokenizer)
        records = load_records(pathlib.Path(a.corpus))
        if a.render_template:
            records += render_records(a.render_template)
        if a.limit:
            records = records[:a.limit]
        if not records:
            raise RuntimeError("no records in %s" % a.corpus)
        if a.write_records:
            write_records(records, pathlib.Path(a.write_records))
            print("wrote %d records to %s" % (len(records), a.write_records))
            return 0
        model = a.model or a.gguf
        if a.theirs:
            theirs = parse_theirs(pathlib.Path(a.theirs).read_text(encoding="ascii"))
            oracle = "saved output %s" % a.theirs
        elif a.llama_lines or a.llama_tokenize:
            if not model:
                raise RuntimeError("--model (or --gguf) is needed for the llama.cpp side")
            theirs = theirs_lines(a.llama_lines, model, records) if a.llama_lines else \
                theirs_tokenize(a.llama_tokenize, model, records)
            oracle = a.llama_lines or a.llama_tokenize
        else:
            raise RuntimeError("give --llama-lines, --llama-tokenize or --theirs (or --write-records)")
        r = compare(records, ours(tok, records), theirs, tok)
    except (OSError, ValueError, RuntimeError) as e:
        print("tokenizer_parity: %s" % e, file=sys.stderr)
        return 2
    md = report_md(r, {"pre": tok.pre, "oracle": oracle, "model": model or "-", "corpus": a.corpus})
    sys.stdout.write(md)
    if a.report:
        pathlib.Path(a.report).write_text(md, encoding="utf-8")
    return 0 if r["n_diffs"] == 0 and r["n_roundtrip_ours_bad"] == 0 and r["n_roundtrip_theirs_bad"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
