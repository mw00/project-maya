"""tools/maya_quant/ctx_fill_test.py - the context-fill QA: the context filled with real text, a fact planted at its
very start, the question at its end.  Per size: did the engine run without an error, did the answer find the fact,
did it end by itself, and did it stay free of loops - plus the prompt and decode speed at that size.

    python ctx_fill_test.py --pack PACK [--sizes 8000,16000,30000] [--max-context 32768]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import tempfile
import time

from tokenizers import Tokenizer

from loop_test import EOS, THINK_END, repeats

FACT = "Note for whoever reads this archive: the launch code for Project Maya is ORCHID-7342-KESTREL."
QUESTION = ("Above is a long archive of documents. At its very beginning there is a note with a launch code. "
            "What is the launch code? Reply with the code and one sentence about where you found it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tok", default="/mnt/nvme/maya/glm53-fp8/tokenizer.json")
    ap.add_argument("--exe", default=str(pathlib.Path(__file__).resolve().parents[2] / "build" / "maya"))
    ap.add_argument("--sizes", default="8000,16000,30000", help="prompt sizes in tokens")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--max-new", type=int, default=1500)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tok)
    root = pathlib.Path(__file__).resolve().parents[2]
    docs = [p.read_text(encoding="utf-8", errors="replace") for p in sorted((root / "docs").glob("*.md"))]
    corpus = "\n\n---\n\n".join(docs)
    body_ids = tok.encode(corpus, add_special_tokens=False).ids
    for size in (int(s) for s in a.sizes.split(",")):
        head = tok.encode(f"[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{FACT}\n\n", add_special_tokens=False).ids
        tail = tok.encode(f"\n\n{QUESTION}<|assistant|><think>", add_special_tokens=False).ids
        need = size - len(head) - len(tail)
        filler = (body_ids * (need // max(1, len(body_ids)) + 1))[:need]
        ids = head + filler + tail
        # a long prompt goes through a file: one argument of 30k ids is past Linux's 128 KB per-argument limit
        with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
            f.write(",".join(map(str, ids)))
        t0 = time.time()
        r = subprocess.run([a.exe, "--glm-pack", a.pack, "--max-context", str(a.max_context), "--greedy",
                            "--eos-ids", ",".join(map(str, sorted(EOS))),
                            "--max-new", str(a.max_new), "--tokens-file", f.name],
                           capture_output=True, text=True, cwd=root, env=dict(os.environ, STRATA_GLM_USAGE="0"))
        os.unlink(f.name)
        out = [int(l.split()[1]) for l in r.stdout.splitlines() if l.startswith("T ")]
        text = tok.decode(out, skip_special_tokens=False)
        answer = text.split(THINK_END, 1)[1] if THINK_END in text else ""
        found = "ORCHID-7342-KESTREL" in answer
        finished = bool(out) and (out[-1] in EOS or len(out) < a.max_new)   # stopped at an end token
        longest, share = repeats(out)
        stat = next((l for l in r.stderr.splitlines() if l.startswith("glm stat")), "")
        speed = re.search(r"decode [\d.]+ ms/tok \(([\d.]+) tok/s\).*prompt ([\d.]+) ms/tok", stat)
        ok = r.returncode == 0 and found and finished and longest < 3 and share < 0.15
        print(f"{'OK ' if ok else 'BAD'} {len(ids):6d}-token prompt: engine {'ok' if r.returncode == 0 else 'ERROR'} | "
              f"found the fact {found} | finished {finished} | longest repeat {longest}x32 | "
              f"{len(out)} tokens in {time.time() - t0:.0f} s" +
              (f" | decode {speed.group(1)} tok/s, prompt {speed.group(2)} ms/token" if speed else ""), flush=True)
        print("   answer: " + answer.strip()[:300].replace("\n", " | "), flush=True)
        if r.returncode != 0:
            print("   engine: " + r.stderr[-500:].replace("\n", " | "), flush=True)


if __name__ == "__main__":
    main()
