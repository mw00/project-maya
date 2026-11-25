"""tools/ling_quant/ling_loop_test.py - does a Ling quant loop?  tools/maya_quant/loop_test.py's prompts and checks, run in
maya-ling (PR #1224): long answers with thinking on, at the model card's sampling (T=1.0, top-p 0.95, top-k 20) and
greedy (which loops first if anything does), each checked for

    finished      it stopped by itself (end of turn) before --max-new
    think closed  the reasoning block was closed (</think>) and an answer followed
    repeats       the longest back-to-back repeat of a 32-token window, and the share of the output inside repeated
                  64-grams (a healthy answer: ~0; a loop: most of the tail)

    python ling_loop_test.py --gguf Q.gguf [--max-new 4096] [--label NAME] [--json out.jsonl] [-- engine args]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "maya_quant"))
from loop_test import PROMPTS, repeats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--exe", default=P.MAYA_LING)
    ap.add_argument("--model", default=P.FP8, required=P.FP8 is None)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--save", default=None, help="a folder for the answers' text")
    ap.add_argument("engine", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.engine if x != "--"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    modes = [("sampled", ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--seed", str(a.seed)])]
    if not a.no_greedy:
        modes.append(("greedy", ["--temp", "0"]))
    rows = []
    label = a.label or pathlib.Path(a.gguf).name
    save = pathlib.Path(a.save) if a.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for name, text in PROMPTS:
            s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=True)
            ids = tok(s, add_special_tokens=False)["input_ids"]
            fid = pathlib.Path(td) / "ids.txt"
            fid.write_text(",".join(map(str, ids)))
            for mode, margs in modes:
                t0 = time.time()
                cmd = [a.exe, "--model", a.gguf, "--prompt-ids", str(fid), "--n-predict", str(a.max_new), "--ctx",
                       str(len(ids) + a.max_new + 64), "--print-text"] + margs + extra
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"{name}/{mode}: maya-ling failed\n{r.stderr[-3000:]}")
                m = re.search(r"decode: (\d+) tokens in ([\d.]+) s \(([\d.]+) t/s\)", r.stderr)
                n_dec = int(m.group(1)) if m else -1
                tps = float(m.group(3)) if m else float("nan")
                out = r.stdout
                out_ids = tok(out, add_special_tokens=False)["input_ids"]
                win, rep = repeats(out_ids)
                closed = "</think>" in out and out.split("</think>", 1)[1].strip() != ""
                finished = 0 < n_dec < a.max_new
                row = {"label": label, "prompt": name, "mode": mode, "tokens": n_dec, "finished": finished,
                       "think_closed": closed, "repeat_windows": win, "repeated_share": round(rep, 4),
                       "decode_tps": tps, "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"{name:8s} {mode:7s} {n_dec:5d} tok  finished {finished!s:5s}  think closed {closed!s:5s}  "
                      f"repeat x{win}  repeated {100 * rep:5.1f}%  {tps:6.1f} tok/s", flush=True)
                if save:
                    (save / f"{name}-{mode}.txt").write_text(out)
    bad = [r for r in rows if not r["finished"] or not r["think_closed"] or r["repeated_share"] > 0.2]
    print(f"{label}: {len(rows) - len(bad)}/{len(rows)} clean "
          f"(finished, thinking closed, under 20% repeated)", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
