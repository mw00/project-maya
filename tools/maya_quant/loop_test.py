"""tools/maya_quant/loop_test.py - does a quant loop?  Long answers at the default sampling (T=1.0, top-p 0.95; and
greedy, which loops first if anything does) to prompts that need long, structured output - a three.js scene, a
canvas animation, an explanation, a plan in Portuguese, a maths derivation - each checked for:

    finished      it stopped by itself (an end token) before --max-new
    think closed  the reasoning block was closed (</think>) and an answer followed
    repeats       the longest run of one 32-token window repeating back to back, and the share of the output that
                  repeated 64-grams cover (a healthy answer: ~0; a loop: most of the tail)

    python loop_test.py --pack PACK [--max-new 6000] [--runs 1] [--no-greedy] [--save DIR]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

from tokenizers import Tokenizer

PROMPTS = [
    ("threejs", "Create a single HTML file with a three.js scene: a small solar system with the sun, three planets on "
                "elliptical orbits with moons, a starfield background, bloom on the sun and OrbitControls. Use an "
                "import map for three.js from a CDN. Give the complete file."),
    ("canvas", "Write a complete single-page HTML/CSS/JS animation on a <canvas>: a flock of 300 boids with "
               "separation, alignment and cohesion, that avoid the mouse cursor, with trails. One file, no libraries."),
    ("explain", "Explain how a transformer language model generates text, from tokenization to sampling, for a "
                "programmer who has never studied machine learning. Be thorough and use examples."),
    ("pt-br", "Faça um plano detalhado de estudos de 12 semanas para aprender desenvolvimento web full stack, com "
              "objetivos semanais, projetos práticos e recursos gratuitos."),
    ("maths", "Prove that there are infinitely many primes of the form 4k+3, then explain why the same argument "
              "does not work for primes of the form 4k+1."),
]
EOS = {154820, 154827, 154829}
THINK_END = "</think>"


def repeats(ids):
    """(longest back-to-back repeat of a 32-token window, in windows; share of tokens inside repeated 64-grams)"""
    best = 0
    W = 32
    for start in range(0, max(0, len(ids) - 2 * W), 8):
        win, k = ids[start:start + W], 1
        while ids[start + k * W:start + (k + 1) * W] == win:
            k += 1
        best = max(best, k)
    seen, rep = {}, set()
    for i in range(0, max(0, len(ids) - 64)):
        g = tuple(ids[i:i + 64])
        if g in seen:
            rep.update(range(i, i + 64))
        else:
            seen[g] = i
    return best, len(rep) / max(1, len(ids))


class _PackTok:
    """A Maya pack's tokenizer folder (vocab.json + merges + the GGUF's pre-tokenizer; tools/strata_tokenizer.py) behind
    the two calls this script makes - for models with no Hugging Face tokenizer.json at hand (glm-dsa GGUFs)."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def __init__(self, path):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        import strata_tokenizer
        self.t = strata_tokenizer.Tokenizer.from_dir(pathlib.Path(path))

    def encode(self, text, add_special_tokens=False):
        return self._Enc(self.t.encode(text, parse_special=True))

    def decode(self, ids, skip_special_tokens=False):
        return self.t.decode(ids)


def load_tok(path):
    """--tok: a tokenizer.json, or a pack's tokenizer folder (vocab.json inside)"""
    if (pathlib.Path(path) / "vocab.json").exists():
        return _PackTok(path)
    return Tokenizer.from_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tok", default="/mnt/nvme/maya/glm53-fp8/tokenizer.json")
    ap.add_argument("--exe", default=str(pathlib.Path(__file__).resolve().parents[2] / "build" / "maya"))
    ap.add_argument("--max-new", type=int, default=6000)
    ap.add_argument("--runs", type=int, default=1, help="sampled runs per prompt (plus one greedy)")
    ap.add_argument("--no-greedy", action="store_true", help="only the sampled runs (the greedy one ran already)")
    ap.add_argument("--save", default=None, help="a folder for every run's engine output (NAME.RUN.out / .log, written "
                                                 "as the run goes) and its decoded answer (NAME.RUN.txt)")
    ap.add_argument("--only", default=None, help="comma list of prompt names (threejs,canvas,explain,pt-br,maths)")
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--effort", default="high", choices=["low", "high", "max"],
                    help="the template's thinking level (high = the dashboard's default, Medium)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):   # (the tails below may hold characters a Windows code page lacks)
        sys.stdout.reconfigure(errors="replace")
    tok = load_tok(a.tok)
    bad = 0
    only = set(a.only.split(",")) if a.only else None
    for name, q in PROMPTS:
        if only and name not in only:
            continue
        # the model's own template with the thinking level: GLM-5.3's knows "low" and "high" and is "Max" otherwise
        # (the dashboard's Low / Medium / High send low / high / max - serve/frontend.py)
        effort = f"<|system|>Reasoning Effort: {a.effort.capitalize()}" if a.effort else ""
        ids = tok.encode(f"[gMASK]<sop>{effort}<|user|>{q}<|assistant|><think>", add_special_tokens=False).ids
        for run in range(a.runs if a.no_greedy else a.runs + 1):
            greedy = run == a.runs
            args = [a.exe, "--glm-pack", a.pack, "--max-context", str(a.max_context), "--max-new", str(a.max_new),
                    "--eos-ids", ",".join(map(str, sorted(EOS))),
                    "--tokens", ",".join(map(str, ids))]
            args += ["--greedy"] if greedy else ["--seed", str(1234 + run), "--temperature", "1.0", "--top-p", "0.95"]
            t0 = time.time()
            run_kw = dict(cwd=pathlib.Path(a.exe).parents[1], env=dict(os.environ, MAYA_GLM_USAGE="0"))
            if a.save:
                base = pathlib.Path(a.save) / f"{name}.{'greedy' if greedy else f't1-{run}'}"
                base.parent.mkdir(parents=True, exist_ok=True)
                with open(f"{base}.out", "wb") as fo, open(f"{base}.log", "wb") as fe:
                    rc = subprocess.run(args, stdout=fo, stderr=fe, **run_kw).returncode
                stdout, stderr = pathlib.Path(f"{base}.out").read_bytes(), pathlib.Path(f"{base}.log").read_bytes()
            else:
                r = subprocess.run(args, capture_output=True, **run_kw)
                rc, stdout, stderr = r.returncode, r.stdout, r.stderr
            # the engine writes UTF-8 (text=True would read it in the ANSI code page on Windows)
            stdout, stderr = (b.decode("utf-8", "replace").replace("\r\n", "\n") for b in (stdout, stderr))
            out = [int(l.split()[1]) for l in stdout.splitlines() if l.startswith("T ")]
            text = tok.decode(out, skip_special_tokens=False)
            if a.save:
                pathlib.Path(f"{base}.txt").write_text(text, encoding="utf-8")
            finished = bool(out) and (out[-1] in EOS or len(out) < a.max_new)   # stopped at an end token
            closed = THINK_END in text and len(text.split(THINK_END, 1)[1].strip()) > 20
            longest, share = repeats(out)
            looping = longest >= 3 or share >= 0.15
            ok = finished and closed and not looping
            # CAP: still going at --max-new without repeating itself (a long plan, not a loop)
            status = "OK " if ok else "CAP" if not finished and not looping else "BAD"
            bad += status == "BAD"
            print(f"{status} {name:8s} {'greedy' if greedy else f'T=1.0 #{run}':9s} {len(out):5d} tok "
                  f"{time.time() - t0:5.0f} s | finished {finished} | think closed {closed} | longest repeat "
                  f"{longest}x32 | repeated 64-grams {100 * share:4.1f}%", flush=True)
            if not ok:
                print("   tail: " + text[-400:].replace("\n", " | "), flush=True)
            if rc != 0:
                print("   engine: " + stderr[-400:].replace("\n", " | "), flush=True)
    print(f"loop test: {bad} looping answer(s) (CAP: reached --max-new without repeating)")


if __name__ == "__main__":
    main()
