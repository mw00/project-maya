#!/usr/bin/env python3
"""Long-context needle test for strata-dense (Qwen3.8-27B): filler text with three facts at 10% / 50% / 85% depth,
then a question about all three (thinking off); reports the prompt and decode speed.

    python q38needle.py TOKENS [--ctx N] [--env K=V ...] [--arg ARG ...]
"""
import argparse
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import strata_tokenizer as T  # noqa: E402

GGUF = "/media/per/work/llm/lucebox-hub/server/models/Qwen3.8-27B-Q4_K_M.gguf"
ap = argparse.ArgumentParser()
ap.add_argument("tokens", type=int)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--arg", action="append", default=[])
ap.add_argument("--ctx", type=int, default=0)
ap.add_argument("--new", type=int, default=80)
ap.add_argument("--bin", default="build/strata-dense")
ap.add_argument("--gguf", default=GGUF)
ap.add_argument("--style", default="qwen", help="qwen | gemma | muse (the chat format)")
a = ap.parse_args()
GGUF = a.gguf

tk = T.Tokenizer.from_gguf(GGUF)
rnd = random.Random(7)
subjects = ["The river", "A merchant", "The old library", "Our committee", "The northern road", "A young engineer",
            "The harbour master", "The orchestra", "A small bakery", "The weather station"]
verbs = ["described", "measured", "ignored", "rebuilt", "celebrated", "documented", "questioned", "repaired"]
objects = ["the eastern bridge", "a set of ledgers", "the spring festival", "three copper kettles", "the town archive",
           "a broken telescope", "the market square", "an unusual map", "the winter harvest", "a forgotten song"]
facts = ["The password for the blue vault is MARIGOLD-4471.", "Captain Ilse Varga was born in the city of Tromsø.",
         "The experiment's final reading was 93.62 kelvin."]


def sentence():
    return "%s %s %s in the year %d. " % (rnd.choice(subjects), rnd.choice(verbs), rnd.choice(objects),
                                          rnd.randint(1200, 1999))


chunks, n = [], 0
while n < a.tokens:
    s = sentence()
    chunks.append(s)
    n += len(tk.encode(s))
for depth, f in zip((0.10, 0.50, 0.85), facts):
    chunks.insert(int(len(chunks) * depth), f + " ")
q = ("".join(chunks) + "\n\nUsing only the text above, answer three questions in one line each: 1) the password for "
     "the blue vault, 2) the city where Captain Ilse Varga was born, 3) the experiment's final reading.")
if a.style == "gemma":
    ids = tk.encode("<bos><|turn>user\n" + q + "<turn|>\n<|turn>model\n", parse_special=True)
elif a.style == "muse":
    ids = tk.encode("<|begin_of_text|><|start|>user<|message|>" + q + "<|eot|><|start|>assistant", parse_special=True)
else:
    ids = tk.encode("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    parse_special=True)
env = dict(os.environ)
env.update(kv.split("=", 1) for kv in a.env)
ctx = a.ctx or (len(ids) + a.new + 64)
errf = open("/tmp/q38needle.err", "w+")   # (a file: a pipe nobody reads fills and stops the engine)
p = subprocess.Popen([a.bin, "--serve", "--native", GGUF, "--context", str(ctx)] + a.arg, stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=errf, text=True, env=env)
for line in p.stdout:
    if line.startswith(("READY", "ERR")):
        break
t0 = time.time()
p.stdin.write("GEN %d temperature=0 %s\n" % (a.new, ",".join(map(str, ids))))
p.stdin.flush()
gen, done = [], ""
for line in p.stdout:
    if line.startswith("T "):
        gen.append(int(line.split()[1]))
    elif line.startswith(("DONE", "ERR")):
        done = line.strip()
        break
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=30)
except subprocess.TimeoutExpired:
    p.kill()
errf.seek(0)
err = errf.read()
ans = tk.decode(gen)
ok = [k in ans for k in ("MARIGOLD-4471", "Troms", "93.62")]
f = done.split()
if len(f) >= 5 and f[0] == "DONE":
    g, n_p, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
    print("%d prompt tokens: prompt %.0f tok/s (%.1f s), decode %d tokens %.1f tok/s" %
          (n_p, 1000 * n_p / pms, pms / 1000, g, 1000 * max(g - 1, 1) / max(dms, 1)))
else:
    print("no DONE: %r" % done)
print([l for l in err.splitlines() if "rounds" in l or "ERR" in l or "rror" in l][-2:])
print("answer: %r" % ans[:300])
print("needles found: %d / 3 %s" % (sum(ok), ok))
