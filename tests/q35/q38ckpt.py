#!/usr/bin/env python3
"""Prefix checkpoints: turn 1 (a long document + a question, thinking on), then turn 2 whose history drops turn 1's
reasoning (as chat clients do) - the prompt diverges inside turn 1's answer, so the engine restores the checkpoint at
turn 1's assistant header instead of reading the document again.  Prints both turns' prompt timing."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import strata_tokenizer as T  # noqa: E402

GGUF = "/media/per/work/llm/lucebox-hub/server/models/Qwen3.8-27B-Q4_K_M.gguf"
tk = T.Tokenizer.from_gguf(GGUF)
doc = " ".join("Section %d: the committee reviewed item %d and approved budget line %d." % (i, i * 7, i * 13)
               for i in range(800))
u1 = "Here is a report:\n" + doc + "\nWhich budget line was approved in section 42? Answer briefly."
p1 = "<|im_start|>user\n" + u1 + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
env = dict(os.environ)
env.update(kv.split("=", 1) for kv in sys.argv[1:])
p = subprocess.Popen(["build/strata-dense", "--serve", "--native", GGUF, "--context", "32768"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
for line in p.stdout:
    if line.startswith(("READY", "ERR")): break

def gen(ids, n):
    p.stdin.write("GEN %d temperature=0 %s\n" % (n, ",".join(map(str, ids)))); p.stdin.flush()
    out = []
    for line in p.stdout:
        if line.startswith("T "): out.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")): return out, line.strip()

ids1 = tk.encode(p1, parse_special=True)
o1, d1 = gen(ids1, 600)
a1 = tk.decode(o1)
answer = a1.split("</think>")[-1].replace("<|im_end|>", "").strip()
print("turn 1:", d1, "| answer:", repr(answer[:120]))
p2 = ("<|im_start|>user\n" + u1 + "<|im_end|>\n<|im_start|>assistant\n" + answer +
      "<|im_end|>\n<|im_start|>user\nAnd in section 43?<|im_end|>\n<|im_start|>assistant\n<think>\n")
o2, d2 = gen(tk.encode(p2, parse_special=True), 400)
print("turn 2:", d2, "| answer:", repr(tk.decode(o2).split("</think>")[-1][:120]))
p.stdin.write("QUIT\n"); p.stdin.flush()
try: p.wait(timeout=30)
except subprocess.TimeoutExpired: p.kill()
print("\n".join(l for l in p.stderr.read().splitlines() if "prompt" in l or "checkpoint" in l))
