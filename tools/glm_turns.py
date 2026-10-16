"""Agent-turn prompt time: a ~12K-token prompt and its answer, then the same session gets that history + a short user
turn three times (the engine keeps the prefix); prints each later turn's DONE fields (field 4: the prompt ms).

    python tools/glm_turns.py PACK_DIR [STRATA_BINARY]

The engine's environment (STRATA_GLM_*) is passed through, e.g. STRATA_GLM_PREFILL_MIN=32 for the old token-path
threshold."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as T
pack = sys.argv[1]
binary = sys.argv[2] if len(sys.argv) > 2 else "build/strata"
tk = T.Tokenizer.from_dir(os.path.join(pack, "tokenizer"))
p = subprocess.Popen([binary, "--glm-pack", pack, "--serve", "--max-context", "16384"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=open("/tmp/glm_turns.err", "w"), text=True)
for line in p.stdout:
    if line.startswith(("READY", "ERR")): break
def gen(ids, n):
    p.stdin.write("GEN %d temperature=0 %s\n" % (n, ",".join(map(str, ids)))); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")):
            return toks, line.split()
FILL = "The committee met to discuss the river, the market and the old library. " * 900
h = tk.encode("[gMASK]<sop><|user|>" + FILL + "\nSummarize in one line.<|assistant|></think>", parse_special=True)
a, _ = gen(h, 32)
for q in ["Now list three key words.", "Translate the summary to French.", "Write it as a haiku."]:
    h = h + a + tk.encode("<|user|>" + q + "<|assistant|></think>", parse_special=True)
    a, f = gen(h, 32)
    print("turn: %s" % " ".join(f[:8]), flush=True)
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=120)
