#!/usr/bin/env python3
"""Qwen3.8-27B check: greedy tokens of an engine binary (the strata --serve line protocol) against llama.cpp's
server on the same token ids.  usage: q38check.py [--prompt-file F] [--repeat N] [--new N] [--bin B] [--ctx C]
[--env K=V ...] [--no-ref]"""
import argparse, json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from strata_tokenizer import Tokenizer

GGUF = "/media/per/work/llm/lucebox-hub/server/models/Qwen3.8-27B-Q4_K_M.gguf"
LLAMA = "/media/per/work/llm/llama.cpp-moe-cache/build-cuda/bin/llama-server"
DEFAULT_PROMPT = ("<|im_start|>user\nWrite a short Python function that returns the n-th Fibonacci number, "
                  "then explain how it works in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

def ref_tokens(ids, n_new, ctx):
    p = subprocess.Popen([LLAMA, "-m", GGUF, "-ngl", "99", "-c", str(max(ctx, len(ids) + n_new + 16)), "--port", "8099",
                          "-fa", "on", "--no-warmup", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2).read(); break
            except Exception:
                time.sleep(2)
        body = json.dumps({"prompt": ids, "n_predict": n_new, "temperature": 0, "top_k": 1, "cache_prompt": False,
                           "n_probs": 2, "return_tokens": True, "ignore_eos": True}).encode()
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099/completion", body,
                       {"Content-Type": "application/json"}), timeout=3600).read())
        toks, margins = r.get("tokens", []), []
        for pr in r.get("completion_probabilities", []):
            tp = pr.get("top_logprobs") or pr.get("top_probs") or []
            margins.append(tp[0].get("logprob", 0) - tp[1].get("logprob", 0) if len(tp) > 1 else 99)
        tm = r.get("timings", {})
        print(f"llama.cpp: prompt {tm.get('prompt_per_second', 0):.1f} tok/s, decode {tm.get('predicted_per_second', 0):.1f} tok/s")
        return toks, margins
    finally:
        p.terminate(); p.wait()

def eng_tokens(binary, ids, n_new, ctx, env, extra, model=None):
    e = dict(os.environ); e.update(env)
    p = subprocess.Popen([binary, "--serve", "--native", model or GGUF, "--context", str(ctx)] + extra, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=sys.stderr, text=True, env=e)
    for line in p.stdout:
        if line.startswith("READY") or line.startswith("ERR"):
            print(line.strip()); break
    samp = os.environ.get("Q38_SAMPLING", "temperature=0")
    p.stdin.write(f"GEN {n_new} {samp} ignore_eos=1 " + ",".join(map(str, ids)) + "\n"); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            f = line.split(); gen, n, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
            print(f"engine: prompt {n} tokens {pms:.0f} ms = {1000*n/max(pms,1):.1f} tok/s, decode {gen} in {dms:.0f} ms "
                  f"= {1000*max(gen-1,1)/max(dms,1):.1f} tok/s")
            break
        elif line.startswith("ERR"): print(line.strip()); break
    p.stdin.write("QUIT\n"); p.stdin.flush()
    try: p.wait(timeout=30)
    except subprocess.TimeoutExpired: p.kill(); p.wait()
    return toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file"); ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--new", type=int, default=64); ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--bin", default="build/strata-dense"); ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--arg", action="append", default=[]); ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--model", help="the engine's model (default --gguf): e.g. an EXL3 checkpoint directory; the tokenizer stays --gguf's")
    a = ap.parse_args()
    globals()["GGUF"] = a.gguf
    tok = Tokenizer.from_gguf(GGUF)
    text = open(a.prompt_file).read() if a.prompt_file else DEFAULT_PROMPT
    if a.repeat > 1:
        text = "<|im_start|>user\n" + ("Here is some text. " + text) * a.repeat + "\nSummarize.<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, parse_special=True)
    print(f"prompt tokens: {len(ids)}")
    env = dict(kv.split("=", 1) for kv in a.env)
    mine = eng_tokens(a.bin, ids, a.new, a.ctx, env, a.arg, a.model)
    print("engine :", repr(tok.decode(mine))[:400])
    import hashlib
    print("engine ids:", len(mine), hashlib.sha1(",".join(map(str, mine)).encode()).hexdigest()[:16])
    if a.no_ref: return
    ref, margins = ref_tokens(ids, a.new, a.ctx)
    print("llama  :", repr(tok.decode(ref))[:400])
    same = 0
    for i, (x, y) in enumerate(zip(mine, ref)):
        if x != y:
            print(f"first difference at {i}: engine {x} llama {y} (llama margin {margins[i] if i < len(margins) else '?':.3})")
            break
        same += 1
    print(f"agree: {same}/{min(len(mine), len(ref))} leading tokens")

if __name__ == "__main__":
    main()
