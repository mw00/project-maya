#!/usr/bin/env python3
"""Gemma 4 / Muse Glimmer check: greedy tokens of strata-dense (--serve line protocol) against llama.cpp's server on the same ids.
The ids come from llama.cpp's /tokenize (strata_tokenizer reads byte-level BPE only).
usage: g4check.py [--gguf G] [--repeat N] [--new N] [--ctx C] [--env K=V ...] [--no-ref]"""
import argparse, json, os, subprocess, time, urllib.request

GGUF = "/media/per/work/llm/Maya-data/models/Gemma-4-E4B/gemma-4-E4B-it-Q4_K_M.gguf"
LLAMA = "/media/per/work/llm/llama.cpp-moe-cache/build-cuda/bin/llama-server"
PROMPT = ("<|turn>user\nWrite a short Python function that returns the n-th Fibonacci number, "
          "then explain how it works in two sentences.<turn|>\n<|turn>model\n")
MUSE = ("<|start|>user<|message|>Write a short Python function that returns the n-th Fibonacci number, "
        "then explain how it works in two sentences.<|eot|><|start|>assistant")
URL = "http://127.0.0.1:8099"


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def eng_tokens(binary, gguf, ids, n_new, ctx, env):
    e = dict(os.environ); e.update(env)
    p = subprocess.Popen([binary, "--serve", "--native", gguf, "--context", str(ctx)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=open("/tmp/g4check.err", "w"), text=True, env=e)
    for line in p.stdout:
        if line.startswith(("READY", "ERR")):
            if line.startswith("ERR"): print(line.strip())
            break
    p.stdin.write(f"GEN {n_new} temperature=0 ignore_eos=1 " + ",".join(map(str, ids)) + "\n"); p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "): toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")):
            if line.startswith("ERR"): print(line.strip())
            break
    p.stdin.write("QUIT\n"); p.stdin.flush()
    try: p.wait(timeout=30)
    except subprocess.TimeoutExpired: p.kill(); p.wait()
    for l in open("/tmp/g4check.err"):
        if "generated" in l or "error" in l.lower() or "gemma4" in l: print(l.rstrip())
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF); ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--new", type=int, default=64); ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--bin", default="build/strata-dense"); ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--style", default="gemma", help="gemma | muse (the chat format of the prompt)")
    a = ap.parse_args()
    text = PROMPT if a.style == "gemma" else MUSE
    if a.repeat > 1:
        if a.style == "gemma":
            text = "<|turn>user\n" + ("Here is some text. " + PROMPT) * a.repeat + "\nSummarize.<turn|>\n<|turn>model\n"
        else:
            text = "<|start|>user<|message|>" + ("Here is some text. " + MUSE) * a.repeat + "\nSummarize.<|eot|><|start|>assistant"
    def start(ngl):
        srv = subprocess.Popen([LLAMA, "-m", a.gguf, "-ngl", str(ngl), "-c", str(a.ctx + a.new + 64), "--port", "8099",
                                "-fa", "on", "--no-warmup", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                urllib.request.urlopen(URL + "/health", timeout=2).read(); break
            except Exception:
                time.sleep(2)
        return srv

    def stop(srv):
        srv.terminate(); srv.wait()

    # the reference on the GPU first (tokens, its answer); then the engine alone on the GPU (a large model and
    # llama.cpp's copy do not both fit); then a CPU server for the text
    srv = start(99 if not a.no_ref else 0)
    try:
        ids = post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
        print(f"prompt tokens: {len(ids)}")
        ref, margins = [], []
        if not a.no_ref:
            r = post("/completion", {"prompt": ids, "n_predict": a.new, "temperature": 0, "top_k": 1, "cache_prompt": False,
                                     "n_probs": 2, "return_tokens": True, "ignore_eos": True})
            ref = r.get("tokens", [])
            for pr in r.get("completion_probabilities", []):
                tp = pr.get("top_logprobs") or pr.get("top_probs") or []
                margins.append(tp[0].get("logprob", 0) - tp[1].get("logprob", 0) if len(tp) > 1 else 99)
            tm = r.get("timings", {})
            print(f"llama.cpp: prompt {tm.get('prompt_per_second', 0):.1f} tok/s, decode {tm.get('predicted_per_second', 0):.1f} tok/s")
    finally:
        stop(srv)
    mine = eng_tokens(a.bin, a.gguf, ids, a.new, a.ctx, dict(kv.split("=", 1) for kv in a.env))
    srv = start(0)
    try:
        print("engine :", repr(post("/detokenize", {"tokens": mine}).get("content", ""))[:int(os.environ.get("G4_SHOW", "400"))])
        if ref:
            print("llama  :", repr(post("/detokenize", {"tokens": ref}).get("content", ""))[:400])
    finally:
        stop(srv)
    if ref:
        same = 0
        for i, (x, y) in enumerate(zip(mine, ref)):
            if x != y:
                print(f"first difference at {i}: engine {x} llama {y} (llama margin {margins[i] if i < len(margins) else 0:.3})")
                break
            same += 1
        print(f"agree: {same}/{min(len(mine), len(ref))} leading tokens")


if __name__ == "__main__":
    main()
