#!/usr/bin/env python3
"""Maya's tokenizer (tools/strata_tokenizer.py, from the GGUF) against llama.cpp's /tokenize on a set of texts.
usage: tokcheck.py GGUF"""
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from strata_tokenizer import Tokenizer

LLAMA = "/media/per/work/llm/llama.cpp-moe-cache/build-cuda/bin/llama-server"
TEXTS = [
    "Hello, world! It's a test: don't you think? I'LL see.",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "Numbers 1234567 and 3.14159, 2026-10-14 at 09:48; 100% / 50%.",
    "Ünïcödé café naïve résumé — façade, Tromsø, São Paulo, Kraków.",
    "中文测试：你好，世界！日本語のテキスト。한국어 텍스트.",
    "Emoji 😀🎉👍🏽 and symbols ∑∫√∞ ≠ ≤ ≥ → ←",
    "   leading spaces\ttabs\r\nwindows lines\n\n\nblank lines   trailing   ",
    "MixedCASE camelCase snake_case UPPERCASE PascalCase iPhone McDonald's",
    "<|start|>user<|message|>Hi there<|eot|><|start|>assistant",
    "https://example.com/path?q=1&r=2#frag  user@mail.org  C:\\\\Windows\\\\system32",
    "हिन्दी पाठ और বাংলা লেখা, ภาษาไทย",
    "<|turn>user\nWrite a haiku about rain.<turn|>\n<|turn>model\n",
    "a\n\nb\n\n\n\nc  double  spaces\u00a0nbsp",
]


def main():
    gguf = sys.argv[1]
    tk = Tokenizer.from_gguf(gguf)
    srv = subprocess.Popen([LLAMA, "-m", gguf, "-ngl", "0", "-c", "512", "--port", "8099", "--no-warmup", "-np", "1"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2).read(); break
            except Exception:
                time.sleep(2)
        bad = 0
        for t in TEXTS:
            req = urllib.request.Request("http://127.0.0.1:8099/tokenize",
                                         json.dumps({"content": t, "add_special": False, "parse_special": True}).encode(),
                                         {"Content-Type": "application/json"})
            ref = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
            mine = tk.encode(t, parse_special=True)
            ok = mine == ref and (tk.decode(mine) == t or "<|" in t or "<turn" in t)
            bad += not ok
            print("ok  " if ok else "DIFF", len(ref), repr(t[:50]))
            if not ok:
                print("   llama:", ref[:40]); print("   maya :", mine[:40])
        print("%d / %d texts match" % (len(TEXTS) - bad, len(TEXTS)))
    finally:
        srv.terminate(); srv.wait()


if __name__ == "__main__":
    main()
