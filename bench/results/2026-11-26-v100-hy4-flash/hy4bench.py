# hy4bench.py TAG [rounds]: the served model's decode (TG) on a short code prompt (512 tokens out, fixed - greedy, the
# same tokens every run), its prompt reading (PP) on ~22K tokens behind a nonce (nothing reused) and its decode after
# that prompt (the same prompt again: all of it reused, 256 tokens out); temperature 0.  PP / TG are read from the
# engine's own "maya hy4 serve: prompt ..." lines in its log.  HY4_LOG: the log (maya-hy4-<size>.log), HY4_TEXT: a
# text file of ~60 KB for the long prompt (default: this repo's maya.py), HY4_URL: the server (http://127.0.0.1:8080)
import json, os, re, sys, time, urllib.request, uuid
LOG = os.environ.get("HY4_LOG", "maya-hy4-flash-maya-s.log")
URL = os.environ.get("HY4_URL", "http://127.0.0.1:8080")
tag = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
big = open(os.environ.get("HY4_TEXT", os.path.join(os.path.dirname(__file__), "../../../maya.py"))).read()[:60000]
def ask(msg, n):
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": msg}], "max_tokens": n, "temperature": 0,
            "stream": False}
    off = os.path.getsize(LOG)
    urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read()
    time.sleep(1)
    with open(LOG, errors="replace") as f:
        f.seek(off); t = f.read()
    m = re.findall(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)", t)
    hit = re.findall(r"expert cache ([\d.]+)% hit", t)
    return m[-1], (hit[-1] if hit else "?")
res = []
SHORT = "Write a Python module implementing a thread-safe LRU cache with TTL expiry, with docstrings and unit tests."
BIG = big + "\n\nSummarize what this file does in five bullet points."
for r in range(rounds):
    # TG: fixed prompts, greedy (the same tokens every run); PP: the big prompt behind a nonce (nothing reused); then
    # the same big prompt again (all reused) for TG at 22K
    nonce = f"[{uuid.uuid4().hex[:8]}] "
    for name, msg, n in (("short", SHORT, 512), ("22k-pp", nonce + BIG, 32), ("22k-tg", nonce + BIG, 256)):
        (P, reu, rd, pms, pp, G, gms, tg), hit = ask(msg, n)
        line = (f"{tag} r{r} {name}: prompt {P} (read {rd}) PP {pp} tok/s" if name == "22k-pp" else
                f"{tag} r{r} {name}: TG {tg} tok/s ({G} tok, prompt {P}, read {rd})") + f" | expert hit {hit}%"
        print(line, flush=True); res.append(line)
open("hy4bench.txt", "a").write("\n".join(res) + "\n")
