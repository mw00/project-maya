"""tools/ik2_check.py - maya-ik2 on RTX 20 (Turing) and two cards: the speed table, and the checks that its opt-in settings
(docs/INKLING2.md: IK2_FLASH_TURING, IK2_PF_GPU1, IK2_PF_PEER, --chunk) do not change what the model says.

One engine start a row.  A row is a layout of the PC's cards plus the settings under test, written over a copy of your
config (maya-inkling-2-mini.json by default; the engine's flags are taken out of it first, so a row is exactly what it
says).  For each row it measures

  speed    prompts of 4K / 32K / 100K tokens (each its own text: no prefix reuse), reading tok/s, and the answers' decode
           tok/s with that much context and with a short one (the server's own `timings`; greedy; drafting off unless asked)
  answers  7 greedy answers (chat, code, translation, arithmetic, a 900-line retrieval prompt): each one's hash
  long     3 retrieval prompts of 3,000 / 12,000 / 40,000 words (7K / 27K / 90K tokens): the first words of the text, asked for

and writes DIR/ROW-KV.json.  `compare` reads two of them: which answers are the same tokens, and where the others part.

  python tools/ik2_check.py list
  python tools/ik2_check.py run one two two-nvlink --what speed,answers,long --out check-out
  python tools/ik2_check.py run two-nvlink --kv fp8 --sizes 4K,32K          # a part of the table
  python tools/ik2_check.py compare check-out/one-f16.json check-out/one-tuned-f16.json
  python tools/ik2_check.py table check-out/*.json                          # the speed rows as a markdown table

Rows (the settings are the ones the PR documents; `list` prints them):
  one, one-tuned           one card (the first of the config's), defaults / IK2_FLASH_TURING=1 --chunk 8192
  two                      two cards (--gpus 2), defaults
  two-nobridge             + IK2_FLASH_TURING=1 IK2_PF_GPU1=1 --chunk 8192, the peer copies off (IK2_PF_HOST=1 IK2_HOST_STAGE=1):
                           what two cards without a bridge do
  two-tuned                the same, the engine choosing its copies (peer only where its load-time test passes)
  two-nvlink               + IK2_PF_PEER=1: peer copies on the source card's stream, checked at load
The tuned rows need an FP16 cache (--kv f16, the default here); with --kv fp8 the flash kernel is off by itself.

It starts maya.py (python tools/ik2_check.py is run from the repo's root, or give --root) and stops it by its own process
group; nothing else is touched.  Port 8080 must be free (--port).  No third-party packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ENGINE_ENV = ("IK2_FLASH_TURING", "IK2_FLASH_TV", "IK2_PF_GPU1", "IK2_PF_PEER", "IK2_PF_HOST", "IK2_HOST_STAGE", "IK2_PEER",
              "IK2_DRAFT_GPU", "IK2_PA70_TURING", "IK2_PROF", "IK2_EP", "IK2_TP", "IK2_TWO_GRAPH")
TUNED = {"IK2_FLASH_TURING": "1"}
ROWS = {
    "one": dict(cards=1, env={}, chunk=None),
    "one-tuned": dict(cards=1, env=dict(TUNED), chunk=8192),
    "two": dict(cards=2, env={}, chunk=None),
    "two-nobridge": dict(cards=2, env={**TUNED, "IK2_PF_GPU1": "1", "IK2_PF_HOST": "1", "IK2_HOST_STAGE": "1"}, chunk=8192),
    "two-tuned": dict(cards=2, env={**TUNED, "IK2_PF_GPU1": "1"}, chunk=8192),
    "two-nvlink": dict(cards=2, env={**TUNED, "IK2_PF_GPU1": "1", "IK2_PF_PEER": "1"}, chunk=8192),
}
SIZES = {"4K": 300, "32K": 2480, "100K": 7750}   # lines of the prompt's text (3,887 / 33,707 / 115,237 tokens)
SHORT = ["Explain how a B-tree index works and when it helps a database query.",
         "Write a Python function that merges two sorted lists, with comments.",
         "Describe the main causes of the French Revolution in detail."]
ANSWERS = ["What is the capital of France, and what is it known for?",
           "Write a Python function that checks whether a string is a palindrome.",
           "Explain in three sentences how a hash table handles collisions.",
           "List five prime numbers greater than 100 and say why each is prime.",
           "Translate to Spanish: 'The weather is nice today, let's go for a walk.'",
           "What is 17 * 23 + 144 / 12? Show the steps.",
           "Here is a list:\n" + " ".join(f"line {i}: item {i * 37} is catalogued under code {i * 11}." for i in range(900))
           + "\nHow many lines mention an item whose number ends in 7? Answer briefly."]
WORDS = "alpha river stone engine garden window copper violet signal harbor planet ladder pocket meadow thunder".split()


class Server:
    def __init__(self, root: pathlib.Path, cfg: dict, out: pathlib.Path, tag: str, port: int):
        self.port, self.root = port, root
        if self.up(1):
            sys.exit(f"port {port} already answers: stop that server first (or use --port)")
        cfg = dict(cfg)
        cfg["log"] = str(out / f"{tag}.log")
        self.cfg_path = out / f"{tag}.config.json"
        self.cfg_path.write_text(json.dumps(cfg, indent=1))
        self.out_f = open(out / f"{tag}.out", "w")
        self.p = subprocess.Popen([sys.executable, "maya.py", "--config", str(self.cfg_path), "--port", str(port), "--plain"],
                                  cwd=root, stdout=self.out_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)
        t0 = time.time()
        while not self.up(5):
            if self.p.poll() is not None or time.time() - t0 > 900:
                self.stop()
                sys.exit(f"the server did not come up: see {out / (tag + '.out')}")
            time.sleep(2)

    def up(self, timeout) -> bool:
        try:
            d = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=timeout).read())
            return any((m.get("status") or {}).get("value", "loaded") == "loaded" for m in d.get("data", []))
        except Exception:
            return False

    def chat(self, text: str, max_tokens: int) -> dict:
        body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers={"content-type": "application/json"})
        d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
        return {"text": d["choices"][0]["message"].get("content") or "", "timings": d.get("timings") or {}}

    def stop(self):
        if self.p.poll() is None:
            try:
                self.p.send_signal(signal.SIGTERM)
                self.p.wait(timeout=180)
            except Exception:
                pass
        try:   # (whatever of its process group is left: the launcher's, the engine's)
            os.killpg(self.p.pid, signal.SIGKILL)
        except Exception:
            pass
        self.p.wait()
        self.out_f.close()


def make_config(base: dict, row: dict, kv: str, drafting: str) -> dict:
    c = json.loads(json.dumps(base))
    a = c["args"]
    for flag in ("--chunk", "--gpus"):
        while flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    if "--drafting" in a:
        a[a.index("--drafting") + 1] = drafting
    elif "--draft" in a:
        a += ["--drafting", drafting]
    env = {k: v for k, v in (c.get("env") or {}).items() if k not in ENGINE_ENV}
    env["IK2_KV"] = kv
    env.update(row["env"])
    c["env"] = env
    gpus = c.get("gpu")
    gpus = gpus if isinstance(gpus, list) else [gpus or 0]
    if row["cards"] == 2:
        if len(gpus) < 2:
            sys.exit("two cards asked for, the config names one in \"gpu\"")
        c["gpu"], a[:] = gpus[:2], a + ["--gpus", "2"]
    else:
        c["gpu"] = gpus[:1]
    if row["chunk"]:
        a += ["--chunk", str(row["chunk"])]
    return c


def doc(tag: str, lines: int) -> str:
    return " ".join(f"{tag} line {i}: value {i * 31} ends." for i in range(lines))


def measure_speed(s: Server, sizes: list[str]) -> dict:
    s.chat("Say hi.", 8)
    short = [s.chat(q, 200)["timings"]["predicted_per_second"] for q in SHORT for _ in range(3)]
    r = {"decode_short": round(statistics.median(short), 1)}
    for label in sizes:
        t = s.chat("Summarise in detail: " + doc(f"M{label}", SIZES[label]), 128)["timings"]
        r[f"tokens_{label}"], r[f"read_{label}"] = t["prompt_n"], round(t["prompt_per_second"])
        r[f"decode_{label}"] = round(t["predicted_per_second"], 1)
        print(f"  {label}: {t['prompt_n']} tokens read at {r[f'read_{label}']} tok/s, answers at {r[f'decode_{label}']} tok/s", flush=True)
    return r


def measure_answers(s: Server) -> list:
    out = []
    for q in ANSWERS:
        t = s.chat(q, 90)["text"]
        out.append({"hash": hashlib.md5(t.encode()).hexdigest()[:8], "text": t})
    print("  answers:", [o["hash"] for o in out], flush=True)
    return out


def measure_long(s: Server) -> list:
    out = []
    for n in (3000, 12000, 40000):
        r = random.Random(n)
        text = " ".join(r.choice(WORDS) + str(r.randint(0, 999)) for _ in range(n))
        d = s.chat(text + "\n\nWhat was the very first word of the text above, and list the first five words.", 60)
        out.append({"words": n, "tokens": d["timings"].get("prompt_n"), "hash": hashlib.sha1(d["text"].encode()).hexdigest()[:8],
                    "text": d["text"], "first_word": text.split()[0]})
        print(f"  long {n} words ({out[-1]['tokens']} tokens): {d['text'][:70]!r}", flush=True)
    return out


def cmd_run(a):
    root = pathlib.Path(a.root).resolve()
    base = json.loads((root / a.base).read_text() if not os.path.isabs(a.base) else pathlib.Path(a.base).read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    what = set(a.what.split(","))
    sizes = a.sizes.split(",")
    for name in a.rows:
        if name not in ROWS:
            sys.exit(f"unknown row {name!r}: {', '.join(ROWS)}")
        tag = f"{name}-{a.kv}"
        print(f"== {tag}", flush=True)
        s = Server(root, make_config(base, ROWS[name], a.kv, a.drafting), out, tag, a.port)
        try:
            res = {"row": name, "kv": a.kv, "drafting": a.drafting, "settings": ROWS[name],
                   "vram_mib": subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                              capture_output=True, text=True).stdout.split()}
            if "speed" in what:
                res["speed"] = measure_speed(s, sizes)
            if "answers" in what:
                res["answers"] = measure_answers(s)
            if "long" in what:
                res["long"] = measure_long(s)
        finally:
            s.stop()
        (out / f"{tag}.json").write_text(json.dumps(res, indent=1))


def cmd_compare(a):
    A, B = (json.loads(pathlib.Path(p).read_text()) for p in (a.a, a.b))
    print(f"{A['row']}-{A['kv']}  against  {B['row']}-{B['kv']}")
    for key, label in (("answers", "short answers"), ("long", "long prompts")):
        if key not in A or key not in B:
            continue
        same = 0
        for i, (x, y) in enumerate(zip(A[key], B[key])):
            if x["hash"] == y["hash"]:
                same += 1
                continue
            at = next((j for j, (c, d) in enumerate(zip(x["text"], y["text"])) if c != d), min(len(x["text"]), len(y["text"])))
            extra = ""
            if key == "long":   # (the retrieval itself: the text's first word in both)
                extra = f"; first word kept in both: {x['first_word'] in x['text'][:80] and y['first_word'] in y['text'][:80]}"
            print(f"  {label} #{i + 1}: differs at character {at} of {len(x['text'])}/{len(y['text'])}{extra}")
        print(f"  {label}: {same} of {len(A[key])} the same tokens")


def cmd_table(a):
    print("| Row | KV | Prompts 4K / 32K / 100K (tok/s) | Answers short / 32K / 100K (tok/s) |")
    print("| --- | --- | ---: | ---: |")
    for p in a.files:
        d = json.loads(pathlib.Path(p).read_text())
        s = d.get("speed")
        if not s:
            continue
        g = lambda k: s.get(k, "-")
        print(f"| {d['row']} | {d['kv']} | {g('read_4K')} / {g('read_32K')} / {g('read_100K')} | "
              f"{g('decode_short')} / {g('decode_32K')} / {g('decode_100K')} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("rows", nargs="+")
    r.add_argument("--base", default="maya-inkling-2-mini.json", help="the config the rows are written over")
    r.add_argument("--root", default=".", help="the repo's root (maya.py)")
    r.add_argument("--out", default="ik2-check-out")
    r.add_argument("--what", default="speed,answers,long")
    r.add_argument("--sizes", default="4K,32K,100K")
    r.add_argument("--kv", default="f16", choices=("f16", "fp8"))
    r.add_argument("--drafting", default="off", choices=("off", "auto", "on"))
    r.add_argument("--port", type=int, default=8080)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in ROWS.items():
            print(f"{k:13s} cards {v['cards']}  env {' '.join(f'{x}={y}' for x, y in v['env'].items()) or '-'}  "
                  f"{'--chunk ' + str(v['chunk']) if v['chunk'] else ''}")
    else:
        {"run": cmd_run, "compare": cmd_compare, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
