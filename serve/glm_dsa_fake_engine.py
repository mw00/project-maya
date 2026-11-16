#!/usr/bin/env python3
"""serve/glm_dsa_fake_engine.py - a stand-in for glm-dsa's `strata --serve --glm-pack ...` (glm-dsa) for the serve
tests (serve/test_glm_dsa_serve.py): no GPU, no model.  It speaks the wire protocol the server relies on and that
serve/test_engine_contract.py checks against the real engine:

  stderr (the engine log): "maya generate: session is up", "glm fast: CUDA0 VRAM before the expert pool: ...
          state/KV (<ctx> ctx) ..." (the server's context-cost reader)
  stdout: INFO <the GLM keys, ram_gb included>, READY <ctx> stop pause, then per GEN line T <id> ..., STAT every 8
          tokens, DONE with 10 fields (reused = the prompt's common prefix with the conversation it holds)
  stdin:  GEN <max_new> [key=value ...] <ids> (think_budget / think_end close the thinking as the engine does),
          STOP, WRAP, PAUSE [keep], QUIT
The INFO and STAT lines carry the keys the real engine prints on GLM-5.3 (src/program/generate.cpp's serve start and
its decode loop: the same lines GLM-5.3-Flash's engine prints; checked against a real GLM-5.3 run, 2026-11-07).

Its answer is FAKE_ENGINE_IDS (comma-separated ids; default the bytes of "hello"), repeated up to max_new.
FAKE_ENGINE_SCRIPT=<file> (a JSON list of answers, each a list of ids; read again at every GEN line): the k-th GEN
line since the file last changed gets the k-th answer, the last one repeating - a test rewrites the file before each
request (a tool call, then the answer after its result) and the engine keeps running.
QUIT ends it with exit code 0; with MAYA_GLM_SLOT_KEEP=1 it first "keeps" the conversation it holds: it writes
<FAKE_ENGINE_DIR>/kept-<pid>.json and says so in a NOTE line, as the real engine writes its slot file.
FAKE_ENGINE_IGNORE_QUIT=1 makes it ignore QUIT (a hung engine: the server has to kill it).
FAKE_ENGINE_GEN_LOG=<file> appends every GEN line it reads to <file> (the per-request knobs' forwarding test).
"""
from __future__ import annotations

import json
import os
import sys as _sys   # (an older STRATA_X counts as MAYA_X: tools/maya_names.py makes it so on import)
_sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tools')))
import maya_names  # noqa: E402,F401
import queue
import sys
import threading
import time

args = sys.argv[1:]
ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
ram_gb = os.environ.get("FAKE_ENGINE_RAM_GB", "170.0")
answer = [int(x) for x in os.environ.get("FAKE_ENGINE_IDS", ",".join(str(b) for b in b"hello")).split(",") if x]
script = {"text": None, "turn": 0}               # FAKE_ENGINE_SCRIPT: the file's last text and the GEN lines since
ignore_quit = os.environ.get("FAKE_ENGINE_IGNORE_QUIT") == "1"
keep = os.environ.get("MAYA_GLM_SLOT_KEEP") == "1"
held: list[int] = []                              # the conversation in the "model": prompt + what it wrote
controls: queue.Queue = queue.Queue()
lines: queue.Queue = queue.Queue()


def out(s: str):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def log(s: str):
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


def reader():
    for line in sys.stdin:
        line = line.strip()
        if line in ("STOP", "WRAP", "PAUSE", "PAUSE keep"):
            controls.put(line)
        elif line == "QUIT":
            if not ignore_quit:
                controls.put(line)
                lines.put(line)
        elif line:
            lines.put(line)
    lines.put(None)


def quit_now():
    if keep and held:
        d = os.environ.get("FAKE_ENGINE_DIR") or "."
        with open(os.path.join(d, f"kept-{os.getpid()}.json"), "w", encoding="utf-8") as f:
            json.dump({"tokens": len(held), "context": ctx}, f)
        out(f"NOTE sessions: kept the conversation ({len(held)} tokens)")
    sys.exit(0)


def scripted() -> list[int]:
    """This GEN line's answer: FAKE_ENGINE_SCRIPT's next one when the variable names a file, else FAKE_ENGINE_IDS."""
    path = os.environ.get("FAKE_ENGINE_SCRIPT")
    if not path:
        return answer
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text != script["text"]:
        script.update(text=text, turn=0)
    answers = json.loads(text) or [[]]
    k = min(script["turn"], len(answers) - 1)
    script["turn"] += 1
    return [int(t) for t in answers[k]]


def generate(line: str):
    global held
    answer = scripted()
    if os.environ.get("FAKE_ENGINE_GEN_LOG"):
        with open(os.environ["FAKE_ENGINE_GEN_LOG"], "a", encoding="utf-8") as gen_log:
            gen_log.write(line + "\n")
    f = line.split()
    max_new = int(f[1])
    keys = dict(x.split("=", 1) for x in f[2:-1] if "=" in x)
    think_budget, think_end = int(keys.get("think_budget", 0)), int(keys.get("think_end", -1))
    in_think = think_budget > 0 and think_end >= 0
    ids = [int(x) for x in f[-1].split(",") if x]
    reused = 0
    for a, b in zip(ids, held):
        if a != b:
            break
        reused += 1
    held = list(ids)
    t0 = time.time()
    out(f"PP {len(ids)} {len(ids)} 1.0 1000.0")
    n, p, finish = 0, 0, "length"                 # p: where in the answer (n, unless a closed thinking skipped on)
    while n < max_new:
        try:
            c = controls.get_nowait()
        except queue.Empty:
            c = None
        if c == "QUIT":
            finish = "cancel"
            break
        if c in ("STOP", "PAUSE", "PAUSE keep"):
            finish = "cancel"
            break
        tok = answer[p % len(answer)] if answer else 0
        p += 1
        if in_think and (c == "WRAP" or n >= think_budget):   # the budget or WRAP: the next token closes the thinking
            tok, in_think = think_end, False
            if p - 1 < len(answer) and think_end in answer[p - 1:]:   # the answer goes on after its own </think>,
                p = answer.index(think_end, p - 1) + 1             # as a model's does once its thinking is closed
        elif in_think and tok == think_end:
            in_think = False
        out(f"T {tok}")
        held.append(tok)
        n += 1
        if n % 8 == 0:
            out(f"STAT tok_s=100.00 ms_tok=10.00 vram_hit=0.6000 ram_fetch=1.000 disk=0.000 promo=0.000 "
                f"vram_used=10 vram_slots=10 ram_used=20 ram_slots=20 ram_gb={ram_gb} vram_gb=60.0 disk_gb=0.0 "
                "draft_acc=0 draft_off=0")
        if answer and p >= len(answer) and max_new > len(answer) and os.environ.get("FAKE_ENGINE_REPEAT") != "1":
            finish = "stop"
            break
        time.sleep(0.001)
    ms = 1000.0 * (time.time() - t0)
    out(f"DONE {n} {len(ids)} 1.0 {ms:.1f} {finish} 0 0 {reused} {n} {n}")
    log(f"glm stat: decode {ms / max(1, n):.2f} ms/tok ({1000.0 * n / max(ms, 1e-3):.2f} tok/s) | vram hit 60.00%")


log("glm fast: CUDA0 VRAM before the expert pool: 12.00 of 32.00 GB in use - dense weights 8.00, "
    f"state/KV ({ctx} ctx) 1.50, activations 0.50, other (context, MTP, scratch, prompt path) 2.00")
log("maya generate: session is up (engine fake-glm-dsa)")
threading.Thread(target=reader, daemon=True).start()
out(f"INFO context={ctx} kv=int8 kv_resident=0 expert_slots=100 expert_cache_mib=61440 engine_kind=glm-fast "
    f"experts=19200 vram_slots=100 vram_gb=60.0 ram_slots=400 ram_gb={ram_gb} mtp=1 dflash=0 vision_lend=0 "
    "cpu_threads=36 cpu_threads_max=36 pcie_share=0.50 logit_bias=1")
out(f"READY {ctx} stop pause")
log("maya generate: glm-dsa serve (fake): GEN <max_new> [keys] <ids>, STOP, WRAP, QUIT")
while True:
    line = lines.get()
    if line is None:
        break
    if line == "QUIT":
        quit_now()
    if line.startswith(("GEN ", "GENI ")):
        while not controls.empty():                 # a stray STOP between requests
            if controls.get_nowait() == "QUIT":
                quit_now()
        generate(line)
    elif line in ("VLEND", "VRECLAIM"):
        out(f"ERR {line} text-only model: no vision encoder to lend memory to")
    else:
        out(f"ERR unsupported command '{line.split()[0]}'")
while ignore_quit:                                  # a hung engine does not end when its input closes either
    time.sleep(1)
