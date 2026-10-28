#!/usr/bin/env python3
"""serve/fake_multi_engine.py - a stand-in for `strata --serve` with several conversations (READY ... multi=<n>), for
serve/test_multi.py: GEN ... rid=<id> <ids> answers with the prompt's ids (each + 1000) one at a time, the requests
interleaved token by token as the pipelined decode interleaves them; STOP <id> ends one; QUIT ends.
FAKE_MULTI_INTERLEAVE=<n> (STRATA_GLM_PREFILL_INTERLEAVE's line pattern): a prompt is read n ids a piece, one PP line
each, while the requests already running decode a token each between two pieces; one prompt is read at a time (later
ones queue), and a STOP for the one being read ends it with no token.  A GEN with logprobs=<n> gets an LP line after
each T line: logprob -0.25 * (the token's place % 4), then the token and n - 1 others (id + 1, + 2, ...) each 1 lower.

    serve/fake_multi_engine.py --serve <n>        (StrataEngine starts it as it starts `strata`)
"""
from __future__ import annotations

import os
import sys
import threading
import time

n = int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 2
piece = int(os.environ.get("FAKE_MULTI_INTERLEAVE", "0") or 0)
lock = threading.Lock()
active: dict[int, dict] = {}
reading: list[int] = []                               # FAKE_MULTI_INTERLEAVE: prompts to read, the first being read
stops: set[int] = set()
quit_ = threading.Event()


def out(s: str):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def reader():
    for line in sys.stdin:
        f = line.split()
        if not f:
            continue
        if f[0] == "QUIT":
            quit_.set()
            return
        if f[0] == "STOP" and len(f) > 1:
            with lock:
                stops.add(int(f[1]))
            continue
        if f[0] == "GEN":
            keys = dict(x.split("=", 1) for x in f[2:-1] if "=" in x)
            rid = int(keys["rid"])
            ids = [int(x) for x in f[-1].split(",")]
            with lock:
                if len(active) >= n:
                    out(f"ERR {rid} all {n} conversation states are busy")
                    continue
                active[rid] = {"left": [i + 1000 for i in ids][: int(f[1])], "n": 0, "read": 0, "total": len(ids),
                               "lp": int(keys["logprobs"]) if "logprobs" in keys else -1}
                if piece > 0:
                    reading.append(rid)
                    continue
            out(f"PP {rid} {len(ids)} {len(ids)} 1.0 1000.0")
    quit_.set()


def read_piece():
    """One piece of the first queued prompt (FAKE_MULTI_INTERLEAVE); True when one was read."""
    with lock:
        if not reading:
            return False
        rid = reading[0]
        r = active[rid]
        if rid in stops:                              # its STOP while it was read: no token
            stops.discard(rid)
            reading.pop(0)
            del active[rid]
            out(f"DONE {rid} 0 {r['total']} 1.0 0.0 cancel 0 0 0")
            return True
        r["read"] = min(r["total"], r["read"] + piece)
        out(f"PP {rid} {r['read']} {r['total']} 1.0 1000.0")
        if r["read"] >= r["total"]:
            reading.pop(0)
    time.sleep(0.003)
    return True


threading.Thread(target=reader, daemon=True).start()
out("INFO engine=fake-multi")
out("INFO logprobs=20")
out(f"READY 4096 stop pause multi={n}")
while not quit_.is_set():
    read_piece()
    with lock:
        rids = [r for r in active if r not in reading]
    if not rids:
        time.sleep(0.002)
        continue
    for rid in rids:                                  # one token of each, in turn
        with lock:
            r = active.get(rid)
            if r is None or rid in reading:
                continue
            stopped = rid in stops
            stops.discard(rid)
            if stopped or not r["left"]:
                out(f"DONE {rid} {r['n']} {r['total']} 0.0 1.0 {'cancel' if stopped else 'length'} 0 0 0")   # (prompt: its own)
                del active[rid]
                continue
            t = r["left"].pop(0)
            r["n"] += 1
            k, own = r["lp"], -0.25 * ((r["n"] - 1) % 4)
        out(f"T {rid} {t}")
        if k >= 0:
            out(f"LP {rid} {t} {own:g}" + "".join(f" {t + j}:{own - j:g}" for j in range(k)))
        time.sleep(0.003)
