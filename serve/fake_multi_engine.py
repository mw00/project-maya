#!/usr/bin/env python3
"""serve/fake_multi_engine.py - a stand-in for `strata --serve` with several conversations (READY ... multi=<n>), for
serve/test_multi.py: GEN ... rid=<id> <ids> answers with the prompt's ids (each + 1000) one at a time, the requests
interleaved token by token as the pipelined decode interleaves them; STOP <id> ends one; QUIT ends.

    serve/fake_multi_engine.py --serve <n>        (StrataEngine starts it as it starts `strata`)
"""
from __future__ import annotations

import sys
import threading
import time

n = int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 2
lock = threading.Lock()
active: dict[int, dict] = {}
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
                active[rid] = {"left": [i + 1000 for i in ids][: int(f[1])], "n": 0}
            out(f"PP {rid} {len(ids)} {len(ids)} 1.0 1000.0")
    quit_.set()


threading.Thread(target=reader, daemon=True).start()
out("INFO engine=fake-multi")
out(f"READY 4096 stop pause multi={n}")
while not quit_.is_set():
    with lock:
        rids = list(active)
    if not rids:
        time.sleep(0.002)
        continue
    for rid in rids:                                  # one token of each, in turn
        with lock:
            r = active.get(rid)
            if r is None:
                continue
            stopped = rid in stops
            stops.discard(rid)
            if stopped or not r["left"]:
                out(f"DONE {rid} {r['n']} 1 0.0 1.0 {'cancel' if stopped else 'length'} 0 0 0")
                del active[rid]
                continue
            t = r["left"].pop(0)
            r["n"] += 1
        out(f"T {rid} {t}")
        time.sleep(0.003)
