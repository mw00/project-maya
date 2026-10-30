#!/usr/bin/env python3
"""serve/fake_dspark_engine.py - a stand-in for `maya-ds4 --serve` that reports drafts by source, for
serve/test_dspark_dp.py: every request answers a few tokens, then its DP split and DONE.

Single stream (READY 4096 stop): odd-numbered requests emit `DP kept=6 offered=10 ds_kept=4 ds_offered=7 lk_kept=2
lk_offered=3 rounds=1 pos=1:4/7,2:2/3` before the DONE (kept = DSpark + lookup, as the engine reports); even-numbered
ones emit no DP (an engine without the split), their DONE carrying 3 of 5.

Multi (READY ... multi=<n>): `DP <rid> ...` with values by rid - kept=rid, offered=2*rid, ds_kept=rid-1,
ds_offered=rid, lk_kept=1, lk_offered=rid, rounds=rid (no pos=, like the engine's multi path).

    serve/fake_dspark_engine.py --serve [n]        (StrataEngine starts it as it starts `maya-ds4`)
"""
from __future__ import annotations

import sys
import threading

n = int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 0
lock = threading.Lock()
active: dict[int, list] = {}
requests = 0
quit_ = threading.Event()


def out(s: str):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def reader():
    global requests
    for line in sys.stdin:
        f = line.split()
        if not f:
            continue
        if f[0] == "QUIT":
            quit_.set()
            return
        if f[0] == "GEN":
            with lock:
                requests += 1
                if n:
                    keys = dict(x.split("=", 1) for x in f[2:-1] if "=" in x)
                    rid = int(keys["rid"])
                    active[rid] = [120, 257]
                    out(f"PP {rid} 2 2 1.0 1000.0")
                else:
                    active[0] = [120, 257]
                    out("PP 2 2 1.0 1000.0")
    quit_.set()


def dp_single():
    if requests % 2 == 1:
        out("DP kept=6 offered=10 ds_kept=4 ds_offered=7 lk_kept=2 lk_offered=3 rounds=1 pos=1:4/7,2:2/3")
        out("DONE 2 2 1.0 2.0 stop 6 10 0 0 0")
    else:
        out("DONE 2 2 1.0 2.0 stop 3 5 0 0 0")


threading.Thread(target=reader, daemon=True).start()
out("INFO engine=fake-dspark")
out("INFO logprobs=20")
out(f"READY 4096 stop pause multi={n}" if n else "READY 4096 stop")
while not quit_.is_set():
    with lock:
        rids = list(active)
    if not rids:
        quit_.wait(0.002)
        continue
    for rid in rids:
        with lock:
            left = active.get(rid)
            if left is None:
                continue
            if left:
                t = left.pop(0)
                done = False
            else:
                done = True
                del active[rid]
        if done:
            if n:
                out(f"DP {rid} kept={rid} offered={2 * rid} ds_kept={rid - 1} ds_offered={rid} lk_kept=1 "
                    f"lk_offered={rid} rounds={rid}")
                out(f"DONE {rid} 2 2 1.0 2.0 stop {rid} {2 * rid} 0 0 0")
            else:
                dp_single()
        else:
            out(f"T {rid} {t}" if n else f"T {t}")
        threading.Event().wait(0.002)

