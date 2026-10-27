"""Agents on `--parallel N`, the way a tool-using harness drives the engine: each agent a conversation of turns - a
system prompt and task first, then the conversation so far with the model's last answer and a tool's result - a tool's
time between an answer and the next turn, every agent at once.  The same seeded workload on each engine given, and
per turn when its first token and its last came from its send.

    python tools/ds4/agent_bench.py --pack <pack> --exe build/maya-ds4 [--exe <another build>] [--gpus 0,2,1]
                                    [--config maya-ds4.json] [--agents 4] [--turns 8] [--seed 1]

  turn 1      the system prompt and task: 4,000-12,000 tokens
  turn 2..    a tool's result: 150-2,000 tokens nine times in ten, 5,000-20,000 (a file read) one in ten
  answers     64-400 tokens (a tool call, a short reply)
  tool time   0.5-5 s between an answer and the next turn

`--serial` runs the engines at --parallel 1 instead - the single path, the DSpark drafter drafting (spec=1) - with the
agents' turns one at a time, as a harness queues them for one conversation; a turn's wait in that queue counts.

A conversation stays in its lane between its turns, so a turn reads only its new tokens.  `--env K=V` (repeatable) sets
a setting on every engine; `--exe PATH[:K=V,...]` gives one engine its own.  Exit 0: every turn of every run answered.

`--trace <rig's .served log>` replays real traffic instead: the log's requests (numbers only - prompt, cached,
completion, when, how fast) chained into conversations by their caches (a request whose cache covers the last one's
prompt is its next turn), and the busiest `--minutes` window's `--agents` conversations replayed at once - each first
reads, unmeasured, the context it had at the window's start, then its turns: their new tokens, their answers' lengths
(max_new), the time between them (less the engine time other requests took then: the log's engine ran one at a time),
and where it started in the window.  `--from "MM-DD HH:MM"` picks the window instead.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from parallel_test import Multi  # noqa: E402
from slot_keep_test import config_env, text_tokens  # noqa: E402


def workload(seed: int, agents: int, turns: int) -> list[dict]:
    """Per agent, its turns: the new tokens' count, the answer's max_new, the tool time after it."""
    rng = random.Random(seed)
    plan = []
    for _ in range(agents):
        ts = []
        for k in range(turns):
            if k == 0:
                new = rng.randint(4000, 12000)
            elif rng.random() < 0.1:
                new = rng.randint(5000, 20000)
            else:
                new = int(150 * (2000 / 150) ** rng.random())   # (log-uniform: most results short)
            ts.append({"new": new, "max_new": rng.randint(64, 400), "tool_s": rng.uniform(0.5, 5.0)})
        plan.append({"base": 0, "start_s": 0.0, "turns": ts})
    return plan


def trace_plan(path: str, agents: int, minutes: float, start: str, ctx: int) -> tuple[list[list[dict]], str]:
    """A rig .served log's busiest window (or `start`'s) as a plan: per conversation its turns there (see the doc)."""
    import datetime
    reqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                at = d["at"].rstrip("Z")
                if "." in at:
                    head, frac = at.split(".")
                    at = head + "." + frac[:6]
                end = datetime.datetime.fromisoformat(at).timestamp()
            except (ValueError, KeyError):
                continue
            new = max(0, d["prompt"] - d["cached"])
            pps, dps = d.get("prompt_per_second") or 0, d.get("decode_per_second") or 0
            svc = (new / pps if pps > 0 else 0.0) + (d["completion"] / dps if dps > 0 else 0.0)
            reqs.append({"start": end - svc, "end": end, "prompt": d["prompt"], "cached": d["cached"],
                         "new": new, "comp": d["completion"]})
    reqs.sort(key=lambda r: r["start"])
    convs: list[list[dict]] = []
    for r in reqs:   # (a conversation's next turn: its cache covers the last turn's prompt, not far past its answer)
        best = None
        for c in convs:
            last = c[-1]
            if last["end"] <= r["start"] + 1 and 0.9 * last["prompt"] <= r["cached"] <= last["prompt"] + last["comp"] + 64:
                if best is None or last["end"] > best[-1]["end"]:
                    best = c
        if best is None:
            convs.append([r])
        else:
            best.append(r)
    win = minutes * 60.0
    if start:
        year = datetime.datetime.fromtimestamp(reqs[-1]["end"]).year
        w0 = datetime.datetime.strptime(f"{year}-{start}", "%Y-%m-%d %H:%M").timestamp()
    else:   # (the busiest: the most turns starting in it)
        starts = [r["start"] for r in reqs]
        w0 = max(starts, key=lambda t: sum(1 for x in starts if t <= x < t + win))
    inwin = [[r for r in c if w0 <= r["start"] < w0 + win] for c in convs]
    inwin = sorted((c for c in inwin if c), key=len, reverse=True)[:agents]
    busy = sorted((r["start"], r["end"]) for r in reqs)

    def busy_in(t0: float, t1: float) -> float:   # (the engine's time on other requests between two turns)
        return sum(max(0.0, min(e, t1) - max(b, t0)) for b, e in busy if b < t1 and e > t0)

    plan = []
    for c in inwin:
        ts = []
        for k, r in enumerate(c):
            fresh = r["cached"] == 0
            gap = 0.0
            if k + 1 < len(c):
                gap = max(0.0, c[k + 1]["start"] - r["end"] - busy_in(r["end"], c[k + 1]["start"]))
            ts.append({"new": max(1, r["prompt"] if fresh else r["new"]), "max_new": max(1, r["comp"]),
                       "tool_s": gap, "fresh": fresh and k > 0})
        base = min(c[0]["cached"], ctx // 2)
        plan.append({"base": base, "start_s": c[0]["start"] - w0, "turns": ts})
    label = datetime.datetime.fromtimestamp(w0).strftime("%m-%d %H:%M")
    return plan, f"{path} from {label}, {minutes:g} min: {len(plan)} conversations, {sum(len(p['turns']) for p in plan)} turns"


def run(exe: str, a, tok, plan: list[dict], settings: dict) -> dict:
    env = dict(settings)
    tmp = tempfile.mkdtemp(prefix="ds4-agents-")
    # (--serial: the kept conversations on, as a harness runs --parallel 1 - each agent's turn takes its conversation
    #  back instead of reading it again; with lanes a conversation stays in its own)
    env.update({"MAYA_DS4_SLOT_DIR": tmp, "MAYA_DS4_SLOT_KEEP": "1" if a.serial else "0"})
    args = [*(("--gpus", a.gpus) if a.gpus else ()), "--pack", a.pack, "--max-context", str(a.ctx), "--dspark", "on",
            "--threads", "on", *(() if a.serial else ("--parallel", str(len(plan)))), "--vision", "off",
            "--expert-profile", "none", "--serve"]   # (--serial: no --parallel at all - a build before it runs too)
    eng = Multi(exe, args, env)
    eng.start(timeout=3600.0)
    send_mu = threading.Lock()   # (a prompt's line is far past a pipe's atomic write: one sender at a time)
    turn_mu = threading.Lock()   # (--serial: one turn at a time, as a harness queues them for --parallel 1)
    turns: list[dict] = []
    errors: list[str] = []
    done_at = [0.0] * len(plan)
    warm = threading.Barrier(len(plan) + 1)
    t0 = 0.0

    def agent(i: int) -> None:
        history: list[int] = []
        p = plan[i]
        if p["base"] > 0:   # (the context it had before: read, not measured)
            history = text_tokens(tok, f"Agent {i} context.", p["base"])
            with turn_mu if a.serial else contextlib.nullcontext():
                with send_mu:
                    rid = eng.send(history, 1, 0)
                r = eng.stream(rid)
            if r["error"]:
                errors.append(f"agent {i} context: {r['error']}")
            history = history + r["tokens"]
        warm.wait()
        time.sleep(p["start_s"])
        for k, t in enumerate(p["turns"]):
            if t.get("fresh"):
                history = []
            prompt = history + text_tokens(tok, f"Agent {i} turn {k}.", t["new"])
            ts = time.monotonic()   # (--serial: its wait for the turns ahead of it counts, as a harness's queue does)
            with turn_mu if a.serial else contextlib.nullcontext():
                with send_mu:
                    rid = eng.send(prompt, t["max_new"], 1 if a.serial or a.spec else 0)
                r = eng.stream(rid)
            if r["error"] or not r["times"]:
                errors.append(f"agent {i} turn {k}: {r['error'] or 'no tokens'}")
                return
            turns.append({"agent": i, "turn": k, "new": t["new"], "reused": r["reused"] or 0,
                          "first": r["times"][0] - ts, "done": r["times"][-1] - ts, "tokens": len(r["tokens"])})
            history = prompt + r["tokens"]
            if k + 1 < len(p["turns"]):
                time.sleep(t["tool_s"])
        done_at[i] = time.monotonic() - t0

    try:
        threads = [threading.Thread(target=agent, args=(i,)) for i in range(len(plan))]
        for th in threads:
            th.start()
        warm.wait()
        t0 = time.monotonic()
        for th in threads:
            th.join()
        wall = time.monotonic() - t0
    finally:
        try:
            eng.quit(timeout=120)
        except Exception:  # noqa: BLE001 - (an engine that will not end: ended)
            eng.proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    return {"turns": turns, "errors": errors, "wall": wall, "agents": done_at}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description="agents on --parallel N: seeded multi-turn tool-calling workload")
    ap.add_argument("--pack", required=True, help="the model's pack dir (maya-ds4)")
    ap.add_argument("--exe", action="append", required=True,
                    help="an engine, PATH or PATH:K=V,K=V (its own settings); repeat to compare")
    ap.add_argument("--gpus", default="", help="the GPUs (default: every visible one, the engine's own split)")
    ap.add_argument("--config", default="", help="a model config (maya-ds4.json): its DS4_/MAYA_DS4_ settings")
    ap.add_argument("--env", action="append", default=[], help="K=V on every engine (repeatable)")
    ap.add_argument("--ctx", type=int, default=131072, help="--max-context")
    ap.add_argument("--agents", type=int, default=4, help="agents at once (the engine's --parallel)")
    ap.add_argument("--turns", type=int, default=8, help="turns an agent")
    ap.add_argument("--seed", type=int, default=1, help="the workload's seed (the same on every engine)")
    ap.add_argument("--spec", action="store_true", help="the turns draft (spec=1): the DSpark drafter, on --parallel N too")
    ap.add_argument("--serial", action="store_true",
                    help="--parallel 1 (the DSpark drafter on), the agents' turns one at a time - today's harness queue")
    ap.add_argument("--trace", default="", help="a rig .served log to replay instead (see the doc)")
    ap.add_argument("--minutes", type=float, default=10.0, help="--trace: the window's length")
    ap.add_argument("--from", dest="start", default="", help='--trace: the window\'s start, "MM-DD HH:MM" (default: busiest)')
    a = ap.parse_args()
    try:
        from strata_tokenizer import Tokenizer
        tok = Tokenizer.from_dir(os.path.join(a.pack, "tokenizer"))
    except Exception as e:  # noqa: BLE001
        print(f"{os.path.join(a.pack, 'tokenizer')}: no tokenizer (the prompts are real text): {e}", flush=True)
        return 2
    base = config_env(a.config) if a.config else {}
    base.update(dict(kv.split("=", 1) for kv in a.env))
    if a.trace:
        plan, label = trace_plan(a.trace, a.agents, a.minutes, a.start, a.ctx)
        print(label, flush=True)
    else:
        plan = workload(a.seed, a.agents, a.turns)
        print(f"{a.agents} agents x {a.turns} turns (seed {a.seed})", flush=True)
    news = [t["new"] for p in plan for t in p["turns"]]
    comps = [t["max_new"] for p in plan for t in p["turns"]]
    print(f"  new tokens a turn {min(news)}-{max(news)} (median {int(statistics.median(news))}), {sum(news)} in all; "
          f"answers up to {sum(comps)} tokens; contexts read first {[p['base'] for p in plan]}", flush=True)
    total_turns = sum(len(p["turns"]) for p in plan)
    ok = True
    for spec in a.exe:
        exe, _, kvs = spec.partition(":")
        settings = dict(base)
        if kvs:
            settings.update(dict(kv.split("=", 1) for kv in kvs.split(",")))
        res = run(exe, a, tok, plan, settings)
        firsts = [t["first"] for t in res["turns"]]
        dones = [t["done"] for t in res["turns"]]
        print(f"\n{spec}", flush=True)
        if res["errors"]:
            ok = False
            for e in res["errors"]:
                print(f"  ERR {e}", flush=True)
        print(f"  turns answered {len(res['turns'])} of {total_turns}; wall {res['wall']:.1f} s; agents done "
              + ", ".join(f"{x:.0f}" for x in res["agents"]) + " s", flush=True)
        if not firsts:
            ok = False
            continue
        print(f"  first token  mean {statistics.mean(firsts):6.2f}  p50 {pct(firsts, 50):6.2f}  p95 {pct(firsts, 95):6.2f}  "
              f"max {max(firsts):6.2f} s", flush=True)
        print(f"  turn done    mean {statistics.mean(dones):6.2f}  p50 {pct(dones, 50):6.2f}  p95 {pct(dones, 95):6.2f}  "
              f"max {max(dones):6.2f} s", flush=True)
        short = [t["done"] for t in res["turns"] if t["new"] <= 2000]
        if short:
            print(f"  turns of <= 2,000 new tokens: done mean {statistics.mean(short):.2f}, p95 {pct(short, 95):.2f} s "
                  f"({len(short)} turns)", flush=True)
        reused = sum(1 for t in res["turns"] if t["turn"] > 0 and t["reused"] > 0)
        print(f"  later turns that continued their conversation in place: {reused} of "
              f"{sum(1 for t in res['turns'] if t['turn'] > 0)}", flush=True)
    print("\nevery turn answered" if ok else "\na turn failed (above)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
