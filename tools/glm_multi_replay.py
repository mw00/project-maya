#!/usr/bin/env python3
"""tools/glm_multi_replay.py - a scripted multi-conversation workload against a running Maya server, for A/B runs of
engine settings that only show under several long conversations at once (the expert tiers, --parallel's scheduler).

The workload imitates agentic coding clients: N conversations at once, each a long system prompt shared with other
conversations (a few distinct ones), then turns that append a tool result or a user message of mostly short, sometimes
long new text and ask for a short answer, with a think time between turns (the client running its tools); short
"check" requests that repeat one long system prompt with a fresh transcript at a fixed rate; and cold bursts, where
some clients start over (their whole conversation read again with nothing reusable past the system prompt), as after a
client restart. Everything is fixed by --seed: the texts (slices of this repository's files), the sizes, the think
times, the schedule. The model's own answers are discarded - the next turn sends the scripted history - so every run
sends exactly the same prompts whatever the model wrote; greedy, with the end tokens banned (--ban), so every request
writes exactly its max_tokens.

Partial residency: the tiers only matter where part of the experts live in RAM - a model larger than the VRAM, or
the expert pool capped (`--expert-cache N` in the config's "args", the first GPU's pool at N experts). The prompts
grow with the conversations (a client whose conversation passes --max-context starts a new one from its latest turn,
as a client compacts); a run of 30 minutes reaches contexts of 100-200K tokens.

What it records: every request's timings from the server (GET /metrics?requests=all: prompt tokens, reused, read ms,
decode tok/s, the decode's VRAM hit); the engine's tier readings every --poll seconds (VRAM hit, RAM-tier experts
fetched over PCIe a token, promotions, running / queued); and, with --route-log (the engine started with
MAYA_GLM_ROUTE_LOG=<prefix> on the same host), the VRAM hit of layer bands over time from the route log - whether
the first MoE layers lose their residents as the run goes on. The first --warmup seconds are left out of the summary.

    # one run (one engine start, an empty slot folder, MAYA_GLM_SLOT_KEEP off, the same config every time but the A/B):
    python tools/glm_multi_replay.py run --url http://127.0.0.1:8080 --label t2 --rep 1 --out bench/results/t2-1.json
    # A/B: alternate the configs (A B A B ...), a fresh engine each run, then
    python tools/glm_multi_replay.py compare bench/results/t1-*.json bench/results/t2-*.json
    # the workload without a server (what each client sends, when)
    python tools/glm_multi_replay.py plan --seed 1

Compare runs at the same --seed, --clients and --duration on the same machine and configuration but the setting under
test. The summary splits decode by context length and reads by their new tokens, so runs whose conversations moved
differently still compare like with like; compare lists each metric's mean and standard deviation over the reps of a
label and marks a difference between two labels larger than twice its standard error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"),
           ("tools", "*.py")]
CHARS_PER_TOKEN = 3.2
GLM_ENDS = "154820,154827,154829"   # GLM-5.3-Flash's <|endoftext|>, <|user|>, <|observation|>
# new tokens a turn appends: (low, high, share) - mostly short increments, a few long tool results
FRESH = [(64, 512, 0.35), (512, 2048, 0.45), (2048, 8192, 0.15), (8192, 32768, 0.04), (32768, 65536, 0.01)]
OUT = [(32, 128, 0.4), (128, 512, 0.45), (512, 1024, 0.15)]                 # tokens a turn asks for
READ_BINS = [(1, 512), (512, 2048), (2048, 8192), (8192, 32768), (32768, 1 << 30)]
CTX_BANDS = [(0, 65536), (65536, 131072), (131072, 262144), (262144, 1 << 30)]


# ---------------------------------------------------------------- the workload (pure, seeded)
class Corpus:
    """The repository's text files in a fixed order; text(rnd, tokens) is a slice of about that many tokens."""

    def __init__(self, root: Path = ROOT, text: str | None = None):
        if text is None:
            parts = []
            for d, pat in SOURCES:
                for f in sorted((root / d).rglob(pat)):
                    try:
                        parts.append(f"\n\n=== {f.relative_to(root).as_posix()} ===\n" +
                                     f.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        continue
            text = "".join(parts)
        if len(text) < 1000:
            raise SystemExit("not enough text in the repository to build the prompts from")
        self.text = text

    def take(self, rnd: random.Random, tokens: int) -> str:
        n = max(1, int(tokens * CHARS_PER_TOKEN))
        out, need = [], n
        while need > 0:
            at = rnd.randrange(len(self.text))
            piece = self.text[at:at + need]
            out.append(piece)
            need -= len(piece)
        return "".join(out)


def draw(rnd: random.Random, table) -> int:
    x, acc = rnd.random(), 0.0
    for lo, hi, share in table:
        acc += share
        if x < acc:
            return rnd.randint(lo, hi)
    lo, hi, _ = table[-1]
    return rnd.randint(lo, hi)


def plan(seed: int = 1, clients: int = 8, duration: float = 1800.0, sys_tokens=(30000, 8000, 4000),
         first_tokens=(2000, 6000), think=(2.0, 20.0), checks_per_hour: float = 40.0, check_sys: int = 30000,
         check_tokens=(2000, 4000), bursts=None, burst_clients: int | None = None) -> dict:
    """The whole workload as data (no text): per client its system prompt and turns (new tokens, max tokens, think
    time, cold restart or not), the check requests' times, and the cold bursts' times.  Deterministic in its args."""
    rnd = random.Random(seed)
    bursts = [duration / 3, 2 * duration / 3] if bursts is None else list(bursts)
    burst_clients = clients // 2 if burst_clients is None else burst_clients
    cl = []
    for c in range(clients):
        turns, t = [], 0.0
        first = rnd.randint(*first_tokens)
        while t < duration:
            turns.append({"fresh": draw(rnd, FRESH), "max_tokens": draw(rnd, OUT),
                          "think": round(rnd.uniform(*think), 2)})
            t += turns[-1]["think"] + 2.0      # (a rough floor for the request itself: the real schedule waits on it)
        cl.append({"sys": c % len(sys_tokens), "first": first, "turns": turns})
    # which clients restart at each burst: a fixed, seeded choice
    burst_sets = [sorted(rnd.sample(range(clients), min(burst_clients, clients))) for _ in bursts]
    checks = []
    if checks_per_hour > 0:
        step = 3600.0 / checks_per_hour
        k = 0
        while k * step < duration:
            checks.append({"at": round(k * step + rnd.uniform(0, step / 2), 2), "fresh": rnd.randint(*check_tokens)})
            k += 1
    return {"seed": seed, "clients": cl, "sys_tokens": list(sys_tokens), "check_sys": check_sys, "checks": checks,
            "bursts": [{"at": round(b, 1), "clients": s} for b, s in zip(bursts, burst_sets)], "duration": duration}


class Texts:
    """The texts of a plan, built once from the corpus with their own seeded generator (the same every run)."""

    def __init__(self, p: dict, corpus: Corpus):
        rnd = random.Random(p["seed"] * 7919 + 1)
        self.sys = [corpus.take(rnd, n) for n in p["sys_tokens"]]
        self.check_sys = corpus.take(rnd, p["check_sys"])
        self.first = [corpus.take(rnd, c["first"]) for c in p["clients"]]
        self.turn = [[corpus.take(rnd, t["fresh"]) for t in c["turns"]] for c in p["clients"]]
        # the scripted assistant turns (the model's answers are not fed back): short fixed texts
        self.reply = [[corpus.take(rnd, min(t["max_tokens"], 256)) for t in c["turns"]] for c in p["clients"]]
        self.check = [corpus.take(rnd, k["fresh"]) for k in p["checks"]]
        self.restart = [f"(conversation {i + 1})\n" for i in range(len(p["bursts"]) + 64)]


def messages(texts: Texts, p: dict, c: int, upto: int, restarts: int, start: int = 0) -> list:
    """Client c's request for turn `upto`: its system prompt, its first message (marked by how often it restarted, so a
    restart shares nothing past the system prompt with the earlier reads), then turns start..upto-1 with the scripted
    replies, then turn upto's new text.  (start > 0: the conversation was compacted there - see run's --max-context.)"""
    m = [{"role": "system", "content": texts.sys[p["clients"][c]["sys"]]},
         {"role": "user", "content": texts.restart[min(restarts, len(texts.restart) - 1)] + texts.first[c]}]
    for k in range(start, upto):
        m.append({"role": "assistant", "content": texts.reply[c][k]})
        m.append({"role": "user", "content": texts.turn[c][k]})
    m.append({"role": "assistant", "content": texts.reply[c][upto]})
    m.append({"role": "user", "content": texts.turn[c][upto]})
    return m


# ---------------------------------------------------------------- the summary (pure)
def band_of(x: float, bands) -> int | None:
    return next((i for i, (lo, hi) in enumerate(bands) if lo <= x < hi), None)


def summarize(records: list, tiers: list, t0: float, warmup: float, route_bands: dict | None = None) -> dict:
    """records: /metrics request records ("time", "prompt_tokens", "reused", "output_tokens", "prompt_ms",
    "decode_tok_s", "hit_rate"); tiers: (time, tiers.now dict, live dict) samples.  Only what ends after the warm-up."""
    t_from = t0 + warmup
    rec = [r for r in records if r.get("time", 0) >= t_from]
    out = {"requests": len(rec)}
    now = [n for t, n, _ in tiers if t >= t_from and n]
    live = [lv for t, _, lv in tiers if t >= t_from and lv]
    for k in ("vram_hit", "ram_fetch", "promo", "ms_tok"):
        xs = [n[k] for n in now if n.get(k) is not None]
        out[k] = round(st.mean(xs), 5) if xs else None
    for k in ("active", "queued"):
        xs = [lv.get(k) or 0 for lv in live]
        out[k] = round(st.mean(xs), 3) if xs else None
    hits = [r["hit_rate"] for r in rec if r.get("hit_rate") is not None]
    out["hit_per_request_median"] = round(st.median(hits), 5) if hits else None
    dec = [r for r in rec if (r.get("output_tokens") or 0) >= 50 and r.get("decode_tok_s")]
    out["decode_median"] = round(st.median([r["decode_tok_s"] for r in dec]), 2) if dec else None
    out["decode_by_context"] = {}
    for i, (lo, hi) in enumerate(CTX_BANDS):
        xs = [r["decode_tok_s"] for r in dec if band_of(r.get("prompt_tokens") or 0, CTX_BANDS) == i]
        out["decode_by_context"][f"{lo // 1024}K+"] = [round(st.median(xs), 2), len(xs)] if xs else None
    out["read_by_fresh"] = {}
    for lo, hi in READ_BINS:
        g = [r for r in rec if lo <= (r.get("prompt_tokens") or 0) - (r.get("reused") or 0) < hi and
             (r.get("prompt_ms") or 0) > 0]
        if g:
            tok = sum((r["prompt_tokens"] - (r.get("reused") or 0)) for r in g)
            ms = sum(r["prompt_ms"] for r in g)
            out["read_by_fresh"][f"{lo}+"] = [round(tok / (ms / 1000), 1), len(g)]
    if route_bands:
        out["route_bands"] = route_bands
    return out


def route_band_hits(lines, bands=((3, 7), (8, 15), (16, 23)), slices: int = 4) -> list:
    """A route log's VRAM hit by layer band over `slices` equal parts of `lines` (the log's own masks: an expert in
    no mask of fetch / miss / cpu was in VRAM): [[band hit of slice 0 ...], ...] in percent, None for an empty cell."""
    lines = list(lines)
    per = max(1, math.ceil(len(lines) / slices))
    cnt = [[[0, 0] for _ in bands] for _ in range(slices)]
    for k, line in enumerate(lines):
        p = line.split("|")[0].split()
        if len(p) < 12:
            continue
        il = int(p[0])
        g = band_of(il, [(lo, hi + 1) for lo, hi in bands])
        if g is None:
            continue
        off = int(p[9]) | int(p[10]) | int(p[11])
        c = cnt[min(k // per, slices - 1)][g]
        c[0] += 8
        c[1] += 8 - bin(off & 0xFF).count("1")
    return [[round(100.0 * h / n, 2) if n else None for n, h in s] for s in cnt]


def compare(runs: list) -> str:
    """Runs (each {"label", "rep", "summary"}) by label: mean +- sd of the scalar metrics over the reps, and for two
    labels the difference, marked '*' where it exceeds twice its standard error."""
    by = {}
    for r in runs:
        by.setdefault(r["label"], []).append(r["summary"])
    labels = sorted(by)
    keys = ["vram_hit", "hit_per_request_median", "ram_fetch", "promo", "decode_median", "active", "queued"]
    keys += [f"decode {b}" for b in [f"{lo // 1024}K+" for lo, _ in CTX_BANDS]]
    keys += [f"read {b}" for b in [f"{lo}+" for lo, _ in READ_BINS]]

    def val(s, k):
        if k.startswith("decode "):
            v = (s.get("decode_by_context") or {}).get(k[7:])
            return v[0] if v else None
        if k.startswith("read "):
            v = (s.get("read_by_fresh") or {}).get(k[5:])
            return v[0] if v else None
        return s.get(k)

    lines = ["| metric | " + " | ".join(f"{lb} (n={len(by[lb])})" for lb in labels) +
             (" | difference |" if len(labels) == 2 else " |"),
             "|---|" + "---|" * (len(labels) + (1 if len(labels) == 2 else 0))]
    for k in keys:
        cells, stats = [], []
        for lb in labels:
            xs = [v for v in (val(s, k) for s in by[lb]) if v is not None]
            if not xs:
                cells.append("-")
                stats.append(None)
                continue
            m = st.mean(xs)
            sd = st.stdev(xs) if len(xs) > 1 else 0.0
            cells.append(f"{m:.4g} +- {sd:.2g}")
            stats.append((m, sd, len(xs)))
        row = f"| {k} | " + " | ".join(cells)
        if len(labels) == 2:
            a, b = stats
            if a and b:
                se = math.sqrt(a[1] ** 2 / a[2] + b[1] ** 2 / b[2])
                d = b[0] - a[0]
                rel = f" ({100 * d / a[0]:+.1f}%)" if a[0] else ""
                row += f" | {d:+.4g}{rel}{' *' if se > 0 and abs(d) > 2 * se else ''}"
            else:
                row += " | -"
        lines.append(row + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- the run (talks to a server)
def get_json(url: str, timeout: float = 30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def run(a) -> int:
    p = plan(a.seed, a.clients, a.duration, tuple(int(x) for x in a.sys_tokens.split(",")),
             checks_per_hour=a.checks_per_hour, burst_clients=a.burst_clients)
    texts = Texts(p, Corpus())
    base = a.url.rstrip("/")
    ban = {int(t): -100 for t in a.ban.split(",") if t.strip()} if a.ban else {}
    lock = threading.Lock()
    sent, errors = [], []
    t0 = time.time()
    stop = threading.Event()

    def post(msgs, max_tokens, kind, c):
        body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0, "messages": msgs,
                "chat_template_kwargs": {"enable_thinking": False}}
        if ban:
            body["logit_bias"] = {str(k): v for k, v in ban.items()}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        s = time.time()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                out = json.loads(r.read())
            u = out.get("usage") or {}
            with lock:
                sent.append({"kind": kind, "client": c, "t_send": s - t0, "t_done": time.time() - t0,
                             "prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")})
        except (OSError, urllib.error.HTTPError, ValueError) as e:
            with lock:
                errors.append({"kind": kind, "client": c, "t": s - t0, "error": str(e)[:200]})

    def client(c):
        restarts, k, start = 0, 0, 0
        turns = p["clients"][c]["turns"]
        burst_at = [b["at"] for b in p["bursts"] if c in b["clients"]]
        while k < len(turns) and not stop.is_set():
            now = time.time() - t0
            if now >= p["duration"]:
                break
            kind = "turn"
            while burst_at and now >= burst_at[0]:   # a restart: the next request shares only the system prompt
                burst_at.pop(0)
                restarts += 1
                kind = "cold"
            msgs = messages(texts, p, c, k, restarts, start)
            if sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN > a.max_context:
                # compacted, as a client does when its conversation outgrows the window: a new conversation from here
                restarts, start = restarts + 1, k
                msgs = messages(texts, p, c, k, restarts, start)
                kind = "compact"
            post(msgs, turns[k]["max_tokens"], kind, c)
            stop.wait(turns[k]["think"])
            k += 1

    def checker():
        for i, ck in enumerate(p["checks"]):
            if stop.wait(max(0.0, ck["at"] - (time.time() - t0))):
                return
            msgs = [{"role": "system", "content": texts.check_sys}, {"role": "user", "content": texts.check[i]}]
            threading.Thread(target=post, args=(msgs, 16, "check", -1), daemon=True).start()

    records, tiers, sizes = {}, [], []

    def poller():
        while not stop.is_set():
            try:
                m = get_json(base + "/metrics?requests=all")
                for r in m.get("requests") or []:
                    records[(round(r.get("time", 0), 3), r.get("prompt_tokens"), r.get("output_tokens"))] = r
                tiers.append((time.time(), (m.get("tiers") or {}).get("now"), m.get("live")))
                if a.route_log:
                    sizes.append((time.time(), [os.path.getsize(f"{a.route_log}.{d}") if
                                                os.path.exists(f"{a.route_log}.{d}") else 0 for d in (0, 1)]))
            except (OSError, ValueError):
                pass
            stop.wait(a.poll)

    threads = [threading.Thread(target=client, args=(c,), daemon=True) for c in range(a.clients)]
    threads += [threading.Thread(target=checker, daemon=True), threading.Thread(target=poller, daemon=True)]
    for th in threads:
        th.start()
    end = t0 + p["duration"]
    while time.time() < end and any(th.is_alive() for th in threads[:a.clients]):
        time.sleep(1)
    time.sleep(a.drain)                       # the last answers come back
    stop.set()
    for th in threads[:a.clients]:
        th.join(timeout=a.timeout)
    try:
        for r in get_json(base + "/metrics?requests=all").get("requests") or []:
            records[(round(r.get("time", 0), 3), r.get("prompt_tokens"), r.get("output_tokens"))] = r
    except (OSError, ValueError):
        pass
    route = None
    if a.route_log:
        route = {}
        t_from = t0 + a.warmup
        start = next((s for t, s in sizes if t >= t_from), None)
        for d in (0, 1):
            fn = f"{a.route_log}.{d}"
            if start is None or not os.path.exists(fn):
                continue
            with open(fn, "rb") as f:
                f.seek(start[d])
                body = f.read().decode(errors="replace").splitlines()[1:]   # (a partial first line)
            route[str(d)] = route_band_hits(body, slices=a.slices)
    rec = [r for r in records.values() if r.get("time", 0) >= t0]
    summary = summarize(rec, tiers, t0, a.warmup, route)
    out = {"label": a.label, "rep": a.rep, "seed": a.seed, "clients": a.clients, "duration": a.duration,
           "warmup": a.warmup, "started": t0, "summary": summary, "errors": errors, "sent": sent,
           "records": sorted(rec, key=lambda r: r.get("time", 0)),
           "tiers": [[round(t - t0, 1), n, lv] for t, n, lv in tiers]}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"{len(sent)} requests answered, {len(errors)} failed")
    return 0 if not errors else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--clients", type=int, default=8, help="conversations at once (default 8)")
        s.add_argument("--duration", type=float, default=1800, help="seconds of workload (default 1800)")
        s.add_argument("--sys-tokens", default="30000,8000,4000", help="the shared system prompts' sizes")
        s.add_argument("--checks-per-hour", type=float, default=40)
        s.add_argument("--burst-clients", type=int, default=None, help="clients restarting at each burst (half)")
    r = sub.choices["run"]
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--label", required=True, help="the configuration's name in the comparison")
    r.add_argument("--rep", type=int, default=1)
    r.add_argument("--out", help="the run's JSON")
    r.add_argument("--warmup", type=float, default=300, help="seconds left out of the summary (default 300)")
    r.add_argument("--poll", type=float, default=10)
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--drain", type=float, default=30)
    r.add_argument("--ban", default=GLM_ENDS, help="token ids banned (the model's end tokens); '' = none")
    r.add_argument("--route-log", help="the engine's MAYA_GLM_ROUTE_LOG prefix (the same host)")
    r.add_argument("--slices", type=int, default=4, help="equal parts of the route log after the warm-up (in order)")
    r.add_argument("--max-context", type=int, default=200000,
                   help="a conversation estimated past this many tokens starts over from its latest turn (default "
                        "200000; keep it under the server's context)")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        p = plan(a.seed, a.clients, a.duration, tuple(int(x) for x in a.sys_tokens.split(",")),
                 checks_per_hour=a.checks_per_hour, burst_clients=a.burst_clients)
        for i, cl in enumerate(p["clients"]):
            fresh = [t["fresh"] for t in cl["turns"]]
            print(f"client {i}: system prompt {p['sys_tokens'][cl['sys']]} tokens, first {cl['first']}, {len(fresh)} "
                  f"turns, new tokens median {st.median(fresh)}, max {max(fresh)}, {sum(fresh)} in all")
        print(f"{len(p['checks'])} checks ({p['check_sys']}-token system prompt); bursts: " +
              ", ".join(f"{b['at']} s: clients {b['clients']}" for b in p["bursts"]))
        return 0
    if a.cmd == "compare":
        runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files]
        print(compare(runs))
        rb = {}
        for rr in runs:
            for d, slices in ((rr["summary"].get("route_bands") or {}).items()):
                rb.setdefault((rr["label"], d), []).append(slices)
        for (lb, d), xs in sorted(rb.items()):
            first = [s[0][0] for s in xs if s and s[0][0] is not None]
            last = [s[-1][0] for s in xs if s and s[-1][0] is not None]
            if first and last:
                print(f"{lb} GPU{d} first band's VRAM hit, first slice -> last: {st.mean(first):.2f} -> "
                      f"{st.mean(last):.2f} % (over {len(xs)} runs)")
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
