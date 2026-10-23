#!/usr/bin/env python3
"""tools/agent_bench.py - a benchmark shaped like coding agents: several long conversations at once against a
running server, through its OpenAI API.

`./maya.sh --bench` measures one request at a time on a cold, short prompt. Coding agents load a server differently:
several conversations run at once, each a long system prompt shared with other conversations, then turns that append
a tool result or a message (mostly short, sometimes long) to an ever longer history and ask for a short answer, with a
pause between turns while the agent runs its tools. What an agent waits for is then the read of its new text (behind
the other conversations' reads), its answer's decode while the others decode too, and how both hold up as the
contexts grow. This tool runs that workload from a script, fixed by --seed, so two runs - two settings, two builds,
two machines - send exactly the same requests:

- N conversations at once (--clients), each with one of a few shared system prompts (--sys-tokens) and a first message,
  then turns of new text (64 to 64K tokens, mostly under 2K) that ask for 32-1024 tokens, with a think time of 2-20 s
  between an answer and the next turn;
- short "checks" that send one long system prompt with a fresh question at a fixed rate (--checks-per-hour), as a
  permission or routing check of an agent does;
- two cold bursts (at a third and two thirds of the run), where half the conversations start over - read again with
  nothing reusable past the system prompt, as after an agent restart;
- a conversation that outgrows --max-context starts a new one from its latest turn, as an agent compacts.

The texts are slices of this repository's files (--corpus: another folder). The model's answers are not fed back:
the next turn sends the scripted history, so every run sends the same prompts whatever the model wrote. Requests are
greedy (temperature 0, thinking off) and ban the model's end tokens (--ban; `auto` knows GLM-5.3-Flash by the served
model's name), so every request writes its whole max_tokens.

What it measures, client side (any OpenAI-compatible server; requests are streamed):
- aggregate output tokens/s over the measured window (after --warmup), and the mean number of requests in flight;
- per request: the time to the first token (the read of its new text, and any wait behind other reads), the decode
  rate from the first token to the last (what the agent sees, slowed by the other conversations' reads and steps),
  and the whole request's latency;
- decode by context length, reads by their number of new tokens (the server's `timings` when it sends them, as Maya
  and llama.cpp's server do: the read alone; else the time to the first token);
and, from a Maya server's GET /metrics: each request's VRAM hit of its routed experts (with experts in RAM) and the
expert tiers' readings over the run.

    # a run: start the server as you use it (e.g. ./maya.sh, with --parallel 8), then
    python tools/agent_bench.py run --url http://127.0.0.1:8080 --clients 4 --label base --out runs/base-1.json
    # A/B: alternate the settings (A B A B ...), a fresh server each run, then
    python tools/agent_bench.py compare runs/base-*.json runs/new-*.json
    # the workload without a server: what each conversation sends, when
    python tools/agent_bench.py plan --clients 4

Compare runs at the same --seed, --clients, --duration and --warmup. A run of 15-30 minutes reaches contexts of
100-200K tokens; the first minutes fill the server's caches (leave them out with --warmup). compare lists each
metric's mean and standard deviation over the reps of a label and marks a difference between two labels larger than
twice its standard error: run each setting at least twice. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
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
END_TOKENS = {"glm": "154820,154827,154829"}   # GLM-5.3-Flash's <|endoftext|>, <|user|>, <|observation|>
# new tokens a turn appends: (low, high, share) - mostly short increments, a few long tool results
FRESH = [(64, 512, 0.35), (512, 2048, 0.45), (2048, 8192, 0.15), (8192, 32768, 0.04), (32768, 65536, 0.01)]
OUT = [(32, 128, 0.4), (128, 512, 0.45), (512, 1024, 0.15)]                 # tokens a turn asks for
READ_BINS = [(1, 512), (512, 2048), (2048, 8192), (8192, 32768), (32768, 1 << 30)]
CTX_BANDS = [(0, 32768), (32768, 65536), (65536, 131072), (131072, 262144), (262144, 1 << 30)]
MIN_DECODE = 50          # tokens a request must write for its decode rate to count


# ---------------------------------------------------------------- the workload (pure, seeded)
class Corpus:
    """Text files of a folder (default: this repository's docs and sources) in a fixed order; take(rnd, tokens) is a
    slice of about that many tokens."""

    def __init__(self, root: Path | None = None, text: str | None = None):
        if text is None:
            parts = []
            if root is None:
                files = [f for d, pat in SOURCES for f in sorted((ROOT / d).rglob(pat))]
                base = ROOT
            else:
                base = Path(root)
                files = sorted(f for f in base.rglob("*") if f.is_file() and f.suffix in
                               (".md", ".txt", ".py", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".rs", ".go", ".js",
                                ".ts", ".java", ".json", ".toml", ".yaml", ".yml", ".sh"))
            for f in files:
                try:
                    parts.append(f"\n\n=== {f.relative_to(base).as_posix()} ===\n" +
                                 f.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
            text = "".join(parts)
        if len(text) < 1000:
            raise SystemExit("not enough text to build the prompts from (--corpus: a folder of text files)")
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


def plan(seed: int = 1, clients: int = 4, duration: float = 900.0, sys_tokens=(30000, 8000, 4000),
         first_tokens=(2000, 6000), think=(2.0, 20.0), checks_per_hour: float = 40.0, check_sys: int = 30000,
         check_tokens=(2000, 4000), bursts=None, burst_clients: int | None = None) -> dict:
    """The whole workload as data (no text): per conversation its system prompt and turns (new tokens, max tokens,
    think time), the checks' times and the cold bursts.  Deterministic in its arguments."""
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
            t += turns[-1]["think"] + 2.0      # (a floor for the request itself: the real schedule waits on it)
        cl.append({"sys": c % len(sys_tokens), "first": first, "turns": turns})
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
    """Conversation c's request for turn `upto`: its system prompt, its first message (marked by how often it started
    over, so a restart shares nothing past the system prompt with the reads before), turns start..upto-1 with the
    scripted replies, then turn upto's new text.  (start > 0: compacted there.)"""
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


def pct(xs: list, q: float):
    """The q-quantile (0..1) of xs by the nearest rank below; None for none."""
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def rnd_or_none(x, nd: int = 2):
    return None if x is None else round(x, nd)


def summarize(reqs: list, warmup: float, duration: float, server: dict | None = None) -> dict:
    """reqs: the client's request records ("kind", "t_send", "t_first", "t_done", "prompt_tokens", "cached",
    "output_tokens", optional "read_ms" from the server's timings).  The window is [warmup, duration]: a request
    counts when it ends inside it."""
    win = max(1e-9, duration - warmup)
    done = [r for r in reqs if r.get("t_done") is not None and warmup <= r["t_done"] <= duration and not r.get("error")]
    turns = [r for r in done if r["kind"] != "check"]
    out = {"requests": len(done), "turns": len(turns), "checks": len(done) - len(turns),
           "errors": sum(1 for r in reqs if r.get("error"))}
    out["output_tok_s"] = round(sum(r.get("output_tokens") or 0 for r in done) / win, 2)
    # requests in flight, averaged over the window (every request, checks too)
    out["in_flight"] = round(sum(max(0.0, min(r["t_done"] if r.get("t_done") is not None else duration, duration) -
                                      max(r["t_send"], warmup)) for r in reqs) / win, 3)
    lat = [r["t_done"] - r["t_send"] for r in turns]
    ttft = [r["t_first"] - r["t_send"] for r in turns if r.get("t_first") is not None]
    med = lambda xs: rnd_or_none(st.median(xs)) if xs else None
    out["latency_median"], out["latency_p90"] = med(lat), rnd_or_none(pct(lat, 0.9))
    out["ttft_median"], out["ttft_p90"] = med(ttft), rnd_or_none(pct(ttft, 0.9))
    dec = [(r, (r["output_tokens"] - 1) / (r["t_done"] - r["t_first"])) for r in done
           if (r.get("output_tokens") or 0) >= MIN_DECODE and r.get("t_first") is not None and r["t_done"] > r["t_first"]]
    xs = [d for _, d in dec]
    out["decode_median"], out["decode_p10"] = med(xs), rnd_or_none(pct(xs, 0.1))
    out["decode_by_context"] = {}
    for i, (lo, hi) in enumerate(CTX_BANDS):
        ys = [d for r, d in dec if band_of(r.get("prompt_tokens") or 0, CTX_BANDS) == i]
        out["decode_by_context"][f"{lo // 1024}K+"] = [round(st.median(ys), 2), len(ys)] if ys else None
    out["read_by_fresh"] = {}
    for lo, hi in READ_BINS:
        g = [r for r in done if lo <= (r.get("prompt_tokens") or 0) - (r.get("cached") or 0) < hi]
        ms = [(r["read_ms"] if r.get("read_ms") else
               (1000 * (r["t_first"] - r["t_send"]) if r.get("t_first") is not None else None)) for r in g]
        pairs = [((r.get("prompt_tokens") or 0) - (r.get("cached") or 0), m) for r, m in zip(g, ms) if m]
        if pairs:
            out["read_by_fresh"][f"{lo}+"] = [round(sum(t for t, _ in pairs) / (sum(m for _, m in pairs) / 1000), 1),
                                              len(pairs)]
    out["read_source"] = "server" if any(r.get("read_ms") for r in done) else "client (time to first token)"
    if server:
        out.update(server)
    return out


def server_summary(records: list, tiers: list, t0: float, warmup: float, duration: float) -> dict:
    """A Maya server's view (GET /metrics): the median VRAM hit of the requests that ended in the window and the
    tiers' readings (VRAM hit, RAM-tier experts fetched over PCIe a token, promotions a token) averaged over it."""
    out = {}
    rec = [r for r in records if t0 + warmup <= (r.get("time") or 0) + (r.get("duration_s") or 0) <= t0 + duration]
    hits = [r["hit_rate"] for r in rec if r.get("hit_rate") is not None]
    if hits:
        out["vram_hit_per_request"] = round(st.median(hits), 5)
    now = [n for t, n in tiers if warmup <= t <= duration and n]
    for k in ("vram_hit", "ram_fetch", "promo"):
        xs = [n[k] for n in now if n.get(k) is not None]
        if xs:
            out[k] = round(st.mean(xs), 5)
    return out


def compare(runs: list) -> str:
    """Runs (each {"label", "rep", "summary"}) by label: mean +- sd of the scalar metrics over the reps, and for two
    labels the difference, marked '*' where it exceeds twice its standard error."""
    by = {}
    for r in runs:
        by.setdefault(r["label"], []).append(r["summary"])
    labels = list(dict.fromkeys(r["label"] for r in runs))
    keys = ["output_tok_s", "decode_median", "decode_p10", "latency_median", "latency_p90", "ttft_median", "ttft_p90",
            "in_flight", "turns", "errors", "vram_hit_per_request", "vram_hit", "ram_fetch", "promo"]
    keys += [f"decode {b}" for b in [f"{lo // 1024}K+" for lo, _ in CTX_BANDS]]
    keys += [f"read {b}" for b in [f"{lo}+" for lo, _ in READ_BINS]]

    def val(s, k):
        if k.startswith("decode ") and k != "decode_median":
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
        if all(c == "-" for c in cells):
            continue
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
def get_json(url: str, timeout: float = 30, key: str | None = None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def served_model(base: str, key: str | None) -> tuple[str, int | None]:
    """The served model's id and context (/v1/models; meta.n_ctx where the server says it), or ("", None)."""
    try:
        d = (get_json(base + "/v1/models", 10, key).get("data") or [{}])[0]
        return str(d.get("id") or ""), (d.get("meta") or {}).get("n_ctx")
    except (OSError, ValueError, IndexError):
        return "", None


def ban_of(spec: str, model: str) -> dict:
    """--ban: 'auto' (the model's end tokens when its name is known), 'none', or token ids '1,2,3'."""
    if spec == "auto":
        spec = next((ids for name, ids in END_TOKENS.items() if name in model.lower()), "none")
    if spec in ("none", ""):
        return {}
    return {str(int(t)): -100 for t in spec.split(",") if t.strip()}


def stream_request(base: str, body: dict, timeout: float, key: str | None, t0: float) -> dict:
    """One streamed chat completion: the times of the send, the first token (content or reasoning) and the end, and
    the usage / timings of its last chunk."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    rec = {"t_send": time.time() - t0, "t_first": None}
    usage, timings = {}, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                c = json.loads(data)
            except ValueError:
                continue
            for ch in c.get("choices") or []:
                d = ch.get("delta") or {}
                if rec["t_first"] is None and (d.get("content") or d.get("reasoning_content")):
                    rec["t_first"] = time.time() - t0
            usage = c.get("usage") or usage
            timings = c.get("timings") or timings
    rec["t_done"] = time.time() - t0
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    rec["output_tokens"] = usage.get("completion_tokens")
    rec["cached"] = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or
                     (timings or {}).get("cache_n") or 0)
    if timings and timings.get("prompt_ms") is not None:
        rec["read_ms"] = timings["prompt_ms"]
    return rec


def run(a) -> int:
    base = a.url.rstrip("/")
    model, n_ctx = served_model(base, a.api_key)
    max_context = a.max_context or min(200000, int(0.8 * n_ctx) if n_ctx else 200000)
    p = plan(a.seed, a.clients, a.duration, tuple(int(x) for x in a.sys_tokens.split(",")),
             checks_per_hour=a.checks_per_hour, burst_clients=a.burst_clients)
    texts = Texts(p, Corpus(Path(a.corpus) if a.corpus else None))
    ban = ban_of(a.ban, model)
    lock = threading.Lock()
    reqs = []
    t0 = time.time()
    stop = threading.Event()

    def post(msgs, max_tokens, kind, c):
        body = {"model": model or "default", "max_tokens": max_tokens, "temperature": 0, "messages": msgs,
                "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        if ban:
            body["logit_bias"] = ban
        s = time.time() - t0
        try:
            rec = stream_request(base, body, a.timeout, a.api_key, t0)
        except (OSError, urllib.error.HTTPError, ValueError) as e:
            rec = {"t_send": s, "t_done": time.time() - t0, "error": str(e)[:200]}
        rec.update({"kind": kind, "client": c})
        with lock:
            reqs.append(rec)

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
            if sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN > max_context:
                restarts, start = restarts + 1, k        # compacted: a new conversation from this turn
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

    tiers, maya = [], [True]

    def poller():
        while not stop.is_set() and maya[0]:
            try:
                m = get_json(base + "/metrics", 10, a.api_key)
                tiers.append((time.time() - t0, (m.get("tiers") or {}).get("now")))
            except (OSError, ValueError):
                maya[0] = False                       # not a Maya server (or no /metrics): client-side only
            stop.wait(a.poll)

    threads = [threading.Thread(target=client, args=(c,), daemon=True) for c in range(a.clients)]
    threads += [threading.Thread(target=checker, daemon=True), threading.Thread(target=poller, daemon=True)]
    for th in threads:
        th.start()
    end = t0 + p["duration"]
    while time.time() < end and any(th.is_alive() for th in threads[:a.clients]):
        time.sleep(1)
    stop.set()
    for th in threads[:a.clients]:
        th.join(timeout=a.drain)                     # the answers in flight come back (they end after the window)
    records = []
    try:
        records = get_json(base + "/metrics?requests=all", 30, a.api_key).get("requests") or []
    except (OSError, ValueError):
        pass
    server = server_summary(records, tiers, t0, a.warmup, a.duration) if (records or tiers) else None
    with lock:
        done = sorted(reqs, key=lambda r: r["t_send"])
    summary = summarize(done, a.warmup, a.duration, server)
    out = {"label": a.label, "rep": a.rep, "seed": a.seed, "clients": a.clients, "duration": a.duration,
           "warmup": a.warmup, "started": t0, "model": model, "max_context": max_context, "url": base,
           "summary": summary, "requests": done,
           "server_records": [r for r in records if (r.get("time") or 0) >= t0],
           "tiers": [[round(t, 1), n] for t, n in tiers]}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"{summary['requests']} requests in the window, {summary['errors']} failed")
    return 0 if not summary["errors"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--clients", type=int, default=4, help="conversations at once (default 4)")
        s.add_argument("--duration", type=float, default=900, help="seconds of workload (default 900)")
        s.add_argument("--sys-tokens", default="30000,8000,4000", help="the shared system prompts' sizes in tokens")
        s.add_argument("--checks-per-hour", type=float, default=40)
        s.add_argument("--burst-clients", type=int, default=None, help="conversations starting over at each burst "
                                                                       "(default: half)")
    r = sub.choices["run"]
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--api-key", help="the server's API key, if it wants one")
    r.add_argument("--label", required=True, help="the setting's name in the comparison")
    r.add_argument("--rep", type=int, default=1)
    r.add_argument("--out", help="the run's JSON (every request, the summary)")
    r.add_argument("--warmup", type=float, default=180, help="seconds left out of the summary (default 180)")
    r.add_argument("--poll", type=float, default=10, help="seconds between two reads of a Maya server's /metrics")
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--drain", type=float, default=120, help="seconds to wait for the answers in flight at the end")
    r.add_argument("--ban", default="auto", help="end tokens banned so every answer writes its max_tokens: auto "
                                                 "(known by the model's name: GLM), none, or token ids")
    r.add_argument("--corpus", help="a folder of text files to cut the prompts from (default: this repository)")
    r.add_argument("--max-context", type=int, default=0,
                   help="a conversation estimated past this many tokens starts over from its latest turn (default: "
                        "80%% of the server's context, at most 200000)")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        p = plan(a.seed, a.clients, a.duration, tuple(int(x) for x in a.sys_tokens.split(",")),
                 checks_per_hour=a.checks_per_hour, burst_clients=a.burst_clients)
        for i, cl in enumerate(p["clients"]):
            fresh = [t["fresh"] for t in cl["turns"]]
            print(f"conversation {i}: system prompt {p['sys_tokens'][cl['sys']]} tokens, first {cl['first']}, "
                  f"{len(fresh)} turns at most, new tokens median {st.median(fresh)}, max {max(fresh)}")
        print(f"{len(p['checks'])} checks ({p['check_sys']}-token system prompt); bursts: " +
              ", ".join(f"{b['at']} s: conversations {b['clients']}" for b in p["bursts"]))
        return 0
    if a.cmd == "compare":
        print(compare([json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files]))
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
