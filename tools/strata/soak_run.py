#!/usr/bin/env python3
"""tools/strata/soak_run.py - the B12.7 daily-driver soak (M5) and the tool-call suite, against the served engine.

    # the 50-case tool-call suite once (M5: 100% parse), OpenAI non-streamed and streamed
    python3 tools/strata/soak_run.py toolcalls --url http://127.0.0.1:8090 --stream both --report tc.json
    # the 8-hour scripted session: 200 requests, 2 reloads, memory watch
    python3 tools/strata/soak_run.py soak --url http://127.0.0.1:8090 --hours 8 --requests 200 \
        --reload-contexts 65536,32768 --server-pid $(pgrep -f 'serve.server' | head -1) --report soak.json

Server API (serve/server.py): POST /v1/chat/completions (OpenAI, stream or not), POST /v1/messages (Anthropic),
GET /health ({"status": "ok" | "reloading" | "updating"}), POST /api/context {"max_context": n} (reload the engine
with a new context; GET /api/context reports {"reload": {"state": ...}}).  --reload-cmd runs a shell command instead
(e.g. a systemctl restart), after which the soak waits for /health.

Tool-call checks (data/glm-dsa/toolcalls/*.json, tools/strata/make_toolcall_corpus.py):
  parse  (the M5 gate, 100%): every returned call names a declared tool, its arguments are a JSON object that
         satisfies the tool's schema (required keys, types, enums, nested objects and arrays), the number of calls
         is within the case's min/max, a Chinese case keeps CJK characters in a string argument, and an
         error-reply case does not repeat the failed call with the same arguments.
  match  (reported): the expected tool names, in any order, and the expected argument values (string compare
         case-insensitive).
The soak: requests drawn (seeded) from chat, code, tool-call (the 50 cases, with a tool result sent back for a final
answer), and long-document (--doc, about --doc-chars characters of data/glm-dsa/bench-corpus/prompt_32k.txt) groups,
spread over --hours (with --mcp-share, also requests that use the server's MCP tools, "strata_mcp": true); a
conversation thread states a codename at the start and asks for it after each reload (the
"reloads keep the conversation" check: the client resends the history, so the engine must answer it after a restart).
Watched: /health every --health-s (a failure outside a reload is a crash event), MemAvailable in /proc/meminfo
every 10 s (minimum reported; M5 needs >= 6 GiB), the server process (--server-pid; gone = a crash) and, with
--dmesg, kernel OOM-kill lines.  The report (JSON, plus markdown with --md) ends with the M5 verdict:
0 crashes, 0 OOM kills, tool-call parse 100%, every reload READY within --ready-limit-s (default 360) and keeping
the conversation, MemAvailable >= --min-mem-gib.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLCALLS = ROOT / "data" / "glm-dsa" / "toolcalls"
DOC = ROOT / "data" / "glm-dsa" / "bench-corpus" / "prompt_32k.txt"
CJK = re.compile(r"[㐀-鿿]")

CHAT = [
    "Give me three tips for keeping basil alive indoors.",
    "Explain the difference between a mutex and a semaphore in two paragraphs.",
    "Summarise the plot of a heist story you invent, in five sentences.",
    "What are the trade-offs of a four-day work week? Be concise.",
    "Write a short haiku about a slow train.",
    "用三句话解释一下什么是通货膨胀。",
    "How do I politely decline a meeting invitation? Give an example e-mail.",
    "List five common mistakes when learning to ride a bicycle.",
]
CODE = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two tests.",
    "Explain what this does and fix the bug: `for i in range(len(a)): if a[i] = 0: print(i)`",
    "Write a SQL query that returns the top 3 customers by total order value from orders(customer_id, value).",
    "Convert this to TypeScript with types: function add(a, b) { return a + b }",
    "Write a bash one-liner that counts lines in all .cpp files under src/.",
]


# ------------------------------------------------------------------------------------------------ schema checks
def check_value(v, schema: dict, path: str, errs: list[str]) -> None:
    t = schema.get("type")
    if "enum" in schema and v not in schema["enum"]:
        errs.append(f"{path}: {v!r} not in {schema['enum']}")
    if t == "object":
        if not isinstance(v, dict):
            errs.append(f"{path}: expected an object, got {type(v).__name__}")
            return
        for k in schema.get("required", []):
            if k not in v:
                errs.append(f"{path}.{k}: required key missing")
        props = schema.get("properties", {})
        for k, sub in v.items():
            if k in props:
                check_value(sub, props[k], f"{path}.{k}", errs)
    elif t == "array":
        if not isinstance(v, list):
            errs.append(f"{path}: expected an array, got {type(v).__name__}")
            return
        for i, sub in enumerate(v):
            check_value(sub, schema.get("items", {}), f"{path}[{i}]", errs)
    elif t == "string":
        if not isinstance(v, str):
            errs.append(f"{path}: expected a string, got {type(v).__name__}")
    elif t == "integer":
        if isinstance(v, bool) or not isinstance(v, int):
            errs.append(f"{path}: expected an integer, got {v!r}")
    elif t == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"{path}: expected a number, got {v!r}")
    elif t == "boolean":
        if not isinstance(v, bool):
            errs.append(f"{path}: expected a boolean, got {v!r}")


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def _eq(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < 1e-9
    return a == b


def check_case(case: dict, calls: list[tuple[str, str]], content: str) -> dict:
    """calls: (name, raw JSON arguments).  Returns {"parse", "match", "errors", "calls"}."""
    exp = case["expect"]
    tools = {t["function"]["name"]: t["function"] for t in case["request"]["tools"]}
    errs: list[str] = []
    parsed = []
    for name, raw in calls:
        if name not in tools:
            errs.append(f"call {name!r}: not a declared tool")
            continue
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            errs.append(f"call {name}: arguments are not JSON ({e.msg}): {str(raw)[:80]!r}")
            continue
        if not isinstance(args, dict):
            errs.append(f"call {name}: arguments are not an object")
            continue
        check_value(args, tools[name].get("parameters", {}), name, errs)
        parsed.append((name, args))
    n = len(calls)
    if n < exp.get("min_calls", 0):
        errs.append(f"{n} calls, at least {exp['min_calls']} expected")
    if n > exp.get("max_calls", 99):
        errs.append(f"{n} calls, at most {exp['max_calls']} expected")
    if exp.get("no_call") and n:
        errs.append("a call where none was expected")
    if exp.get("cjk_args") and parsed and not any(CJK.search(s) for _, a in parsed for s in _strings(a)):
        errs.append("no string argument keeps the Chinese text")
    rep = exp.get("must_not_repeat")
    if rep and any(nm == rep["name"] and a == rep["args"] for nm, a in parsed):
        errs.append("repeated the failed call with the same arguments")
    if n == 0 and exp.get("min_calls", 0) == 0 and not content.strip():
        errs.append("neither a call nor any text")
    match = True
    pool = list(parsed)
    for want in exp.get("calls", []):
        hit = None
        for i, (nm, a) in enumerate(pool):
            if nm == want["name"] and all(k in a and _eq(a[k], v) for k, v in want.get("args", {}).items()):
                hit = i
                break
        if hit is None:
            match = False
        else:
            pool.pop(hit)
    return {"parse": not errs, "match": match and not errs, "errors": errs,
            "calls": [{"name": nm, "args": a} for nm, a in parsed]}


# ------------------------------------------------------------------------------------------------ HTTP
class Client:
    def __init__(self, url: str, api_key: str | None = None, timeout: float = 1800.0, model: str = "strata"):
        self.url = url.rstrip("/")
        self.key = api_key
        self.timeout = timeout
        self.model = model

    def _req(self, path: str, body: dict | None = None, method: str | None = None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
            h["x-api-key"] = self.key
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        return urllib.request.Request(self.url + path, data=data, headers=h, method=method)

    def get(self, path: str, timeout: float = 10.0) -> dict:
        with urllib.request.urlopen(self._req(path), timeout=timeout) as r:
            return json.loads(r.read().decode())

    def post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        with urllib.request.urlopen(self._req(path, body), timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def chat(self, messages: list, tools: list | None = None, tool_choice=None, stream: bool = False,
             max_tokens: int = 1024, extra: dict | None = None) -> tuple[str, list[tuple[str, str]], dict]:
        """OpenAI chat: (content, [(name, raw arguments)], info)."""
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                      "stream": stream, **(extra or {})}
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if not stream:
            r = self.post("/v1/chat/completions", body)
            msg = r["choices"][0]["message"]
            calls = [(c["function"]["name"], c["function"].get("arguments", "")) for c in msg.get("tool_calls") or []]
            return msg.get("content") or "", calls, {"finish": r["choices"][0].get("finish_reason"),
                                                     "usage": r.get("usage")}
        content, parts, finish = [], {}, None
        with urllib.request.urlopen(self._req("/v1/chat/completions", body), timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if "error" in ev:
                    raise RuntimeError(f"stream error: {ev['error']}")
                for ch in ev.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(d["content"])
                    for tc in d.get("tool_calls") or []:
                        slot = parts.setdefault(tc.get("index", 0), {"name": "", "args": []})
                        f = tc.get("function") or {}
                        if f.get("name"):
                            slot["name"] = f["name"]
                        if f.get("arguments"):
                            slot["args"].append(f["arguments"])
                    finish = ch.get("finish_reason") or finish
        calls = [(parts[i]["name"], "".join(parts[i]["args"])) for i in sorted(parts)]
        return "".join(content), calls, {"finish": finish}

    def messages(self, messages: list, tools: list | None = None, max_tokens: int = 1024):
        """Anthropic Messages, non-streamed: (content, [(name, arguments as JSON)], info)."""
        atools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                   "input_schema": t["function"].get("parameters", {})} for t in tools or []]
        body: dict = {"model": self.model, "max_tokens": max_tokens, "messages": messages, "temperature": 0.0}
        if atools:
            body["tools"] = atools
        r = self.post("/v1/messages", body)
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        calls = [(b["name"], json.dumps(b.get("input", {}), ensure_ascii=False))
                 for b in r.get("content", []) if b.get("type") == "tool_use"]
        return text, calls, {"finish": r.get("stop_reason")}


def load_cases(d: pathlib.Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def run_case(cl: Client, case: dict, stream: bool) -> dict:
    req = case["request"]
    t0 = time.time()
    try:
        if case.get("api") == "anthropic":
            content, calls, info = cl.messages(req["messages"], req["tools"])
        else:
            content, calls, info = cl.chat(req["messages"], req["tools"], req.get("tool_choice"), stream=stream)
    except (urllib.error.URLError, OSError, RuntimeError, KeyError, ValueError) as e:
        return {"id": case["id"], "group": case["group"], "stream": stream, "parse": False, "match": False,
                "errors": [f"request failed: {e}"], "transport_error": True, "s": time.time() - t0}
    r = check_case(case, calls, content)
    r.update({"id": case["id"], "group": case["group"], "stream": stream and case.get("api") != "anthropic",
              "s": round(time.time() - t0, 3), "finish": info.get("finish")})
    return r


def summarize_toolcalls(results: list[dict]) -> dict:
    n = len(results)
    parse = sum(r["parse"] for r in results)
    match = sum(r["match"] for r in results)
    groups: dict = {}
    for r in results:
        g = groups.setdefault(r["group"], {"n": 0, "parse": 0, "match": 0})
        g["n"] += 1
        g["parse"] += r["parse"]
        g["match"] += r["match"]
    return {"n": n, "parse": parse, "match": match, "parse_pct": 100.0 * parse / max(n, 1),
            "match_pct": 100.0 * match / max(n, 1), "groups": groups,
            "failures": [{"id": r["id"], "stream": r["stream"], "errors": r["errors"]} for r in results if not r["parse"]]}


# ------------------------------------------------------------------------------------------------ soak
class Watch(threading.Thread):
    """Health, memory, the server process and (optionally) kernel OOM kills, until stop."""

    def __init__(self, cl: Client, health_s: float, pid: int | None, dmesg: bool):
        super().__init__(daemon=True)
        self.cl, self.health_s, self.pid, self.dmesg = cl, health_s, pid, dmesg
        self.stop = threading.Event()
        self.reloading = threading.Event()
        self.lock = threading.Lock()
        self.down_events: list[dict] = []
        self.mem_min_kib: int | None = None
        self.samples = 0
        self.pid_gone = False
        self._down = False
        self._dmesg0 = self._oom_lines() if dmesg else 0

    @staticmethod
    def mem_available_kib() -> int | None:
        try:
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1])
        except OSError:
            return None
        return None

    @staticmethod
    def _oom_lines() -> int:
        try:
            out = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return 0
        return sum(1 for ln in out.splitlines() if "Out of memory" in ln or "oom-kill" in ln)

    def oom_kills(self) -> int | None:
        return self._oom_lines() - self._dmesg0 if self.dmesg else None

    def note_down(self, why: str) -> None:
        with self.lock:
            if not self._down and not self.reloading.is_set():
                self.down_events.append({"t": time.time(), "why": why})
            self._down = True

    def note_up(self) -> None:
        with self.lock:
            self._down = False

    def run(self) -> None:
        last_mem = 0.0
        while not self.stop.is_set():
            now = time.time()
            if now - last_mem >= 10.0:
                m = self.mem_available_kib()
                if m is not None:
                    self.mem_min_kib = m if self.mem_min_kib is None else min(self.mem_min_kib, m)
                    self.samples += 1
                last_mem = now
            try:
                st = self.cl.get("/health", timeout=10.0).get("status")
                if st in ("ok", "reloading", "updating"):
                    self.note_up()
                else:
                    self.note_down(f"/health status {st!r}")
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.note_down(f"/health: {e}")
            if self.pid and not self.pid_gone:
                try:
                    os.kill(self.pid, 0)
                except ProcessLookupError:
                    self.pid_gone = True
                    self.note_down(f"server process {self.pid} is gone")
                except PermissionError:
                    pass
            self.stop.wait(self.health_s)


def wait_ready(cl: Client, limit_s: float, started: float) -> float | None:
    """Seconds from `started` until /health is ok and no reload is running; None after limit_s."""
    while time.time() - started < limit_s:
        try:
            if cl.get("/health").get("status") == "ok":
                try:
                    st = (cl.get("/api/context").get("reload") or {}).get("state")
                except (urllib.error.URLError, OSError, ValueError):
                    st = None
                if st not in ("waiting", "running"):
                    return time.time() - started
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return None


def reload_once(cl: Client, w: Watch, ctx: int | None, cmd: str | None, limit_s: float) -> dict:
    w.reloading.set()
    t0 = time.time()
    try:
        if cmd:
            subprocess.run(cmd, shell=True, check=False, timeout=limit_s)
        else:
            cl.post("/api/context", {"max_context": int(ctx)}, timeout=60.0)
        time.sleep(1.0)
        ready = wait_ready(cl, limit_s, t0)
    except (urllib.error.URLError, OSError, ValueError, subprocess.SubprocessError) as e:
        ready = None
        return {"context": ctx, "cmd": cmd, "ready_s": None, "error": str(e)}
    finally:
        w.reloading.clear()
    return {"context": ctx, "cmd": cmd, "ready_s": None if ready is None else round(ready, 1)}


def build_schedule(n: int, cases: list[dict], rng: random.Random, doc_share: float,
                   mcp_share: float = 0.0) -> list[tuple[str, object]]:
    kinds = []
    tc = list(cases)
    rng.shuffle(tc)
    n_doc = max(1, int(round(n * doc_share))) if doc_share > 0 else 0
    n_mcp = max(1, int(round(n * mcp_share))) if mcp_share > 0 else 0
    n_tool = min(len(tc), max(1, n // 4))
    n_rest = n - n_doc - n_tool - n_mcp
    kinds += [("doc", None)] * n_doc + [("tool", c) for c in tc[:n_tool]] + [("mcp", i) for i in range(n_mcp)]
    for i in range(max(n_rest, 0)):
        kinds.append(("chat", rng.choice(CHAT)) if i % 2 == 0 else ("code", rng.choice(CODE)))
    rng.shuffle(kinds)
    return kinds[:n]


def soak(a, cl: Client) -> dict:
    rng = random.Random(a.seed)
    cases = load_cases(pathlib.Path(a.toolcalls))
    sched = build_schedule(a.requests, cases, rng, a.doc_share, a.mcp_share)
    doc_text = pathlib.Path(a.doc).read_text(encoding="utf-8") if a.doc_share > 0 else ""
    reloads = [int(x) for x in a.reload_contexts.split(",") if x] if a.reload_contexts else []
    n_reload = len(reloads) if reloads else (a.reloads if a.reload_cmd else 0)
    reload_at = {int(len(sched) * (i + 1) / (n_reload + 1)) for i in range(n_reload)}
    duration = a.hours * 3600.0 if a.duration_s is None else a.duration_s
    gap = duration / max(len(sched), 1)
    w = Watch(cl, a.health_s, a.server_pid, a.dmesg)
    w.start()
    codename = rng.choice(["MARIGOLD", "BASALT", "PELICAN", "JUNIPER", "ORCHID"]) + "-" + str(rng.randint(10, 99))
    convo = [{"role": "user", "content": f"Remember this: the project codename is {codename}. Reply with OK."}]
    results, reload_log, t_start = [], [], time.time()

    def one(kind: str, payload) -> dict:
        t0 = time.time()
        rec: dict = {"kind": kind, "t": round(t0 - t_start, 1)}
        try:
            if kind == "tool":
                r = run_case(cl, payload, stream=rng.random() < 0.5)
                rec.update({"id": r["id"], "parse": r["parse"], "match": r["match"], "errors": r["errors"]})
                if r.get("transport_error"):
                    raise ConnectionError(r["errors"][0])
                if r["parse"] and r["calls"] and payload.get("api") != "anthropic":   # the tool result, the answer
                    req = payload["request"]
                    msgs = list(req["messages"]) + [{"role": "assistant", "content": "", "tool_calls": [
                        {"id": f"c{i}", "type": "function", "function": {"name": c["name"], "arguments":
                                                                          json.dumps(c["args"], ensure_ascii=False)}}
                        for i, c in enumerate(r["calls"])]}]
                    msgs += [{"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"ok": True, "result": 42})}
                             for i in range(len(r["calls"]))]
                    content, _, _ = cl.chat(msgs, req["tools"], max_tokens=512)
                    rec["final_chars"] = len(content)
            elif kind == "mcp":   # the server's own MCP tools (serve/mcp.py), e.g. serve/mcp_fake_server.py's echo
                word = f"soak-{payload}-{rng.randint(1000, 9999)}"
                content, _, _ = cl.chat([{"role": "user", "content": f"Use the echo tool to echo the text {word}, then "
                                                                     f"tell me what it returned."}],
                                        max_tokens=a.max_tokens, extra={"strata_mcp": True})
                rec.update({"chars": len(content), "mcp_echo": word in content})
            elif kind == "doc":
                off = rng.randint(0, max(0, len(doc_text) - a.doc_chars))
                q = "Question: which names appear most often in the text above? Answer in two sentences."
                content, _, info = cl.chat([{"role": "user", "content": doc_text[off:off + a.doc_chars] + "\n\n" + q}],
                                           max_tokens=256)
                rec.update({"chars": len(content), "usage": info.get("usage")})
            else:
                content, _, _ = cl.chat([{"role": "user", "content": payload}], max_tokens=a.max_tokens,
                                        stream=rng.random() < 0.5)
                rec["chars"] = len(content)
            rec["ok"] = True
            w.note_up()
        except (urllib.error.HTTPError) as e:
            rec.update({"ok": False, "error": f"HTTP {e.code}"})
            if e.code >= 500 and not w.reloading.is_set():
                w.note_down(f"HTTP {e.code} on {kind}")
        except (urllib.error.URLError, OSError, ValueError, KeyError, RuntimeError, ConnectionError) as e:
            rec.update({"ok": False, "error": str(e)[:200]})
            w.note_down(f"{kind} request: {e}")
        rec["s"] = round(time.time() - t0, 2)
        return rec

    def ask_codename(when: str) -> dict:
        try:
            content, _, _ = cl.chat(convo + [{"role": "user", "content": "What is the project codename? Answer with "
                                                                           "the codename only."}], max_tokens=64)
            return {"when": when, "kept": codename.lower() in content.lower(), "answer": content[:80]}
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            return {"when": when, "kept": False, "answer": f"error: {e}"}

    try:
        content, _, _ = cl.chat(convo, max_tokens=16)
        convo.append({"role": "assistant", "content": content or "OK"})
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        w.note_down(f"first request: {e}")
    memory = [ask_codename("start")]
    k_reload = 0
    for i, (kind, payload) in enumerate(sched):
        if i in reload_at:
            ctx = reloads[k_reload % len(reloads)] if reloads else None
            k_reload += 1
            rl = reload_once(cl, w, ctx, a.reload_cmd, a.ready_limit_s)
            reload_log.append(rl)
            memory.append(ask_codename(f"after reload {k_reload}"))
        target = t_start + gap * i
        if time.time() < target:
            time.sleep(target - time.time())
        results.append(one(kind, payload))
        if a.verbose:
            print(json.dumps(results[-1], ensure_ascii=False)[:200], flush=True)
    w.stop.set()
    w.join(timeout=30)
    tool = [r for r in results if r["kind"] == "tool"]
    lat = sorted(r["s"] for r in results if r.get("ok"))
    oom = w.oom_kills()
    rep = {"started": t_start, "duration_s": round(time.time() - t_start, 1), "requests": len(results),
           "ok": sum(1 for r in results if r.get("ok")), "by_kind": {}, "crashes": len(w.down_events),
           "down_events": w.down_events, "server_pid_gone": w.pid_gone, "oom_kills": oom,
           "mem_available_min_gib": None if w.mem_min_kib is None else round(w.mem_min_kib / 1048576, 2),
           "mem_samples": w.samples, "reloads": reload_log, "conversation": memory,
           "toolcall_parse_pct": 100.0 * sum(r.get("parse", False) for r in tool) / max(len(tool), 1),
           "toolcall_match_pct": 100.0 * sum(r.get("match", False) for r in tool) / max(len(tool), 1),
           "toolcall_n": len(tool), "latency_p50_s": lat[len(lat) // 2] if lat else None,
           "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None, "results": results}
    for r in results:
        k = rep["by_kind"].setdefault(r["kind"], {"n": 0, "ok": 0})
        k["n"] += 1
        k["ok"] += bool(r.get("ok"))
    crit = [
        ("crashes", rep["crashes"] == 0, f"{rep['crashes']} down events"),
        ("oom_kills", oom in (0, None), "not checked (--dmesg)" if oom is None else f"{oom}"),
        ("toolcall_parse", rep["toolcall_parse_pct"] == 100.0, f"{rep['toolcall_parse_pct']:.1f}% of {len(tool)}"),
        ("reloads", len(reload_log) == n_reload and all(r.get("ready_s") is not None and r["ready_s"] <= a.ready_limit_s
                                                       for r in reload_log),
         ", ".join(f"{r.get('ready_s')} s" for r in reload_log) or "none"),
        ("conversation_kept", all(m["kept"] for m in memory), f"{sum(m['kept'] for m in memory)}/{len(memory)}"),
        ("mem_available", rep["mem_available_min_gib"] is None or rep["mem_available_min_gib"] >= a.min_mem_gib,
         f"min {rep['mem_available_min_gib']} GiB (>= {a.min_mem_gib})"),
        ("requests_ok", rep["ok"] == rep["requests"], f"{rep['ok']}/{rep['requests']}"),
    ]
    rep["criteria"] = [{"id": c, "pass": bool(p), "detail": d} for c, p, d in crit]
    rep["verdict"] = "PASS" if all(p for _, p, _ in crit) else "FAIL"
    return rep


def soak_markdown(r: dict) -> str:
    lines = [f"## Soak {time.strftime('%Y-%m-%d %H:%M', time.gmtime(r['started']))} UTC: **{r['verdict']}**", "",
             f"{r['requests']} requests in {r['duration_s'] / 3600:.2f} h ({r['ok']} ok); latency p50 "
             f"{r['latency_p50_s']} s, p95 {r['latency_p95_s']} s; tool calls {r['toolcall_n']}, parse "
             f"{r['toolcall_parse_pct']:.1f}%, match {r['toolcall_match_pct']:.1f}%.", "",
             "| Criterion | Result | Detail |", "|---|---|---|"]
    for c in r["criteria"]:
        lines.append(f"| {c['id']} | {'pass' if c['pass'] else 'FAIL'} | {c['detail']} |")
    return "\n".join(lines) + "\n"


def toolcall_markdown(s: dict) -> str:
    lines = [f"## Tool-call suite: parse {s['parse']}/{s['n']} ({s['parse_pct']:.1f}%), match {s['match']}/{s['n']}",
             "", "| Group | n | parse | match |", "|---|---|---|---|"]
    for g, v in sorted(s["groups"].items()):
        lines.append(f"| {g} | {v['n']} | {v['parse']} | {v['match']} |")
    for f in s["failures"]:
        lines.append(f"- {f['id']} ({'stream' if f['stream'] else 'non-stream'}): {'; '.join(f['errors'])[:300]}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("toolcalls", "soak"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8090")
        p.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"))
        p.add_argument("--toolcalls", default=str(TOOLCALLS))
        p.add_argument("--timeout", type=float, default=1800.0)
        p.add_argument("--report", help="JSON report")
        p.add_argument("--md", help="markdown report")
    tc = sub.choices["toolcalls"]
    tc.add_argument("--stream", choices=("off", "on", "both"), default="both")
    tc.add_argument("--cases", help="comma list of case ids")
    sk = sub.choices["soak"]
    sk.add_argument("--hours", type=float, default=8.0)
    sk.add_argument("--duration-s", type=float, default=None, help="overrides --hours (tests)")
    sk.add_argument("--requests", type=int, default=200)
    sk.add_argument("--reload-contexts", default="65536,32768", help="POST /api/context sizes, one per reload")
    sk.add_argument("--reload-cmd", help="a shell command per reload instead of /api/context")
    sk.add_argument("--reloads", type=int, default=2, help="reload count with --reload-cmd")
    sk.add_argument("--ready-limit-s", type=float, default=360.0)
    sk.add_argument("--doc", default=str(DOC))
    sk.add_argument("--doc-chars", type=int, default=150000, help="about 32K tokens of the bench corpus")
    sk.add_argument("--doc-share", type=float, default=0.05)
    sk.add_argument("--mcp-share", type=float, default=0.0,
                    help="share of requests using the server's MCP tools (strata_mcp; needs an echo tool)")
    sk.add_argument("--max-tokens", type=int, default=1024)
    sk.add_argument("--health-s", type=float, default=15.0)
    sk.add_argument("--server-pid", type=int)
    sk.add_argument("--dmesg", action="store_true", help="count kernel OOM-kill lines (needs dmesg access)")
    sk.add_argument("--min-mem-gib", type=float, default=6.0)
    sk.add_argument("--seed", type=int, default=12)
    sk.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    cl = Client(a.url, a.api_key, a.timeout)
    if a.cmd == "toolcalls":
        cases = load_cases(pathlib.Path(a.toolcalls))
        if a.cases:
            keep = set(a.cases.split(","))
            cases = [c for c in cases if c["id"] in keep]
        if not cases:
            print("soak_run: no tool-call cases", file=sys.stderr)
            return 2
        modes = {"off": [False], "on": [True], "both": [False, True]}[a.stream]
        results = []
        for st in modes:
            for c in cases:
                if st and c.get("api") == "anthropic":
                    continue
                results.append(run_case(cl, c, st))
        s = summarize_toolcalls(results)
        s["results"] = results
        md = toolcall_markdown(s)
        print(md)
        if a.report:
            pathlib.Path(a.report).write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
        if a.md:
            pathlib.Path(a.md).write_text(md)
        return 0 if s["parse"] == s["n"] else 1
    r = soak(a, cl)
    md = soak_markdown(r)
    print(md)
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    if a.md:
        pathlib.Path(a.md).write_text(md)
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
