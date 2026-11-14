"""Strict local HTTP regression check for GLM DFlash fair switching.

Targets an already-running Maya server with DFlash, ``--parallel N`` and a
positive MAYA_FAIR_SLICE_S. It never starts a server/model. It fills N server
FIFO places, queues one ``spec=1`` request, proves its own ``draft_n=0``, then
proves a later one-place probe's own DFlash drafts. Server FIFO is not native
engine Pending-queue coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

FAIR_SLICE = re.compile(r"\[maya\]\s+fair slice:")
NATIVE_CANCEL = re.compile(r"glm multi: request \d+ done \(cancel\):")


class CheckError(RuntimeError):
    pass

def request_body(model, label, budget, spec):
    return {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": budget,
        "ignore_eos": True,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      f"Synthetic DFlash switch check {label}. Write a long numbered list."}],
        "strata_tune": {"spec": spec},
    }

def headers(key, content=False):
    out = {"Content-Type": "application/json"} if content else {}
    if key:
        out["Authorization"] = f"Bearer {key}"
    return out

def get_json(url, timeout, key):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers(key)), timeout=timeout) as r:
        return json.loads(r.read())

def serving(status):
    n = (status.get("concurrency") or {}).get("serving")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise CheckError("/v1/status lacks concurrency.serving")
    return n

def status_counts(status):
    a = status.get("activity") or {}
    n, q = a.get("in_flight"), (a.get("queue") or {}).get("waiting")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (n, q)):
        raise CheckError("/v1/status activity counts are invalid")
    return n, q

def active_count(metrics):
    n = (metrics.get("live") or {}).get("active")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise CheckError("/metrics lacks live.active")
    return n

def slice_cancel_evidence(log, offset):
    try:
        with log.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            lines = list(f)
    except OSError as e:
        raise CheckError(f"cannot read --log-file: {e}") from e
    return any(FAIR_SLICE.search(line) for line in lines) and any(NATIVE_CANCEL.search(line) for line in lines)

@dataclass
class Stream:
    base: str
    body: dict
    timeout: float
    key: str | None
    label: str
    first: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    abandon: threading.Event = field(default_factory=threading.Event)
    tokens: int = 0
    saw_done: bool = False
    finish_reason: str | None = None
    usage: dict | None = None
    timings: dict | None = None
    http_status: int | None = None
    error: str | None = None
    _response: object | None = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name=f"dflash-{self.label}").start()

    def _run(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(self.body).encode(),
                                     headers=headers(self.key, True))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self._response, self.http_status = r, r.status
                for raw in r:
                    if self.abandon.is_set():
                        return
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        self.saw_done = True
                        return
                    try:
                        event = json.loads(data)
                    except ValueError:
                        self.error = "invalid SSE JSON"
                        return
                    if event.get("error"):
                        self.error = f"SSE error: {event['error']}"
                        return
                    if isinstance(event.get("usage"), dict):
                        self.usage = event["usage"]
                    if isinstance(event.get("timings"), dict):
                        self.timings = event["timings"]
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason") is not None:
                            self.finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            self.tokens += 1
                            self.first.set()
                if not self.abandon.is_set():
                    self.error = "SSE ended before [DONE]"
        except (OSError, urllib.error.HTTPError, ValueError) as e:
            if not self.abandon.is_set():
                self.error = str(e)
        finally:
            self.done.set()

    def close(self):
        self.abandon.set()
        if self._response is not None:
            self._response.close()

    def require_complete(self):
        budget = self.body["max_tokens"]
        completed = (self.http_status == 200 and self.saw_done and self.finish_reason == "length" and
                     isinstance(self.usage, dict) and self.usage.get("completion_tokens") == budget and
                     isinstance(self.timings, dict) and not self.error)
        if not completed:
            raise CheckError(f"{self.label} incomplete: http={self.http_status}, done={self.saw_done}, "
                             f"finish={self.finish_reason!r}, usage={self.usage!r}, timings={self.timings!r}, "
                             f"error={self.error!r}")

def wait_until(description, timeout, predicate):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise CheckError(f"timed out waiting for {description}")

def run(args):
    base = args.url.rstrip("/")
    log = Path(args.log_file)
    key = os.environ.get(args.api_key_env) or None
    if not log.is_file():
        raise CheckError("--log-file must name the current server log")
    offset = log.stat().st_size
    slots = serving(get_json(base + "/v1/status", args.timeout, key))
    streams = []

    def launch(label, budget, spec):
        stream = Stream(base, request_body(args.model, label, budget, spec), args.timeout, key, label)
        streams.append(stream)
        stream.start()
        return stream

    def ready(x):
        if not x.first.wait(args.timeout):
            raise CheckError(f"{x.label} produced no streamed token ({x.error or 'timeout'})")

    def idle():
        status = get_json(base + "/v1/status", args.timeout, key)
        metrics = get_json(base + "/metrics", args.timeout, key)
        return status_counts(status) == (0, 0) and active_count(metrics) == 0

    try:
        wait_until("an idle server before the check", args.timeout, idle)
        solo = launch("solo", 384, args.spec)
        ready(solo)
        followers = [launch(f"multi-{i}", 96, args.spec) for i in range(2, slots + 1)]
        for stream in followers:
            ready(stream)
        if solo.done.is_set():
            raise CheckError("solo finished before multi-request phase")
        wait_until(f"{slots} active requests", args.timeout,
                   lambda: active_count(get_json(base + "/metrics", args.timeout, key)) >= slots)
        queued = launch("queued-spec-off", 32, 1)
        wait_until("a queued N+1 request", args.timeout,
                   lambda: status_counts(get_json(base + "/v1/status", args.timeout, key))[1] >= 1)
        wait_until("fair-slice and native cancel evidence", args.timeout, lambda: slice_cancel_evidence(log, offset))
        for stream in followers:
            stream.close()
        if not queued.done.wait(args.timeout):
            raise CheckError("queued spec=1 request did not complete")
        queued.require_complete()
        if queued.timings.get("draft_n") != 0:
            raise CheckError(f"queued spec=1 did not preserve draft_n=0: {queued.timings!r}")
        if not solo.done.wait(args.timeout):
            raise CheckError("an original, non-cancelled stream did not complete after fair slicing")
        solo.require_complete()
        if not isinstance(solo.timings.get("draft_n"), int) or solo.timings["draft_n"] <= 0:
            raise CheckError(f"original stream has no DFlash drafts: {solo.timings!r}")

        stopped = launch("stop", 96, args.spec)
        ready(stopped)
        stopped.close()
        wait_until("disconnected stream slot release", args.timeout, lambda: idle() and stopped.done.is_set())
        probe = launch("post-stop-probe", 96, args.spec)
        if not probe.done.wait(args.timeout):
            raise CheckError("probe did not complete after disconnect")
        probe.require_complete()
        if not isinstance(probe.timings.get("draft_n"), int) or probe.timings["draft_n"] <= 0:
            raise CheckError(f"one-active probe has no DFlash drafts: {probe.timings!r}")
        wait_until("all test streams and server counters idle", args.timeout,
                   lambda: idle() and all(stream.done.is_set() for stream in streams))
        return {"serving": slots, "solo_draft_n": solo.timings["draft_n"],
                "queued": {"finish_reason": queued.finish_reason,
                                                "draft_n": queued.timings["draft_n"]},
                "fair_slice_native_cancel": True,
                "probe": {"finish_reason": probe.finish_reason, "draft_n": probe.timings["draft_n"]},
                "authenticated": bool(key)}
    finally:
        for stream in streams:
            stream.close()

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--log-file", required=True)
    p.add_argument("--api-key-env", default="MAYA_API_KEY", help="environment variable only; never a key value")
    p.add_argument("--model", default="glm-5.3-flash")
    p.add_argument("--spec", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120)
    a = p.parse_args(argv)
    if not 2 <= a.spec <= 16:
        p.error("--spec must be 2..16")
    if a.timeout <= 0:
        p.error("--timeout must be positive")
    try:
        print("PASS: " + json.dumps(run(a), sort_keys=True))
        return 0
    except CheckError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
