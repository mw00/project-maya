"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions and /v1/responses (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `maya-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.  DeepSeek-V4.1 (maya-ds4, `INFO vision=ds4`) runs its vision tower in the engine:
the server only resizes the pictures (serve/ds4_vision.py) and sends their pixels with `GENI`; so does
MiniMax-M3.1-Flash (maya-mm3 with --vision on, `INFO vision=mm3`, serve/mm3_vision.py).
Requests whose prompt plus max tokens exceed the engine's context are REJECTED with 400, never truncated.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import base64
import hashlib
import hmac
import codecs
import itertools
import json
import os
import queue
import re
import subprocess
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve import ds4_vision, gpu_map, loop_guard, mm3_vision, models, savings  # noqa: E402

ENGINE_VISION = (ds4_vision.Vision, mm3_vision.Vision)   # the engines that run their vision tower themselves
from serve.frontend import (CALL_START, MINIMAX_NS, ChatTemplate, Event, OutputParser,  # noqa: E402
                            EFFORT_WORDS, anthropic_to_messages, deepseek_effort_kwargs, deepseek_images, effort_kwargs,
                            effort_level, effort_name, literal_tags, mark_literals, media_of, openai_to_messages,
                            unmark_literals)
from serve.mcp import McpCancelled, hub_from_config  # noqa: E402
from serve.responses import (ResponseStore, input_items, input_page, normalize_request,  # noqa: E402
                             new_response, response_collect, response_events)
from serve.response_config import ENV_KEYS as RESPONSE_ENV_KEYS, limits_from_env  # noqa: E402
from serve.response_jobs import BackgroundJobs, JobError  # noqa: E402
from serve.response_store import StoreBusy  # noqa: E402
from serve.update import RESTART_EXIT, Updater  # noqa: E402
import maya_names  # noqa: E402  (tools/) the settings' names users see (MAYA_*) and the older ones the code reads
import maya_quants  # noqa: E402  (tools/) the quants' names users see (the model first: "GLM-5.3 · Maya-S")

try:                        # Project Maya's release (the dashboard's About; the engine's INFO carries none)
    MAYA_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip() or None
except OSError:
    MAYA_VERSION = None
def wait_gpu_release(pid: int, timeout: float = 60.0):
    """An engine that has ended still holds its GPU memory for a few seconds - the driver frees tens of GB of VRAM and
    its pinned RAM tier after the process is gone - and an engine started at once finds the cards full (the split
    search measures them before anything loads: the context reload's new engine got 0.09 GB and failed).  Waits until
    no GPU lists the process any more (NVML), or a few seconds where NVML can't say (AMD)."""
    try:
        from serve.telemetry import gpu_pids
    except ImportError:
        gpu_pids = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        pids = gpu_pids() if gpu_pids else None
        if pids is None:
            time.sleep(max(0.0, 5.0 - (time.time() - t0)))
            return
        if pid not in pids:
            if time.time() - t0 > 1:
                print(f"[maya] the GPUs freed the engine's memory in {time.time() - t0:.0f} s", flush=True)
            return
        time.sleep(0.5)
    print(f"[maya] the GPUs still list the old engine after {timeout:.0f} s; starting anyway", flush=True)


def nvml_pids(pid: int | None, listed: set) -> set:
    """The pids NVML may list a process the server started under (the GPU map): its own - or, when NVML lists
    processes but not that one, also those of every process it started in turn: an engine run through a launcher (a
    script that does not exec it, `nsys launch`, a debugger) holds its GPU memory in a child of the process the server
    started.  psutil finds them (setup installs it); without it, the pid alone."""
    if pid is None:
        return set()
    if pid in listed or not listed:
        return {pid}
    try:
        import psutil
        return {pid} | {c.pid for c in psutil.Process(pid).children(recursive=True)}
    except Exception:  # noqa: BLE001 - no psutil, or the process has just ended: the pid alone
        return {pid}


CONTEXT_MIN = 4096          # the dashboard's Context size: the smallest it offers ...
CONTEXT_FALLBACK_MAX = 262144   # ... and the largest when the model file does not say what it was trained for
# the largest context the dashboard offers without an "untested" warning, by engine: maya-ds4 its model's own 1M; the
# GLM engine's 256K, and maya-mm3's 256K (MiniMax-M3.1 is offered up to its 1M; past 256K it has not been run end to
# end yet)
CONTEXT_TESTED = {"ds4": 1048576, "mm3": 262144}
CONTEXT_TESTED_DEFAULT = 262144
# the engine log's lines a report carries (maya.py's --report reads the same ones)
REPORT_LINES = re.compile(r"glm fast:|glm prefill: (?:CUDA|HIP)|glm split|glm stat|glm slots|glm prefill: \d|ERR|error|"
                          r"failed|out of memory|WARNING", re.I)
IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.  5 s: drafted
# tokens come in bursts of 1-4 a step, and over 2 s the rate swung +-25% around the engine's steady one
RATE_WINDOW_S = 5.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# A request that waited this long in the FIFO gets a line in the server window when its turn comes: with nothing else
# logged, a queue that had stopped moving (and requests piling up behind it) was indistinguishable from a busy server.
QUEUE_LOG_S = 5.0
# A queued request's keep-alive while it waits for its first turn (#483): often enough for any stream-idle timeout,
# seldom enough to cost nothing (queue_ping_seconds: MAYA_QUEUE_PING_S sets another).
QUEUE_PING_S = 2.0
# The queue watchdog (Service.start_queue_watch): how often it looks, the age at which the first waiter means the
# queue has stopped moving, and how often it says so again while it stays that way.
QUEUE_TICK_S = 15.0
QUEUE_WEDGED_S = 120.0
QUEUE_WARN_S = 60.0


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        self.last_prompt: list[int] = []
        self.info = {"logit_bias": 1}   # it samples nothing, so every request's bias is as good as applied

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class FairLock:
    """The service's FIFO: waiters get the engine in the order they asked (threading.Lock promises no order), and the
    holder can see that someone is waiting (`waiting()`, the fair slices).  acquire/release/with as threading.Lock.

    With an engine that runs several conversations at once (READY ... multi=<n>) it has n places: a request takes
    one (`units=1`), and what needs the engine to itself - an image encode, a context reload, an update - takes them
    all (the default, and `with`).  Places go in arrival order: a waiter for all of them holds back the ones behind
    it, so it is not starved."""

    def __init__(self, capacity: int = 1):
        self._cv = threading.Condition(threading.Lock())
        self._queue: collections.deque = collections.deque()   # [token, units]
        self.capacity = max(1, int(capacity))
        self._used = 0

    def set_capacity(self, n: int):
        with self._cv:
            n = max(1, int(n))
            if n != self.capacity:
                print(f"[maya] the engine's places: {self.capacity} -> {n} ({self._used} in use)", flush=True)
            self.capacity = n
            # A place held for "all of them" was sized by the old capacity: a smaller engine cannot hold more than it
            # has, or the FIFO would stay jammed (waiting for a place that is never free) - the releases that follow
            # give their places back and are clamped in release().
            self._used = min(self._used, self.capacity)
            self._cv.notify_all()

    def _units(self, units):
        return self.capacity if units is None else max(1, min(int(units), self.capacity))

    def acquire(self, blocking: bool = True, timeout: float = -1, units: int | None = None) -> bool:
        k = self._units(units)
        with self._cv:
            if not self._queue and self._used + k <= self.capacity:
                self._used += k
                return True
            if not blocking:
                return False
        me = self.enqueue(units)
        if self.wait_turn(me, timeout):
            return True
        self.leave(me)
        return False

    def enqueue(self, units: int | None = None) -> object:
        """A place in line (wait_turn waits for it, leave gives it up): a waiter that wakes up now and then (to send
        a heartbeat) keeps its place."""
        me = [object(), self._units(units), time.monotonic()]    # (token, units, asked at: the queue watchdog's age)
        with self._cv:
            self._queue.append(me)
        return me

    def wait_turn(self, me, timeout: float = -1) -> bool:
        """Wait until `me` is first in line and its places are free, then hold them (True); False after `timeout`,
        still in line - and False at once when `me` was given up (leave) or is no longer in the line: a wedged FIFO
        (waiting for a place that never comes free) is what a queue that stops moving was."""
        with self._cv:
            end = None if timeout is None or timeout < 0 else time.monotonic() + timeout
            while True:
                if me not in self._queue:
                    return False
                if self._queue[0] is me and self._used + min(me[1], self.capacity) <= self.capacity:
                    self._queue.remove(me)
                    me[1] = min(me[1], self.capacity)
                    self._used += me[1]
                    self._cv.notify_all()                   # the next in line may fit in what is left
                    return True
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(left)

    def leave(self, me):
        with self._cv:
            if me in self._queue:
                self._queue.remove(me)
                self._cv.notify_all()                   # the next one may now be first in line

    def release(self, units: int | None = None):
        k = self._units(units)
        with self._cv:
            if self._used < k:
                # A capacity that shrank under a "all places" holder (the engine restarted with fewer conversations)
                # gives back what it has: raising here would leave the place held for ever - and the next release
                # would too, with every request behind it waiting for a place that never comes.
                print(f"[maya] a place was given back that is not held ({k} of {self._used} in use, capacity "
                      f"{self.capacity}): the engine's conversations changed while it ran", flush=True)
                k = self._used
            self._used -= k
            self._cv.notify_all()

    def locked(self) -> bool:
        return self._used > 0

    def waiting(self) -> int:
        return len(self._queue)

    def oldest_wait(self) -> float:
        """Seconds the request first in line has been waiting (0 when nobody waits): 0 while it is being served."""
        with self._cv:
            if not self._queue:
                return 0.0
            asked = self._queue[0][2] if len(self._queue[0]) > 2 else None
            return time.monotonic() - asked if isinstance(asked, float) else 0.0

    def held_by(self) -> str:
        """What has the engine (for a line when the queue stops moving): the places in use of its capacity."""
        with self._cv:
            return f"{self._used} of {self.capacity} place(s) in use"

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def queue_ping_seconds() -> float:
    """MAYA_QUEUE_PING_S: how often a request waiting for its first turn sends a keep-alive (QUEUE_PING_S)."""
    try:
        return max(0.01, float(os.environ.get("MAYA_QUEUE_PING_S", "") or QUEUE_PING_S))
    except ValueError:
        return QUEUE_PING_S


def fair_slice_seconds() -> float:
    """STRATA_FAIR_SLICE_S: a request that has been answering this long while others wait gives the engine to the next
    one and queues again behind them (0, the default: it keeps the engine to the end, as before)."""
    try:
        return max(0.0, float(os.environ.get("STRATA_FAIR_SLICE_S", "0") or 0))
    except ValueError:
        return 0.0


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""


class ContextOverflow(ValueError):
    """A request too long for the context (rejected, never truncated): a 400 with OpenAI's error code and wording -
    "maximum context length is N tokens" - which OpenAI-compatible clients (opencode, pi-ai, OpenClaw) recognise as a
    context overflow and answer by compacting the conversation (#432)."""
    code = "context_length_exceeded"


def error_body(e: Exception, kind: str = "invalid_request_error") -> dict:
    """An error response's body: OpenAI's shape, with the exception's `code` when it has one (ContextOverflow)."""
    err = {"type": kind, "message": str(e)}
    if getattr(e, "code", None):
        err["code"] = e.code
    return {"error": err}


# the engine's ERR when a route met a NaN or an infinity (Glm5Model::route_ok): it has dropped that conversation's state
NONFINITE = "non-finite"

# what Service._sliced yields when the loop guard ends an answer (a loop that came back): run() ends it, "stop"
LOOP_STOP = object()


def loop_guard_setting(cfg: dict | None, env=None) -> bool:
    """The loop guard (serve/loop_guard.py), on unless MAYA_LOOP_GUARD (STRATA_LOOP_GUARD) says 0 / off / false / no
    or the run config has "loop_guard": false; the environment, when set, wins over the config."""
    off = ("0", "off", "false", "no", "none")
    env = os.environ if env is None else env
    v = env.get("STRATA_LOOP_GUARD", env.get("MAYA_LOOP_GUARD"))   # (main: MAYA_* aliased, the config's env applied)
    if v is not None and str(v).strip():
        return str(v).strip().lower() not in off
    v = (cfg or {}).get("loop_guard")
    return str(v).strip().lower() not in off if isinstance(v, str) else v is not False and v != 0


def client_stop_strings(req: dict | None) -> list[str]:
    """A request's own stop strings: OpenAI's "stop" (a string or a list) or Anthropic's "stop_sequences" (from
    Strata #454 - they were ignored, so a client that relied on them got text past its marker)."""
    s = (req or {}).get("stop")
    if s is None:
        s = (req or {}).get("stop_sequences")
    if isinstance(s, str):
        s = [s]
    return [x for x in s if isinstance(x, str) and x][:16] if isinstance(s, list) else []


# the penalties' window at most: what the engines count (GLM's and DeepSeek's samplers keep the last 4096 tokens) -
# a request's penalty_last_n / repeat_last_n of -1 (llama.cpp's "the whole context") asks for this
PENALTY_WINDOW_MAX = 4096

# a request's sampling fields and their JSON type (True: an integer); llama.cpp's names for two of them
SAMPLING_TYPES = (("temperature", False), ("top_p", False), ("min_p", False), ("presence_penalty", False),
                  ("frequency_penalty", False), ("repetition_penalty", False), ("top_k", True),
                  ("penalty_last_n", True), ("seed", True))


def check_request_sampling(req: dict) -> None:
    """A request's sampling fields, read the way the run config's are, or a 400 (ValueError) that names the field.
    Some clients send numbers as strings ("temperature": "0.7"): read and used.  A value that cannot be read as its
    type ("temperature": "warm", "top_k": 40.5, "seed": true) is refused - before, it replaced the configured default
    and the engine line left it out, so the answer was silently greedy.  Ranges behave as before (a temperature <= 0 is
    greedy, top_k 0 or above 64 the widest list, a seed <= 0 none) and an absent or null field is not checked.
    llama.cpp's repeat_penalty / repeat_last_n are taken as repetition_penalty / penalty_last_n (those win); the window
    means what it means there (penalty_keys): 0 = no penalties, -1 = the whole context, below -1 refused."""
    if not isinstance(req, dict):
        return
    if req.get("repeat_penalty") is not None and req.get("repetition_penalty") is None:
        req["repetition_penalty"] = req["repeat_penalty"]
    window_key = "penalty_last_n"
    if req.get("penalty_last_n") is None and req.get("repeat_last_n") is not None:
        req["penalty_last_n"] = req["repeat_last_n"]
        window_key = "repeat_last_n"
    for key, integer in SAMPLING_TYPES:
        value = req.get(key)
        if value is None:
            continue
        what = f"{window_key if key == 'penalty_last_n' else key}: {'an integer' if integer else 'a number'}"
        if isinstance(value, str) and value.strip():
            try:
                num = float(value)
            except ValueError:
                raise ValueError(what) from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError(what)
            if integer:
                if num != int(num):
                    raise ValueError(what)
                num = int(num)
            elif re.fullmatch(r"\s*[+-]?\d+\s*", value):
                num = int(value)
            req[key] = value = num
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(what)
    if isinstance(req.get("penalty_last_n"), int) and req["penalty_last_n"] < -1:
        raise ValueError(f"{window_key}: -1 (the whole context), 0 (no penalties) or a positive number of tokens")
    guard = req.get("loop_guard")                   # the loop guard for this request only (serve/loop_guard.py)
    if guard is not None and not isinstance(guard, bool):
        if isinstance(guard, str) and guard.strip().lower() in ("true", "false", "on", "off", "1", "0"):
            req["loop_guard"] = guard.strip().lower() in ("true", "on", "1")
        elif isinstance(guard, int) and guard in (0, 1):
            req["loop_guard"] = bool(guard)
        else:
            raise ValueError("loop_guard: true or false")


def logit_bias_of(value, vocab_size: int | None = None) -> dict[int, float]:
    """OpenAI's logit_bias - {"<token id>": bias, ...}, bias -100..100 - as {id: bias}; {} for none.  A bias of -100
    bans the token.  Anything else is a 400 (ValueError) naming the field, before anything is sent."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('logit_bias: an object of {"<token id>": bias}')
    out = {}
    for token, bias in value.items():
        if isinstance(token, bool) or not re.fullmatch(r"[0-9]+", str(token)):
            raise ValueError("logit_bias: token ids are non-negative integers")
        tid = int(token)
        if tid > 2 ** 31 - 1 or (vocab_size and tid >= vocab_size):
            raise ValueError(f"logit_bias: token {tid} is outside the model's vocabulary")
        if isinstance(bias, str):
            try:
                bias = float(bias)
            except ValueError:
                raise ValueError("logit_bias: biases are numbers from -100 to 100") from None
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not -100 <= bias <= 100:
            raise ValueError("logit_bias: biases are numbers from -100 to 100")
        out[tid] = float(bias)
    if vocab_size and sum(b <= -100 for b in out.values()) >= vocab_size:
        raise ValueError("logit_bias: it cannot ban the whole vocabulary")
    return out


# A FROZEN engine (Strata #1317).  Silence alone proves nothing: on a slow PC the engine reads a long prompt chunk for
# minutes without a line.  But one that prints nothing for ENGINE_STALL_S and in that time uses no CPU, reads or writes
# no disk and leaves its GPUs idle is not slow, it is stuck (a deadlock, a driver stall), and waiting cannot help: it is
# ended, the request gets an error, and the next request starts it again.  One that is silent but working is never
# ended.  STRATA_ENGINE_STALL_S sets it (0 = off); it needs psutil (setup installs it), without it nothing is ended.
def _stall_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("STRATA_ENGINE_STALL_S", "90")))
    except ValueError:
        return 90.0


ENGINE_STALL_S = _stall_seconds()
# A request whose OWN lines stop (T / PP / DONE / ERR for its rid) is not slow, it is lost: the engine is working on
# other requests, so the engine-wide stall check above never fires, and this one waited for a line that no longer came
# while it held its place in the FIFO - four of those under --parallel and every request behind them waited for ever
# (nothing was logged, nothing expired).  After REQUEST_QUIET_S with no line of its own the request ends with an error
# (its place is given back) and the timeout is logged.  A prompt read reports PP lines as it goes, so quiet means lost,
# not busy; the waiting requests an engine admits later print "waits for a context" ... and 900 s is far past any read.
def _quiet_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("STRATA_REQUEST_QUIET_S", "900")))
    except ValueError:
        return 900.0


REQUEST_QUIET_S = _quiet_seconds()


# The longest a stopped request's own lines (T / PP / DONE) are waited for when it is drained after a cancel: a STOP
# an engine does not honour (or a rid it has already let go) would otherwise hold the FIFO place for ever.
def _drain_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("STRATA_DRAIN_S", "60")))
    except ValueError:
        return 60.0


DRAIN_S = _drain_seconds()
STALL_CPU_SHARE = 0.02      # the engine may use 2% of one core over the window and still count as stuck: its idle
                            # threads' polls (the GLM service threads sleep 2 ms at a time when quiet); work is cores
STALL_IO_EPS_B = 1 << 20    # bytes it may read or write in that window
STALL_GPU_BUSY = 5.0        # a GPU busier than this (%, NVML) is work


def engine_frozen(base: tuple[float, int], now: tuple[float, int], window_s: float) -> bool:
    """True when two (CPU seconds, disk bytes) samples of the engine process, window_s apart, show no work between."""
    return (now[0] - base[0]) <= max(0.5, STALL_CPU_SHARE * window_s) and (now[1] - base[1]) <= STALL_IO_EPS_B


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[maya] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line:
                    say("arena", f"[maya] loading the experts into RAM ({size}) and locking part of them for the GPU.\n"
                                 "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                                 "         Please wait and don't close this window; the browser opens when it is ready.")
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[maya] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[maya] filling the GPU's expert cache ({n}) ...")
                elif "session is up" in line:
                    say("up", "[maya] almost ready ...")
                elif "WARNING - " in line:   # the engine's own (e.g. Windows' commit limit capping the RAM tier)
                    print("[maya] WARNING: " + line.split("WARNING - ", 1)[1].strip(), flush=True)
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[maya] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """

    gpu_busy = None   # () -> bool: the engine's GPUs are working (the service's telemetry); kept across restarts
    multi = 0         # READY ... multi=<n> (set per start; the class default for engines built without __init__)

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None):
        self.spawn = (exe, list(args), cwd, log, env)   # to start it again after it died (issue #27)
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        self.log_path = log
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0
        if log:
            threading.Thread(target=narrate_start, args=(log, log_start, args, loading),
                             daemon=True).start()
        self.proc = subprocess.Popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        self.max_context = 0
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.can_pause = False           # ... and PAUSE: a STOP that keeps the state with what was written (fair slices)
        self.multi = 0                   # READY ... multi=<n>: n conversations at once (requests tagged rid=<id>)
        self._routes: dict[int, queue.Queue] = {}       # rid -> its lines (multi)
        self._routes_lock = threading.Lock()
        self._rids = itertools.count(1)
        self._tls = threading.local()                   # the request this thread is generating (pause())
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.vram = []                   # VRAM lines: what each part holds on its GPU (the Monitor's GPU map)
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total, reuse, ms) prompt progress while a prompt is read, from PP lines
        try:                             # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        for line in self.proc.stdout:
            if line.startswith("NOTE "):
                print("[maya] " + line[5:].strip(), flush=True)
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("VRAM "):
                part = gpu_map.parse(line)
                if part:
                    self.vram.append(part)
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                self.can_pause = "pause" in f[2:]
                self.multi = next((int(x[6:]) for x in f[2:] if x.startswith("multi=") and x[6:].isdigit()), 0)
                break
        loading.set()
        if self.max_context <= 0:
            raise RuntimeError("the engine exited before it was ready" + (f" (see {log})" if log else ""))
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        if log:
            self._context_cost(log_start)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _context_cost(self, offset: int):
        """What this start's context costs in VRAM, from the GLM engine's own start lines ("CUDA0 VRAM before the
        expert pool: ... state/KV (32768 ctx) 0.66"), summed over its GPUs: the dashboard scales it to the context
        sizes it offers (measured here, so it fits this model, this split and these cards)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                text = f.read(8 << 20).decode("utf-8", "replace")
        except OSError:
            return
        per = {}
        for m in re.finditer(r"\b((?:CUDA|HIP)\d+) VRAM before the expert pool:.*?state/KV \((\d+) ctx\) ([\d.]+)", text):
            per[m.group(1)] = (int(m.group(2)), float(m.group(3)))
        if per:
            self.info["kv_ctx"] = max(c for c, _ in per.values())
            self.info["kv_gb"] = round(sum(g for _, g in per.values()), 3)

    def set_context(self, n: int):
        """The next start's context: --max-context in the command (restart() runs it)."""
        exe, args, cwd, log, env = self.spawn
        args = list(args)
        if "--max-context" in args[:-1]:
            args[args.index("--max-context") + 1] = str(int(n))
        else:
            args += ["--max-context", str(int(n))]
        self.spawn = (exe, args, cwd, log, env)

    def stop(self, timeout: float = 60.0):
        """Ask the engine to end (QUIT: it frees its GPU memory itself), and end it if it does not in time - then wait
        until the GPUs have that memory back."""
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
                self.proc.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.ended = True
        wait_gpu_release(self.proc.pid)

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith("VRAM "):
                self._vram_line(line)
                continue
            if self.multi and self._route(line):
                continue
            self.lines.put(line)
        self.ended = True                               # its output closed: it is gone, even before the OS says so
        self.lines.put(None)
        with self._routes_lock:
            for q in self._routes.values():
                q.put(None)

    def _vram_line(self, line: str) -> None:
        """A VRAM line after the start (maya-ds4: its vision tower's room taken for a request's pictures, then given
        back): the Monitor's map of that GPU follows it."""
        part = gpu_map.parse(line)
        if part:
            self.vram = [part if v.get("dev") == part["dev"] else v for v in self.vram]

    _TAGGED = ("T ", "PP ", "DONE ", "ERR ")

    def _route(self, line: str) -> bool:
        """Several conversations at once: a request's own lines (T / PP / DONE / ERR <rid> ...) go to its queue, the
        engine's (STAT, NOTE) are handled here; True when the line was taken."""
        if line.startswith(self._TAGGED):
            f = line.split(None, 2)
            if len(f) >= 2 and f[1].isdigit():
                with self._routes_lock:
                    q = self._routes.get(int(f[1]))
                if q is not None:
                    q.put(line)
                    return True
                return line.startswith(("T ", "PP ", "DONE "))   # a request already gone: dropped
            return False
        if line.startswith("STAT "):
            self._parse_stat(line)
            return True
        if line.startswith("NOTE "):
            print("[maya] " + line[5:].strip(), flush=True)
            return True
        return False

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        tail = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            pass
        for line in reversed(tail.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/mw00/project-maya/issues.")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        return not getattr(self, "ended", False) and self.proc.poll() is None

    def _activity(self):
        """(CPU seconds, disk bytes read + written) the engine process has used so far; None without psutil."""
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            t, io = p.cpu_times(), p.io_counters()
            return t.user + t.system, io.read_bytes + io.write_bytes
        except Exception:  # noqa: BLE001 - no psutil or no access: nothing to decide on
            return None

    def _stall_check(self, heard: float, state: dict) -> str | None:
        """Called while the engine is silent (since `heard`, monotonic): a baseline soon after its last line, and at
        ENGINE_STALL_S of silence the comparison - no CPU, disk or GPU work in all that time: why it is ended."""
        t = time.monotonic()
        age = t - heard
        if "base" not in state:
            state["base"], state["t"], state["next"] = self._activity(), t, ENGINE_STALL_S
            return None
        if state["base"] is None or age < state["next"]:
            return None
        now = self._activity()
        try:
            gpu = bool(self.gpu_busy()) if self.gpu_busy is not None else False
        except Exception:  # noqa: BLE001 - a telemetry hiccup must not decide anything
            gpu = False
        if now is not None and not gpu and engine_frozen(state["base"], now, t - state["t"]):
            return (f"the engine said nothing for {age:.0f} s and used no CPU, disk or GPU in that time, so it was stuck "
                    "and has been ended (STRATA_ENGINE_STALL_S sets this, 0 = off)")
        state["base"], state["t"], state["next"] = now, t, age + ENGINE_STALL_S   # working: look again a window later
        return None

    def _end(self, why: str):
        """End a stuck engine now; the next request starts it again (Service.run)."""
        print(f"[maya] {why}", flush=True)
        self.ended = True
        try:
            self.proc.kill()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def exit_code(self):
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def restart(self):
        """Start the engine again (the same command) after it died; the new process has its own line queue."""
        try:
            self.proc.kill()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            pass
        wait_gpu_release(self.proc.pid)
        with self._routes_lock:                         # requests still waiting on the dead one: it is gone
            for q in self._routes.values():
                q.put(None)
        info = dict(self.info)
        # The rid counter and the routes go on across the start: __init__ makes a new counter (from 1) and a new
        # table, and a request that comes in while the old ones are still unwinding would then be given the number of
        # one of them - its lines went to THAT request's queue, the request whose number it took popped its entry, and
        # it waited for its own lines for ever, holding its place in the FIFO (four of those under --parallel and no
        # request starts any more, the queue filling with requests nothing ever takes: the engine restart was the
        # trigger, the silent wait the bug).
        rids, routes, routes_lock = self._rids, self._routes, self._routes_lock
        ctx = self.max_context
        self.ended = False
        try:
            self.__init__(*self.spawn)
        except Exception:
            # (#351: a start that fails - a GPU that fell off the bus - left max_context 0, so every later request
            #  failed its size check with "exceeds the context (0)" before ensure_engine could try again; the context
            #  stays what it was, the next request starts it again or says why it cannot)
            self.max_context = ctx
            self.info = {**info, **self.info}
            self._rids, self._routes, self._routes_lock = rids, routes, routes_lock
            raise
        self._rids, self._routes, self._routes_lock = rids, routes, routes_lock
        self.info = {**info, **self.info}

    @property
    def last(self) -> dict:
        """The last DONE's figures.  With several conversations at once (multi), the figures of the request this thread
        runs (#310): requests decoded together finish within a step of each other, and one shared field let a request
        report another's timings and cached tokens.  Each request thread has its own (an empty one until its DONE)."""
        if self.multi:
            t = self._tls
            if not hasattr(t, "last"):
                t.last = {}
            return t.last
        return self._last

    @last.setter
    def last(self, v: dict):
        if self.multi:
            self._tls.last = v
        else:
            self._last = v

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))

    def _parse_stat(self, line):
        """STAT key=value ... (the GLM engine, every few tokens): live expert-tier counters for the Monitor."""
        st = {}
        for kv in line.split()[1:]:
            k, _, v = kv.partition("=")
            try:
                st[k] = float(v)
            except ValueError:
                st[k] = v
        st["time"] = time.time()
        self.stat = st
        hist = getattr(self, "stat_history", None)
        if hist is None:
            hist = self.stat_history = collections.deque(maxlen=120)
        hist.append(st)

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = ""
        t = sampling.get("temperature")
        tb, te = sampling.get("_think_budget"), sampling.get("_think_end")   # the run config's (StrataService.run)
        think = f" think_budget={tb} think_end={te}" if isinstance(tb, int) and isinstance(te, int) and tb > 0 and \
            te >= 0 else ""
        # setup's calibration (tools/calibrate.py): engine settings for this request only, measured without a restart
        tune, tune_keys = sampling.get("strata_tune"), ""
        if isinstance(tune, dict):
            for k in ("pcie_frac", "spec_min_p", "spec_cum"):
                v = tune.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
                    tune_keys += f" {k}={float(v)!r}"
            sd = tune.get("suffix_draft")   # the DeepSeek engine's prompt lookup beside DSpark: off (0) / on (1)
            if isinstance(sd, int) and not isinstance(sd, bool) and sd in (0, 1):
                tune_keys += f" suffix_draft={sd}"
            ct = tune.get("cpu_threads")   # the GLM engine's CPU lane threads (fewer than it started with)
            if isinstance(ct, int) and not isinstance(ct, bool) and ct > 0:
                tune_keys += f" cpu_threads={ct}"
            rc = tune.get("read_chunks")   # ... and the pieces it reads an expert from the SSD in (x3)
            if isinstance(rc, int) and not isinstance(rc, bool) and 1 <= rc <= 16:
                tune_keys += f" read_chunks={rc}"
            mw = tune.get("miss_skip_w")   # the GLM engine's light-miss bar (STRATA_GLM_MISS_SKIP_W) for this request
            if isinstance(mw, (int, float)) and not isinstance(mw, bool) and 0.0 <= float(mw) <= 1.0:
                tune_keys += f" miss_skip_w={float(mw)!r}"
            mf = tune.get("miss_skip_from")   # ... and its first layer (STRATA_GLM_MISS_SKIP_FROM)
            if isinstance(mf, int) and not isinstance(mf, bool) and 0 <= mf <= 1000:
                tune_keys += f" miss_skip_from={mf}"
            sw = tune.get("spec")   # the GLM engine's DFlash decode: its verify window (the anchor and drafts), 2..16;
            if isinstance(sw, int) and not isinstance(sw, bool) and 1 <= sw <= 16:   # 1: no drafting (the token loop)
                tune_keys += f" spec={sw}"
            sn = tune.get("spec_n_min")   # ... drafts it keeps whatever their confidence
            if isinstance(sn, int) and not isinstance(sn, bool) and 0 <= sn <= 15:
                tune_keys += f" spec_n_min={sn}"
        bias = sampling.get("logit_bias")      # {id: bias}, checked by the handler (logit_bias_of)
        bias_key = (" logit_bias=" + ",".join(f"{int(i)}:{float(b):g}" for i, b in sorted(bias.items()))
                    if isinstance(bias, dict) and bias else "")
        penalty_keys = StrataEngine.penalty_keys(sampling)
        if isinstance(t, (int, float)) and not isinstance(t, bool) and float(t) <= 0.0:
            # temperature 0 is greedy: the engine reads temperature=0 as that whatever else the line says, and no other
            # sampler key goes (with only a config's top_p the engine took its sampled path: not deterministic) - but
            # the penalties do: they change the logits greedy picks from (#273: a greedy answer's loops)
            return (" temperature=0" + penalty_keys + think + tune_keys + StrataEngine.projection_key(sampling) +
                    bias_key)
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        keys += penalty_keys
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        keys += think + tune_keys
        return keys + StrataEngine.projection_key(sampling) + bias_key

    @staticmethod
    def penalty_keys(sampling: dict) -> str:
        """The repetition / frequency / presence penalties (llama.cpp's repeat_penalty and repeat_last_n are read as
        repetition_penalty and penalty_last_n): only the ones that change something, and their window with them.  ""
        when none does - the GEN line is then the one it was without them.  The window as llama.cpp's: none given = the
        last 64 tokens, 0 = no penalties, -1 = the whole context - the engines count at most PENALTY_WINDOW_MAX."""
        keys = ""
        pln = sampling.get("penalty_last_n")
        if isinstance(pln, int) and not isinstance(pln, bool) and pln == 0:
            return keys
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            elif pln == -1 and not isinstance(pln, bool):
                keys += f" penalty_last_n={PENALTY_WINDOW_MAX}"
            else:
                keys += " penalty_last_n=64"
        return keys

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1`: the experimental-speed-projection control vector for this request, when the engine was
        started with one (--control-vector-scaled; an engine without one ignores the key).  Absent = on."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        if self.multi:
            yield from self._generate_multi(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress = None
        self.prefill_tok_s_mean = None
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        head = f"GENI {int(max_new)}{self.sampling_keys(sampling or {})} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.sampling_keys(sampling or {})}"
        try:
            self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        heard, stall = time.monotonic(), {}               # the last line's time; the stall watchdog's samples
        try:
            while True:
                try:
                    line = self.lines.get(timeout=10)
                except queue.Empty:
                    if cancel.is_set():
                        return
                    if ENGINE_STALL_S > 0:
                        why = self._stall_check(heard, stall)
                        if why:
                            done = True                   # no STOP and no drain: nothing is listening
                            self._end(why)
                            raise EngineDied(why)
                    yield None
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()
                stall.clear()
                if line.startswith("T "):
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        # (read, total, reuse, ms): prompt progress, one per chunk: also a heartbeat (the
                        # lines reset the 10 s wait, so without this a long prompt would send no keep-alives at all).
                        # `reuse` (engines that print the field): what the conversation cache already held,
                        # `ms` the prompt read's own clock - both go to the stream's prompt_progress
                        self.progress = (int(f[1]), int(f[2]),
                                         int(f[5]) if len(f) >= 6 and f[5].isdigit() else 0,
                                         float(f[3]) if len(f) >= 4 else 0.0)
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("STAT "):
                    self._parse_stat(line)
                elif line.startswith("NOTE "):                # what happens to the kept conversations
                    print("[maya] " + line[5:].strip(), flush=True)
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    self._done_after_err()        # (maya-ds4 / maya-mm3 end the request with a DONE after it)
                    raise ValueError(line[4:].strip())
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # Bounded like the multi one (see DRAIN_S): a STOP the engine does not honour must not hold the
                # request's place in the FIFO for ever.
                end = time.monotonic() + DRAIN_S
                while True:
                    try:
                        line = self.lines.get(timeout=max(0.05, end - time.monotonic()))
                    except queue.Empty:
                        print(f"[maya] stopped, and the engine did not say the request had ended within {DRAIN_S:.0f} s "
                              "(STRATA_DRAIN_S) - going on; the engine may still be writing it", flush=True)
                        break
                    if line is None:
                        self.lines.put(None)                # (the engine is gone: the next reader hears it too)
                        break
                    if line.startswith("ERR"):
                        self._done_after_err()
                        break
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        break

    # How long, after an ERR, the reader waits for the DONE that may follow it: maya-ds4 and maya-mm3 print one right
    # after (in the same flush), the GLM engine none for most errors - so a short wait, not DRAIN_S.
    ERR_DONE_S = 0.5

    def _done_after_err(self):
        """After an ERR (single stream): take the DONE the engine may print after it, so the next request does not read
        that stale DONE as its own end (it returned an empty answer).  STAT / NOTE lines meanwhile are handled; the
        wait ends at another line, at the engine's end or after ERR_DONE_S."""
        end = time.monotonic() + self.ERR_DONE_S
        while True:
            try:
                line = self.lines.get(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                return
            if line is None:
                self.lines.put(None)
                return
            if line.startswith("DONE"):
                return
            if line.startswith("STAT "):
                self._parse_stat(line)
            elif line.startswith("NOTE "):
                print("[maya] " + line[5:].strip(), flush=True)
            else:
                return                                  # (not this request's end: dropped, as before)

    def command(self, line: str, expect: str, timeout: float = 60.0) -> str:
        """One control line between requests (the GLM engine's VLEND / VRECLAIM) -> its answer line; ValueError on
        the engine's ERR or no answer.  The caller holds the service's FIFO, so no request is reading the lines."""
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except OSError:
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        end = time.time() + timeout
        while True:
            try:
                got = self.lines.get(timeout=max(0.1, end - time.time()))
            except queue.Empty:
                raise ValueError(f"the engine did not answer {line.split()[0]}") from None
            if got is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            if got.startswith(expect):
                return got.strip()
            if got.startswith("ERR"):
                raise ValueError(got[4:].strip())

    def wrap(self):
        """The server is stopping: a request still thinking closes its reasoning at the next token and answers."""
        try:
            self.proc.stdin.write("WRAP\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def _generate_multi(self, ids, max_new, sampling, cancel, embeddings=None):
        """generate() on an engine that runs several conversations: this request's GEN carries rid=<id>, its lines come
        back through its own queue (_route), and STOP <id> ends it alone."""
        rid = next(self._rids)
        q: queue.Queue = queue.Queue()
        with self._routes_lock:
            self._routes[rid] = q
        self._tls.rid = rid
        self.progress = None
        self.prefill_tok_s_mean = None
        keys = self.sampling_keys(sampling or {}) + f" rid={rid}"
        head = f"GENI {int(max_new)}{keys} {embeddings}" if embeddings else f"GEN {int(max_new)}{keys}"
        done = False
        try:
            try:
                self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
                self.proc.stdin.flush()
            except OSError:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
            heard, stall = time.monotonic(), {}
            # how long a wait before the checks (the engine-wide stall watchdog, this request's own quiet): 10 s, or
            # less when STRATA_REQUEST_QUIET_S is set low (so a test - or a tuned server - sees it in its own time)
            poll_s = 10.0 if REQUEST_QUIET_S <= 0 else min(10.0, max(0.2, REQUEST_QUIET_S / 2))
            while True:
                try:
                    line = q.get(timeout=poll_s)
                except queue.Empty:
                    if cancel.is_set():
                        return
                    if ENGINE_STALL_S > 0:
                        why = self._stall_check(heard, stall)
                        if why:
                            done = True
                            self._end(why)
                            raise EngineDied(why)
                    quiet = time.monotonic() - heard
                    if REQUEST_QUIET_S > 0 and quiet >= REQUEST_QUIET_S:
                        # The engine is alive (the other requests' lines reach them), so silence for this rid is this
                        # request's loss: end it here rather than hold the FIFO place for a line that is not coming.
                        print(f"[maya] request {rid}: the engine has said nothing of it for {quiet:.0f} s - stopping it "
                              f"and giving its place in the queue back (STRATA_REQUEST_QUIET_S={REQUEST_QUIET_S:.0f})",
                              flush=True)
                        try:
                            self.proc.stdin.write(f"STOP {rid}\n")
                            self.proc.stdin.flush()
                        except OSError:
                            pass
                        done = True                        # nothing of its own to drain: later lines are dropped
                        raise ValueError(f"the engine never answered this request ({quiet:.0f} s without a line of "
                                         "its own); it was stopped")
                    yield None
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()
                stall.clear()
                f = line.split()
                if line.startswith("T "):
                    if cancel.is_set():
                        return
                    yield int(f[2])
                elif line.startswith("PP "):
                    if len(f) >= 4 and f[2].isdigit() and f[3].isdigit():
                        self.progress = (int(f[2]), int(f[3]),
                                         int(f[6]) if len(f) >= 7 and f[6].isdigit() else 0,
                                         float(f[4]) if len(f) >= 5 else 0.0)
                        self.prefill_tok_s_mean = float(f[5]) if len(f) >= 6 else None
                    if cancel.is_set():
                        return
                    yield None
                elif line.startswith("DONE "):
                    self._parse_done("DONE " + line.split(None, 2)[2])
                    done = True
                    return
                elif line.startswith("ERR "):
                    done = True
                    raise ValueError(line.split(None, 2)[2].strip() if len(f) > 2 else "engine error")
        finally:
            if not done:                                  # the consumer stopped early: stop this request, drain it
                try:
                    self.proc.stdin.write(f"STOP {rid}\n")
                    self.proc.stdin.flush()
                except OSError:
                    pass
                # The drain is bounded: an engine that does not honour the STOP (or whose DONE for this rid was
                # already dropped) left this loop waiting for a line that never came, holding the request's place in
                # the FIFO for ever.  Past DRAIN_S: say so, drop the route, and let the next request run.
                end = time.monotonic() + DRAIN_S
                while True:
                    try:
                        line = q.get(timeout=max(0.05, end - time.monotonic()))
                    except queue.Empty:
                        print(f"[maya] request {rid}: stopped, and the engine did not say it had ended within "
                              f"{DRAIN_S:.0f} s (STRATA_DRAIN_S) - going on; the engine may still be writing it",
                              flush=True)
                        break
                    if line is None or line.startswith("ERR "):
                        break
                    if line.startswith("DONE "):
                        self._parse_done("DONE " + line.split(None, 2)[2])
                        break
            with self._routes_lock:
                if self._routes.get(rid) is q:                # (a restarted engine's table may hold another one)
                    del self._routes[rid]
            self._tls.rid = None

    def pause(self, keep: bool = False):
        """End the request in flight at its next step and keep the conversation's state with what it wrote (the
        fair slices: prompt + written tokens then continues with one token to read).  The tokens it still writes
        before its DONE come as usual.  keep (a request's first slice): the engine first sets its prompt's state
        aside for the chat's next turn, whose history does not repeat the answer as written (no reasoning)."""
        rid = getattr(getattr(self, "_tls", None), "rid", None) if self.multi else None
        try:
            if rid is not None:
                self.proc.stdin.write(f"PAUSE {rid}{' keep' if keep else ''}\n")
            else:
                self.proc.stdin.write("PAUSE keep\n" if keep else "PAUSE\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def stop(self):
        """End the request in flight at its next step (its answer is cut there)."""
        if self.can_stop:
            try:
                self.proc.stdin.write("STOP\n")
                self.proc.stdin.flush()
            except OSError:
                pass

    def close(self):
        """QUIT, then wait for the engine to end: with kept conversations (STRATA_GLM_SLOT_KEEP) it first writes the
        one it holds, so it gets STRATA_ENGINE_QUIT_S seconds (50; llama-swap's unloadTimeout should exceed it)."""
        wait = float(os.environ.get("STRATA_ENGINE_QUIT_S", "50") or 50)
        try:
            if self.can_stop:                           # a request in flight ends at its next step first: QUIT
                self.proc.stdin.write("STOP\n")         # alone lets it run to max_tokens before the engine ends
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            end = time.monotonic() + wait
            src = getattr(self, "lines", None)          # after start-up the pump thread owns stdout: read its queue
            while self.proc.poll() is None and time.monotonic() < end:
                if src is not None:
                    try:
                        line = src.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
                    except queue.Empty:
                        continue
                else:
                    line = self.proc.stdout.readline()
                if not line:
                    break
                if line.startswith("NOTE "):
                    print("[maya] " + line[5:].strip(), flush=True)
            self.proc.wait(timeout=max(0.1, end - time.monotonic()))
        except Exception:
            print(f"[maya] the engine did not end within {wait:.0f} s: stopped", flush=True)
            self.proc.kill()


class Vision:
    """The resident image encoder: `maya-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    and `VID <video> <out>` lines and writes each picture's embeddings (a video: one record per frame); results
    are cached by the source's hash, so a conversation that sends the same picture again (every turn, with most
    clients) encodes it once.

    On demand (start=False, `lend`/`reclaim` set - the GLM engine's VLEND / VRECLAIM): no process until a picture
    needs encoding; the engine first empties the tail of its expert cache on the first GPU and frees it, the encoder
    starts in that memory, and once the request's pictures are encoded (`active()`) it ends and the engine takes the
    memory back.  The decode keeps the whole cache between pictures (~18% faster on Mercury than with the encoder
    resident), for a second or two of start-up per request with new pictures."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None, start: bool = True):
        args = [cfg["exe"], "--mmproj", cfg["mmproj"], "--model", cfg["model"]]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        if cfg.get("no_flash_attn"):   # GLM-5.3's encoder on a GPU: full-precision attention (cosine 0.9997 vs 0.999)
            args.append("--no-flash-attn")
        self.args, self.log, self.env = args, log, env
        self.dir = Path(tempfile.mkdtemp(prefix="maya-vision-"))
        self.proc = None
        self.lend = self.reclaim = None   # on demand: callables (the engine's VLEND / VRECLAIM), wired by main()
        self.owed = False                 # the engine's memory is not back yet (a reclaim that kept failing)
        self.lock = threading.RLock()
        self.cache: dict[str, tuple[Path, int]] = {}
        if start:
            self._start()

    def _start(self):
        if self.lend is not None:
            self.settle()
            self.lend()
        try:
            self.proc = subprocess.Popen(self.args + (["--no-warmup"] if self.lend is not None else []),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log or subprocess.DEVNULL, text=True, encoding="utf-8",
                                         bufsize=1, env=self.env)
            line = self.proc.stdout.readline()
            if not line.startswith("READY"):
                raise RuntimeError("the vision encoder did not start: " + line.strip())
        except Exception:
            self._stop()
            raise

    def _stop(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.reclaim is not None:
            self.owed = True
            for _ in range(40):          # the driver frees an ended process's memory within moments
                if self.settle():
                    return
                time.sleep(0.25)
            print("[maya] the vision encoder's GPU memory is not back yet: prompts are read token by token until "
                  "it is", flush=True)

    def settle(self) -> bool:
        """The engine takes back the memory an earlier encode borrowed (true when nothing is owed)."""
        if not self.owed:
            return True
        try:
            self.reclaim()
            self.owed = False
        except ValueError:
            pass
        return not self.owed

    @contextlib.contextmanager
    def active(self):
        """The request's pictures are encoded inside: an on-demand encoder ends at the exit and the engine has its
        memory back before the request runs."""
        with self.lock:
            try:
                yield self
            finally:
                if self.lend is not None and self.proc is not None:
                    self._stop()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            return base64.b64decode(source.split(",", 1)[1])
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "maya"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        path = source[7:] if source.startswith("file://") else source
        if path and os.path.isfile(path):
            return Path(path).read_bytes()
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats maya-vision's decoder (stb_image) reads pass through; anything else is converted to PNG."""
        if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode(self, source: str) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens)."""
        data = self.normalize(self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            if self.proc is None:
                self._start()
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            self.proc.stdin.write(f"ENC {img} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    VIDEO_FPS = 4.0                                        # (llama.cpp's mtmd default: the frames a second it reads)

    @staticmethod
    def video_container(data: bytes) -> str | None:
        """The container a video's first bytes name - MP4 / MOV, WebM / MKV or AVI - else None.  Only those reach
        ffmpeg: anything else (a playlist is text) could make it open other files or URLs while it reads."""
        if data[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
            return "mp4"
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
        if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
            return "avi"
        return None

    @staticmethod
    def video_seconds(path: Path) -> float | None:
        """A video file's duration by ffprobe (files only - no other protocol), None when it does not say."""
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries",
                                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                               capture_output=True, text=True, timeout=30)
            return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None

    def encode_video(self, source: str) -> tuple[Path, list[int]]:
        """-> (embeddings file, the frames' token counts), one SVE1 record per frame."""
        import shutil
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise ValueError("videos need ffmpeg and ffprobe on this PC (Linux: sudo apt install ffmpeg; Windows: winget "
                             "install Gyan.FFmpeg; macOS: brew install ffmpeg), then restart Maya")
        data = self.load(source)
        if self.video_container(data) is None:
            raise ValueError("this video's format is not one Maya reads: send MP4, MOV, WebM, MKV or AVI")
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            vid, out = self.dir / f"{key}.vid", self.dir / f"{key}.sve"
            vid.write_bytes(data)
            # a long video is many frames (every one an image's tokens): refused before it is decoded
            limit = max(1, int(os.environ.get("STRATA_VIDEO_MAX_FRAMES", "240") or 240))
            seconds = self.video_seconds(vid)
            if seconds is not None and seconds * self.VIDEO_FPS > limit:
                vid.unlink(missing_ok=True)
                raise ValueError(f"the video is {seconds:.0f} s long: {seconds * self.VIDEO_FPS:.0f} frames at "
                                 f"{self.VIDEO_FPS:g} a second, more than the {limit} Maya reads (MAYA_VIDEO_MAX_FRAMES "
                                 f"sets it; every frame takes an image's share of the context)")
            if self.proc is None:
                self._start()
            self.proc.stdin.write(f"VID {vid} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            vid.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the video could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            # the frames' token counts come from the record headers: int32 {magic, n_tokens, nx, ny, n_embd} each
            counts = []
            with open(out, "rb") as f:
                while hdr := f.read(20):
                    magic, n, _, _, n_embd = struct.unpack("<5i", hdr)
                    if magic != 0x31455653 or n_embd <= 0:
                        raise ValueError("not an SVE1 embeddings file")
                    counts.append(n)
                    f.seek(n * n_embd * 4, 1)
            self.cache[key] = (out, counts)
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def vision_footprint(vcfg: dict, env: dict, cache_file: Path, gpu: int) -> tuple[int, int] | None:
    """(image token cap, bytes) for the on-demand vision encoder on THIS machine's first GPU: measured once by
    `maya-vision --measure` (before the model loads, so the GPU is otherwise idle) and cached per encoder file, cap
    and GPU.  A cap the config does not set is the largest of 4096 / 2048 / 1024 image tokens whose footprint stays
    within 15% of the card (a 32 GB card keeps 4096, an 8-12 GB card gets 1024).  None: no GPU to measure on, or
    no measurement (the encoder failed or printed no well-formed MEM line)."""
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    if not isinstance(cache, dict):                     # a hand-edited or damaged file: measured again
        cache = {}
    st = Path(vcfg["mmproj"]).stat()
    caps = [int(vcfg["max_tokens"])] if vcfg.get("max_tokens") else [4096, 2048, 1024]
    for i, cap in enumerate(caps):
        key = f"{Path(vcfg['mmproj']).resolve()}|{st.st_size}|{int(st.st_mtime)}|gpu{gpu}|cap{cap}|" \
              f"fa{0 if vcfg.get('no_flash_attn') else 1}"
        m = cache.get(key)
        if not isinstance(m, dict) or not all(isinstance(m.get(k), int) and m[k] > 0 for k in ("bytes", "total")):
            m = None
        if m is None:
            args = [vcfg["exe"], "--mmproj", vcfg["mmproj"], "--model", vcfg["model"], "--gpu", "--max-tokens", str(cap),
                    "--measure"] + (["--no-flash-attn"] if vcfg.get("no_flash_attn") else [])
            print(f"[maya] images: measuring the vision encoder on this GPU ({cap} image tokens) ...", flush=True)
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=600, env=env)
                # only a measurement that ran to its end (an encoder whose warm-up failed exits 1 without one)
                f = next((l.split() for l in r.stdout.splitlines() if l.startswith("MEM ")), None) \
                    if r.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                f = None
            if f is None or len(f) != 4 or not all(x.isdigit() for x in f[1:]) or int(f[3]) == 0:
                return None
            # its weights and work buffers, the context this process had (and whatever else used the card then -
            # the safe side), and 256 MB of slack
            m = cache[key] = {"bytes": int(f[1]) + int(f[2]) + (256 << 20), "total": int(f[3])}
            try:
                cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            except OSError:
                pass
        if len(caps) == 1 or m["bytes"] <= 0.15 * m["total"] or i == len(caps) - 1:
            return cap, int(m["bytes"])
    return None


def trained_context(model_dir) -> int | None:
    """The context the model was trained for (`<arch>.context_length` in its first GGUF's header; GLM-5.3-Flash:
    1,048,576): the top of the dashboard's Context size.  None when there is no readable GGUF."""
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from gguf_reader import GGUFFile
        shards = sorted(Path(model_dir).glob("*-00001-of-*.gguf")) or sorted(Path(model_dir).glob("*.gguf"))
        if not shards:
            return None
        md = GGUFFile(shards[0]).metadata
        n = md.get(f"{md.get('general.architecture', '')}.context_length")
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:  # noqa: BLE001 - a header it can't read: the fallback range
        return None


def trained_context_ds4(pack) -> int | None:
    """DeepSeek's and MiniMax's: the context its pack says the model was trained for (ds4.txt's / mm3.txt's
    max_position_embeddings, from the model's config.json or a GGUF's context_length; V4.1-Flash and M3.1-Flash:
    1,048,576, their native reach), the top of the dashboard's Context size.  None when the pack does not say."""
    for name in ("ds4.txt", "mm3.txt"):
        try:
            with open(Path(pack) / name, encoding="utf-8", errors="replace") as f:
                for line in f:
                    k, _, v = line.strip().partition(" ")
                    if k == "max_position_embeddings":
                        n = int(v.split()[0])
                        return n if n > 0 else None
        except (OSError, ValueError, IndexError):
            continue
    return None


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32")."""
    args = list(cfg["args"])
    if len(gpu_list(cfg)) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    return args


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    if gpu_list(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (setup's KFD order)
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    elif gpu_list(cfg):                              # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries (AMD: the GEMM tuning table)
        env[str(k)] = str(v)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}

    def encode(self, text, parse_special=False, plain=()):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if parse_special and text.startswith(s, i) and not any(a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


class RequestStat:
    """One request's live figures: when its first answer token came (None while it reads the prompt) and its
    (time, generated) samples over RATE_WINDOW_S for the live decode rate."""

    def __init__(self):
        self.first_token = None
        self.base = None                                # (time, generated) at the window's start: the rate counts from it
        self.samples = collections.deque(maxlen=4096)   # the samples inside the window, oldest first

    def clear(self):
        self.base = None
        self.samples.clear()

    def _slide(self, now):
        while self.samples and self.samples[0][0] <= now - RATE_WINDOW_S:
            self.base = self.samples.popleft()

    def note(self, now, n):
        self.samples.append((now, n))
        self._slide(now)

    def tok_s(self, now) -> float:
        """From `base` (the last sample at or before now - RATE_WINDOW_S) to the newest.  Measuring from the first
        sample INSIDE the window read a burst after a long quiet stretch (tokens held up, then sent at once) as its size
        over a fraction of a second - a 300-token burst and a token 0.3 s later read ~1000 tok/s.  Younger than
        RATE_MIN_SPAN_S: the mean so far, span floored."""
        self._slide(now)
        start = self.base or (self.samples[0] if self.samples else None)
        if start is None:
            return 0.0
        t1, n1 = self.samples[-1] if self.samples else self.base
        if t1 - start[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (n1 - start[1]) / (t1 - start[0]))
        return (n1 - start[1] + 1) / max(RATE_MIN_SPAN_S, now - start[0])


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.fit_max_tokens = fit_max_tokens          # --fit-max-tokens: clamp the output cap instead of 400
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.default_effort = None                    # the run config's `reasoning_effort`: a request without one
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.fifo = FairLock()
        self.stopping = False            # the server is ending: a request still queued is refused, not started
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        self.response_limits = limits_from_env()
        self.responses = ResponseStore.from_limits(self.response_limits)
        self.response_jobs = BackgroundJobs(self.responses, self.response_limits)
        # pages on OTHER origins allowed to call /v1/* from a browser (CORS; the config's "cors_origins"): exact origins
        # like "http://127.0.0.1:8090", or "*" for every page.  Empty = no CORS headers at all.
        self.cors_origins: list[str] = []
        self.status = {"busy": False, "queued": 0}      # GET /status: what the model is doing right now
        self.active = 0                                 # requests with the engine now (several: READY multi=<n>)
        self.current_requests: set[RequestStat] = set()  # the requests decoding now: first token to the end of run()
        self.latest_request: RequestStat | None = None  # the request that last got the engine
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0}
        # the loop guard (serve/loop_guard.py; MAYA_LOOP_GUARD=0 / "loop_guard": false: off) and what it did since
        # the server started: loops seen, and how each was broken (the Monitor's totals, /metrics, Prometheus)
        self.loop_guard = True
        self.loop_guard_counts = {"detected": 0, "penalised": 0, "thinking_closed": 0, "stopped": 0}
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        self.mcp = None                                  # serve/mcp.py's McpHub when MCP servers are configured
        self.config_path = None                          # the run config (a context change is saved into it)
        self.savings = None                              # serve/savings.py's Ledger: what the API would have charged
        self.model_dir = None                            # the model's folder (free disk on the dashboard)
        self.model_label = None                          # what runs, as users see it: "MiniMax M3.1 · Maya-XS"
        self.trained_context = None                      # what the model was trained for (its GGUF): the slider's top
        self.dspark = None                               # DeepSeek's: drafts with its DSpark blocks (its engine's INFO
        #                                                  does not say: the config's --dspark and the pack's drafter)
        self.reload = None                               # a context change: {state, from, to, started, error}
        # "idle_unload_s" / --idle-unload (as Strata's): after this long without a request the engine ends, giving back
        # its GPU memory and its pinned RAM tier, and the next request starts it again (0: never).  With
        # STRATA_GLM_SLOT_KEEP=1 the conversation in the model is kept on disk across it.
        self.idle_unload_s = 0.0
        self.unloaded = False                            # the engine was ended by the idle unload (not by a crash)
        self._engine_start = threading.Lock()            # ensure_engine: one start, also with --parallel's requests
        self.updater = Updater(MAYA_VERSION)             # About > Updates: a newer release, and updating to it
        # The ids that end an answer server-side.  A string that is not a special token in this vocabulary
        # (GLM-5.3 has no <|im_end|>) encodes to its TEXT pieces, whose ids occur inside ordinary answers -
        # stopping on them would cut generation at any '<'.  Only single-token (real special) encodings count.
        def _special_ids(s: str) -> list[int]:
            ids = tokenizer.encode(s, parse_special=True)
            return ids if len(ids) == 1 else []
        # GLM ends an assistant turn with the next role's token (<|user|> / <|observation|>): the engine stops on it,
        # and it must not reach the answer's text either
        self.stop_ids = set(_special_ids(IM_END) + _special_ids("<|endoftext|>") + _special_ids("<|user|>") +
                            _special_ids("<|observation|>") +
                            # DeepSeek-V4.1 ends a turn with its EOS (and would start the next with <｜User｜>)
                            _special_ids("<｜end▁of▁sentence｜>") + _special_ids("<｜User｜>") +
                            # MiniMax-M3.1 with [e~[ (and would start the next turn with ]~b])
                            _special_ids("[e~[") + _special_ids("]~b]"))
        # ... and a model that spells a role marker out of ordinary pieces ('<|', 'user', '|>') has ended its turn just
        # the same: the text stops there.  Only markers that are special tokens in this vocabulary count.
        self.stop_texts = [s for s in ("<|user|>", "<|observation|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<｜User｜>",
                                       "[e~[", "]~b]")
                           if _special_ids(s)]
        # the thinking budget (run config `thinking_budget`, tokens; 0 = none): a reasoning block that reaches it is
        # closed with the template's </think> and the answer follows - GLM at High can plan for 14k+ tokens
        self.think_budget = 0
        # (MiniMax-M3.1 closes its reasoning with </mm:think>; its vocabulary may hold a plain "</think>" too)
        end = _special_ids("</mm:think>") if getattr(template, "minimax", False) else _special_ids("</think>")
        self.think_end_id = end[0] if end else None
        # the special literals that stay text inside a message (frontend.mark_literals): every one the vocabulary has
        self.literals = literal_tags(getattr(tokenizer, "special_tokens", None))

    def check_request(self, req: dict) -> None:
        """A request's sampling fields and logit_bias, checked (a ValueError is the 400) and kept in the form the engine
        line takes, before anything is sent.  A logit_bias needs an engine that applies it (INFO logit_bias=1): an
        older one would answer as if it had none."""
        check_request_sampling(req)
        bias = logit_bias_of(req.get("logit_bias"), len(getattr(self.tok, "tokens", None) or ()) or None)
        if bias and (getattr(self.engine, "info", None) or {}).get("logit_bias") != 1:
            raise ValueError("logit_bias: this engine does not apply token biases - update Maya (./maya.sh --setup)")
        if "logit_bias" in req:
            req["logit_bias"] = bias

    def render_ids(self, messages, tools, kwargs, suffix="") -> list[int]:
        """The template rendered for `messages` and `tools` (+ `suffix`, text the server appends to the prompt) and
        tokenized.  A special token's text inside a message or a tool is encoded as that text; only the literals the
        template (or the suffix) writes are special tokens."""
        marked, marked_tools, changed = mark_literals(messages, tools, self.literals)
        prompt = self.template.render(marked, tools=marked_tools, **kwargs)
        if kwargs.get("enable_thinking") is False and prompt.endswith("<think>"):
            # GLM-5.3's template opens the thinking block whatever it is told (no enable_thinking, and an effort it
            # does not know means 'max'): with thinking off the answer was written inside an open <think>.  Close
            # it empty, the form the same template writes for a past turn that did not think.
            prompt += "</think>"
        if not changed:
            return self.tok.encode(prompt + suffix, parse_special=True)
        prompt, plain = unmark_literals(prompt, self.literals)
        return self.tok.encode(prompt + suffix, parse_special=True, plain=plain)

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[maya] could not save the shared settings: {e}", flush=True)
        return self.shared

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        if not s:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        effort = s.get("reasoning_effort")
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
                if not req.get("reasoning_effort") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def close_for_restart(self):
        """Before the server ends for an update: the engine, the pictures encoder and the MCP servers stop first."""
        for part in (self.response_jobs, self.engine, self.vision, self.mcp):
            try:
                if part is not None and hasattr(part, "close"):
                    part.close()
            except Exception as e:  # noqa: BLE001 - ending anyway
                print(f"[maya] stopping {type(part).__name__}: {e}", flush=True)

    def ensure_engine(self):
        """The engine running (the caller holds the FIFO): started again after an idle unload, or after it died in an
        earlier request (issue #27) instead of failing every request.  With --parallel several requests hold a place
        each: the first one starts it, the others wait for that start."""
        if not hasattr(self.engine, "alive") or self.engine.alive():
            return
        with self._engine_start:
            if self.engine.alive():                      # (another request started it meanwhile)
                return
            if self.unloaded:
                print("[maya] loading the model again after an idle unload (a minute or so) ...", flush=True)
            else:
                code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
                print(f"[maya] the engine had stopped (exit code {code}); starting it again "
                      "(a minute or two) ...", flush=True)
            self.engine.restart()
            self.unloaded = False
            # the FIFO's places are the conversations the restarted engine reports: a start that came back with other
            # numbers (a different --parallel, no multi= at all) queued or over-admitted every request from here on
            places = max(1, int(getattr(self.engine, "multi", 0) or 1))
            if places != self.fifo.capacity:
                self.fifo.set_capacity(places)
            print("[maya] the engine is running again", flush=True)

    def unload(self, idle_for: float | None = None) -> str:
        """End the engine between requests: "unloaded", "not loaded", "busy" (a request is running or waiting, a context
        reload is on, or - with idle_for - one ran more recently than that) or "unsupported"."""
        if not (hasattr(self.engine, "close") and hasattr(self.engine, "restart") and hasattr(self.engine, "alive")):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):        # every place: nothing runs or waits for the engine
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued") or self.active:
                    return "busy"
                if self.reload and self.reload.get("state") in ("waiting", "running"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            # QUIT (close: with STRATA_GLM_SLOT_KEEP the engine first writes the conversation it holds), then the GPU has
            # its memory back.  (StrataEngine.stop is the request's STOP, not the engine's end.)
            self.engine.close()
            self.engine.ended = True
            pid = getattr(getattr(self.engine, "proc", None), "pid", None)
            if pid is not None:
                wait_gpu_release(pid)
            self.unloaded = True
            print(f"[maya] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''} (GPU memory and the "
                  "RAM tier given back); the next request loads it again", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "close"):
            return
        print(f"[maya] the model unloads after {self.idle_unload_s:g} s without requests", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                try:
                    self.unload(idle_for=self.idle_unload_s)
                except Exception as e:  # noqa: BLE001 - tried again at the next turn; the thread keeps running
                    print(f"[maya] idle unload: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def reloading(self) -> bool:
        with self.status_lock:
            return bool(self.reload) and self.reload.get("state") in ("waiting", "running")

    def context_limits(self) -> dict:
        """The dashboard's Context size: the range it offers and the engine's measured cost of the current one."""
        info = dict(getattr(self.engine, "info", {}) or {})
        top = self.trained_context or CONTEXT_FALLBACK_MAX
        with self.status_lock:
            reload = dict(self.reload) if self.reload else None
        # a streamed cache's RAM (KV streaming: maya-mm3's VRAM lines' kv_ram - every position's K / V in pinned RAM,
        # the engine's own count at this context) and the RAM free now: the dashboard scales the one against the other
        kv_ram = sum(int(p.get("kv_ram") or 0) for p in getattr(self.engine, "vram", None) or [])
        ram = {}
        if kv_ram:
            try:
                import psutil
                vm = psutil.virtual_memory()
                ram = {"ram_gb": round(vm.total / 2**30, 1), "ram_free_gb": round(vm.available / 2**30, 1)}
            except Exception:  # noqa: BLE001 - psutil is optional: no RAM figures, no RAM warning
                pass
        return {"context": self.engine.max_context, "min": CONTEXT_MIN, "max": max(top, self.engine.max_context),
                "trained": self.trained_context,
                "tested": CONTEXT_TESTED.get(str(info.get("engine_kind") or ""), CONTEXT_TESTED_DEFAULT),
                "kv_gb": info.get("kv_gb"), "kv_ctx": info.get("kv_ctx"),
                "kv_resident": info.get("kv_resident") or 0,
                "kv_ram_gb": round(kv_ram / 2**30, 3) if kv_ram else None, **ram,
                "vram_gb": info.get("vram_gb"), "vram_slots": info.get("vram_slots"), "reload": reload,
                "load_s": getattr(self.engine, "load_s", None)}   # (the engine's last load: what a reload takes here)

    def set_context(self, n: int) -> dict:
        """Reload the engine with an n-token context (the dashboard's Context size), on a thread: after the request in
        flight; new requests get a 503 meanwhile.  A start that fails (the context does not fit) puts the old one back;
        a good one is saved into the run config, so the next start keeps it."""
        lim = self.context_limits()
        if not lim["min"] <= n <= lim["max"]:
            raise ValueError(f"the context must be between {lim['min']} and {lim['max']} tokens")
        with self.status_lock:
            if self.reload and self.reload.get("state") in ("waiting", "running"):
                raise ValueError("the model is already reloading")
            self.reload = {"state": "waiting", "from": self.engine.max_context, "to": n, "started": time.time(),
                           "error": None}
            reload = dict(self.reload)
        threading.Thread(target=self._reload, args=(reload["from"], n), daemon=True).start()
        return reload

    def _restart_with(self, n: int):
        if hasattr(self.engine, "set_context"):
            self.engine.stop()
            self.engine.set_context(n)
            self.engine.restart()
        else:                                            # the mock engine: nothing to start
            time.sleep(float(os.environ.get("STRATA_MOCK_RELOAD_S", "2")))
            self.engine.max_context = n

    def _reload(self, old: int, n: int):
        with self.fifo:                                  # the request in flight finishes first
            with self.status_lock:
                self.reload["state"] = "running"
            print(f"[maya] reloading the model with a {n}-token context (was {old}) ...", flush=True)
            err = None
            t0 = time.time()
            try:
                self._restart_with(n)
                self.engine.load_s = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001 - any failed start: put the old context back
                err = f"the engine did not start with a {n}-token context ({e})"
                print(f"[maya] {err}; starting it again with {old} ...", flush=True)
                try:
                    self._restart_with(old)
                except Exception as e2:  # noqa: BLE001
                    err += f"; starting again with {old} failed too ({e2}) - the next request tries once more"
            places = max(1, int(getattr(self.engine, "multi", 0) or 1))   # the reloaded engine's conversations
            if places != self.fifo.capacity:
                self.fifo.set_capacity(places)
            if err is None:
                self._save_context(n)
                print(f"[maya] the model runs with a {n}-token context now", flush=True)
            with self.status_lock:
                self.reload.update(state="failed" if err else "done", error=err, ended=time.time(),
                                   context=self.engine.max_context)

    def _save_context(self, n: int):
        """--max-context in the run config (its other keys as they are), so the next start keeps the new size."""
        if not self.config_path:
            return
        try:
            p = Path(self.config_path)
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
            inst = cfg.setdefault("installer", {})       # (the first change from the page: what the setup wrote first,
            if "as_set_up" not in inst:                  #  the Models page's Reset - as models.save keeps it)
                inst["as_set_up"] = models.settings_of(cfg)
            args = list(cfg.get("args") or [])
            if "--max-context" in args[:-1]:
                args[args.index("--max-context") + 1] = str(int(n))
            else:
                args += ["--max-context", str(int(n))]
            cfg["args"] = args
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            os.replace(tmp, p)
        except (OSError, ValueError) as e:
            print(f"[maya] could not save the new context in {self.config_path}: {e}", flush=True)

    def port_info(self) -> dict:
        """The dashboard's Port: where the server listens, and whether a new port can apply at once (maya.py started
        this server and starts it again on the saved port) or only from the next start."""
        host, port = getattr(self, "listen", None) or ("127.0.0.1", None)
        return {"port": port, "host": host, "restarts": bool(os.environ.get("MAYA_RESTART_ON_UPDATE")),
                "saved": bool(self.config_path), "restarting": getattr(self, "port_restart", None)}

    def set_port(self, n: int) -> dict:
        """The dashboard's Port: n saved into the run config (and Maya's start scripts for it), then - when maya.py
        started this server - the server ends with RESTART_EXIT after the request in flight, and maya.py starts it again
        on n.  The port must be free on this server's address now (a busy one: an error, nothing saved)."""
        host, port = getattr(self, "listen", None) or ("127.0.0.1", None)
        if not 1024 <= n <= 65535:
            raise ValueError("the port must be a number from 1024 to 65535")
        if n == port:
            raise ValueError(f"Maya listens on {n} already")
        if not self.config_path:
            raise ValueError("this server runs without a config, so a port cannot be saved: start it with --port")
        if getattr(self, "port_restart", None):
            raise ValueError("Maya is already moving to another port")
        import socket
        try:
            with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, n))
        except OSError as e:
            raise ValueError(f"port {n} is in use or not allowed here ({e.strerror or e})") from None
        self._save_port(n)
        restarts = bool(os.environ.get("MAYA_RESTART_ON_UPDATE"))
        if restarts:
            self.port_restart = n
            threading.Thread(target=self._restart_on_port, args=(n,), daemon=True).start()
        print(f"[maya] the port {n} is saved" + (": Maya starts again on it" if restarts else
              " - it applies from the next start"), flush=True)
        return {"port": n, "restarts": restarts}

    def _restart_on_port(self, n: int):
        time.sleep(0.5)                                  # (the answer to the page goes out first)
        with self.fifo:                                  # the request in flight finishes first
            print(f"[maya] starting again on port {n} ...", flush=True)
            self.close_for_restart()
            os._exit(RESTART_EXIT)

    def _save_port(self, n: int):
        """"port" in the run config, and --port in Maya's start scripts beside it that start this config (setup writes
        run-maya-<model>.bat / .sh with the port then chosen; a script would otherwise move it back)."""
        p = Path(self.config_path)
        cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        cfg["port"] = int(n)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        for script in list(p.parent.glob("run-*.bat")) + list(p.parent.glob("run-*.sh")):
            try:
                raw = script.read_bytes()                # (bytes: a .bat keeps its CR LF line ends)
                if p.name.encode() not in raw:
                    continue
                new = re.sub(rb"(--port\s+)\d+", lambda m: m.group(1) + str(int(n)).encode(), raw)
                if new != raw:
                    script.write_bytes(new)
            except OSError as e:
                print(f"[maya] could not change the port in {script.name}: {e}", flush=True)

    # -------------------------------------------------------------- the Models page (serve/models.py)
    def models_root(self) -> Path:
        """Where the configs are: beside this server's own (maya.py's folder)."""
        return Path(self.config_path).parent if self.config_path else ROOT

    def models_info(self) -> dict:
        """The configs the setup wrote, the running one marked, each with its engine settings; whether a switch or a
        change to the running one can start Maya again at once (maya.py started this server)."""
        out = models.listing(self.models_root(), self.config_path)
        out.update(restarts=bool(os.environ.get("MAYA_RESTART_ON_UPDATE")),
                   running=Path(self.config_path).name if self.config_path else None,
                   restarting=getattr(self, "model_restart", None))
        return out

    def _model_path(self, name) -> Path:
        name = str(name or "")
        p = self.models_root() / name
        if (not re.fullmatch(r"maya-[^/\\]+\.json", name) or name.endswith(".shared-settings.json")
                or not p.is_file()):
            raise ValueError(f"no model config named {name!r} here")
        return p

    def _is_running(self, p: Path) -> bool:
        return bool(self.config_path) and p.resolve() == Path(self.config_path).resolve()

    def _restart_busy(self):
        if getattr(self, "model_restart", None) or getattr(self, "port_restart", None):
            raise ValueError("Maya is already starting again")
        if self.reloading() or self.updater.busy():
            raise ValueError("Maya is reloading or updating; try again when it is done")

    def model_settings(self, name, settings, dry_run: bool = False) -> dict:
        """The engine settings of the config `name` (the Models page): checked, and unless dry_run saved into it.  The
        running model's start Maya again with them (when maya.py started this server), after the request in flight;
        another model's apply when it is switched to.  dry_run: the command they make, nothing saved."""
        p = self._model_path(name)
        cfg = models.read(p)
        if not cfg:
            raise ValueError(f"{p.name}: not a config Maya can read")
        new = models.apply(cfg, settings)                 # (a ValueError: what is wrong, nothing saved)
        if dry_run:
            return {"command": models.command(new), "settings": models.shown_settings(new)}
        running = self._is_running(p)
        if running:
            self._restart_busy()
        out = models.save(p, settings)
        restarts = bool(running and out["changed"] and os.environ.get("MAYA_RESTART_ON_UPDATE"))
        if restarts:
            self._restart_into(p, "settings")
        print(f"[maya] {p.name}: engine settings saved" + (": Maya starts again with them" if restarts else
              " - they apply from its next start" if out["changed"] else " (nothing changed)"), flush=True)
        return {**out, "running": running, "restarts": restarts}

    def switch_model(self, name) -> dict:
        """The Models page's Switch: the config `name` becomes the most recently used one (a start without --config
        picks it), and when maya.py started this server it ends with RESTART_EXIT after the request in flight, leaving
        maya.py a note (models.SWITCH_FILE) to start that config, on this port."""
        p = self._model_path(name)
        if self._is_running(p):
            raise ValueError(f"{p.name} is the model running now")
        cfg = models.read(p)
        why = models.missing(cfg)                        # (maya.py would stop at it: no model would run)
        if why:
            raise ValueError(f"{p.name} cannot start: {why} - run the setup again to repair it")
        self._restart_busy()
        os.utime(p)
        restarts = bool(os.environ.get("MAYA_RESTART_ON_UPDATE"))
        if restarts:
            self._restart_into(p, "switch")
        print(f"[maya] {p.name} is the model to start" + (": Maya starts again with it" if restarts else
              " next (a start without --config picks it)"), flush=True)
        return {"to": p.name, "model": cfg.get("model_name") or "", "restarts": restarts}

    def _restart_into(self, p: Path, why: str):
        _, port = getattr(self, "listen", None) or ("127.0.0.1", None)
        note = {"config": str(p.resolve()), "port": port, "why": why, "time": time.time()}
        (self.models_root() / models.SWITCH_FILE).write_text(json.dumps(note), encoding="utf-8")
        self.model_restart = {"to": p.name, "why": why, "started": time.time()}
        threading.Thread(target=self._restart_for_models, args=(p.name, why), daemon=True).start()

    def _restart_for_models(self, name: str, why: str):
        time.sleep(0.5)                                  # (the answer to the page goes out first)
        with self.fifo:                                  # the request in flight finishes first
            print(f"[maya] starting again with {name} ..." if why == "switch" else
                  f"[maya] starting again with {name}'s new engine settings ...", flush=True)
            self.close_for_restart()
            os._exit(RESTART_EXIT)

    def storage(self) -> dict | None:
        """The model's folder and the free space on its drive (About)."""
        if not self.model_dir:
            return None
        try:
            import shutil
            u = shutil.disk_usage(self.model_dir)
            return {"path": str(self.model_dir), "free": u.free, "total": u.total}
        except OSError:
            return None

    def report(self) -> str:
        """A plain-text report for an issue (the dashboard's Copy report): versions, this PC, the engine's facts, the
        run config without its secrets, and the engine log's telling lines."""
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, st = tel.get("now") or {}, tel.get("static") or {}
        gib = lambda b: f"{b / 2**30:.1f} GB" if isinstance(b, (int, float)) else "?"  # noqa: E731
        import platform
        out = [f"Project Maya {MAYA_VERSION or '?'} - report of {time.strftime('%Y-%m-%d %H:%M')}",
               f"System: {platform.platform()}, Python {platform.python_version()}",
               f"GPU: {st.get('gpu_name') or 'not readable'} ({gib(hw.get('gpu_mem_total'))} VRAM)",
               f"CPU: {st.get('cpu_name') or '?'}, {st.get('threads') or '?'} threads; RAM {gib(hw.get('ram_total'))}",
               f"Model: {self.model}" + (f" ({self.model_label})" if self.model_label else "") +
               f", context {self.engine.max_context}"]
        info = dict(getattr(self.engine, "info", {}) or {})
        if info:
            out.append("Engine: " + " ".join(f"{k}={v}" for k, v in sorted(info.items())))
        if self.config_path:
            try:
                cfg = json.loads(Path(self.config_path).read_text(encoding="utf-8-sig"))
                secret = re.compile(r"key|token|secret|password", re.I)
                env = {maya_names.shown(k): ("***" if secret.search(k) else v) for k, v in (cfg.get("env") or {}).items()}
                out.append(f"Config: {Path(self.config_path).name}: args {' '.join(map(str, cfg.get('args') or []))}; "
                           f"gpu {cfg.get('gpu')}; env {json.dumps(env)}")
            except (OSError, ValueError):
                pass
        log = getattr(self.engine, "log_path", None)
        if log:
            try:
                with open(log, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - (2 << 20)))
                    lines = f.read().decode("utf-8", "replace").splitlines()
                keep = [ln for ln in lines if REPORT_LINES.search(ln)]
                out += ["", f"Engine log ({log}), the telling lines of the last 2 MB:"] + keep[-150:]
            except OSError:
                pass
        return "\n".join(out) + "\n"

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (serve/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from serve.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(), "tok_s_mean": self._tok_s_mean(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpu_index=int(getattr(self, "gpu_index", 0) or 0),
                                       gpu_indices=getattr(self, "gpu_indices", None),
                                       amd=getattr(self, "backend", None) == "hip")
            if isinstance(self.engine, StrataEngine):
                self.engine.gpu_busy = self._gpu_busy     # the stall watchdog's third sign of work
        self.start_queue_watch()

    def start_queue_watch(self):
        """A queue that has stopped moving says so, every QUEUE_WARN_S, with what holds the engine and what is waiting:
        a FIFO place that is never given back (a request waiting on an engine line that never came, a place leaked by
        an error) makes every later request wait for ever, and until this line appeared nothing said anything - the
        server looked idle, the requests piled up, and none of them expired."""
        if getattr(self, "_queue_watch", None) is not None:
            return

        def loop():
            warned = 0.0
            while True:
                time.sleep(QUEUE_TICK_S)
                try:
                    oldest, waiting = self.fifo.oldest_wait(), self.fifo.waiting()
                except Exception:  # noqa: BLE001 - a diagnostic must never end the thread
                    continue
                if not waiting or oldest < QUEUE_WEDGED_S or time.monotonic() - warned < QUEUE_WARN_S:
                    continue
                warned = time.monotonic()
                with self.status_lock:
                    s = dict(self.status)
                    active = self.active
                doing = (f"a request is running ({s.get('phase') or 'working'}, {s.get('generated')} tokens so far)"
                         if s.get("busy") else "nothing is running")
                print(f"[maya] WARNING: {waiting} request(s) have waited for the engine for up to {oldest:.0f} s and "
                      f"{doing}; {self.fifo.held_by()}, {active} request(s) counted active. A request that holds the "
                      f"engine and waits on it for ever stops every request behind it - STRATA_REQUEST_QUIET_S / "
                      f"STRATA_DRAIN_S bound those waits, so this line is what a wedged FIFO would look like. "
                      f"Queue state: GET /status, /metrics (queued, active).", flush=True)

        self._queue_watch = threading.Thread(target=loop, daemon=True, name="strata-queue-watch")
        self._queue_watch.start()

    def _gpu_busy(self) -> bool:
        """Any of the engine's GPUs busier than STALL_GPU_BUSY in the telemetry's last sample (one a second)."""
        now = self.telemetry.now
        utils = [g.get("util") for g in now.get("gpus") or []] or [now.get("gpu_util")]
        return any(isinstance(u, (int, float)) and u > STALL_GPU_BUSY for u in utils)

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds, summed over the requests decoding now.  0.0 while none is."""
        with self.status_lock:
            if not self.status.get("busy"):
                return 0.0
            now = time.time()
            return sum(r.tok_s(now) for r in self.current_requests)

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            s = dict(self.status)
            r = self.latest_request if self.latest_request in self.current_requests else None
        if not s.get("busy") or r is None:
            return 0.0
        return s["generated"] / max(1e-6, time.time() - r.first_token)

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.latest_request not in self.current_requests
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        with self.status_lock:
            s = dict(self.status)
            answering = self.latest_request in self.current_requests
            hist = list(self.history)
            totals = dict(self.totals)
            guard = dict(self.loop_guard_counts, on=bool(self.loop_guard))   # (what the loop guard did since start)
            active = self.active
            writing = len(self.current_requests)        # (of the active ones, those past their prompt)
        now = time.time()
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and not answering:
            state = "reading"
        elif s.get("busy"):
            state = "generating"
        else:
            state = "unloaded" if self.unloaded else "idle"
        live = {"state": state, "queued": s.get("queued", 0), "active": active,   # (answering at once: --parallel)
                "writing": writing,
                "phase": s.get("phase") if s.get("busy") else None,
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if s.get("busy") else None,
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress[0], progress[1]
        # the prompt the conversation cache already held (the Monitor's Context bar), from the engine's PP lines
        live["prompt_reused"] = progress[2] if s.get("busy") and progress and len(progress) > 2 else None
        engine = {"model": self.model, "model_label": self.model_label, "max_context": self.engine.max_context,
                  "images": self.vision is not None,
                  **dict(getattr(self.engine, "info", {}) or {}), "maya_version": MAYA_VERSION,
                  "trained_context": self.trained_context,
                  **({"dspark": self.dspark} if self.dspark is not None else {})}
        with self.status_lock:
            reload = dict(self.reload) if self.reload else None
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        # the GLM engine's expert tiers (STAT lines): the latest reading and the recent series for the sparklines
        tiers = None
        stat = getattr(self.engine, "stat", None)
        if stat:
            sh = list(getattr(self.engine, "stat_history", []) or [])
            tiers = {"now": stat, "history": {k: [s.get(k) for s in sh] for k in
                                              ("tok_s", "ms_tok", "vram_hit", "ram_fetch", "disk", "promo",
                                               "cpu", "ram", "disk_mb", "engram_ms")}}   # (the last four: DeepSeek's)
        hw = dict(tel["now"])
        cards = hw.pop("vram", None)                      # (each card's processes: for the GPU map only)
        return {"engine": engine, "live": live, "tiers": tiers, "gpu_map": self._gpu_map(cards),
                "requests": hist[::-1][:None if all_requests else 12],
                "requests_kept": len(hist), "totals": totals, "loop_guard": guard, "hardware": hw,
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now, "reload": reload, "storage": self.storage(),
                "savings": self.savings.summary() if self.savings is not None else None}

    def _gpu_map(self, cards):
        """What fills each GPU (serve/gpu_map.py): the engine's VRAM lines and NVML's cards, with the engine's and the
        vision encoder's processes told apart from other programs; None for an engine without VRAM lines."""
        report = getattr(self.engine, "vram", None)
        if not report:
            return None
        proc = getattr(self.engine, "proc", None)
        v = self.vision
        vproc = getattr(v, "proc", None) if v is not None else None
        lent = v is not None and getattr(v, "lend", None) is not None and (vproc is not None or getattr(v, "owed", False))
        listed = {pid for c in cards or [] for pid in c.get("procs") or {}}
        return gpu_map.view(report, cards, nvml_pids(proc.pid if proc is not None else None, listed),
                            nvml_pids(vproc.pid if vproc is not None else None, listed), lent)

    # the GLM engine's live STAT values (StrataEngine._parse_stat) exported as gauges: key -> help
    PROM_STAT = {"tok_s": "Decode rate over the running request, tokens/s",
                 "vram_hit": "Share of the routed experts found in VRAM (0..1)",
                 "ram_fetch": "Experts a token read from the RAM tier",
                 "disk": "Experts a token read from the SSD",
                 "promo": "Experts a token promoted into VRAM",
                 "cpu": "Experts a token computed on the CPU lane (DeepSeek)",
                 "disk_mb": "MB a token the storage read (DeepSeek, Linux)",
                 "engram_ms": "The n-gram memory's (Engram) read time a token, ms (DeepSeek)",
                 "eg_cache_rows": "Engram rows held in the host cache",
                 "eg_cache_cap": "Engram host cache capacity, rows",
                 "eg_cache_row_bytes": "Engram host cache payload bytes per row",
                 "eg_lookup_rows": "Consumer Engram row lookups since engine start",
                 "eg_lookup_hits": "Consumer Engram row lookups served without a file read since engine start",
                 "eg_lookup_reads": "Consumer Engram on-demand file reads since engine start (prefetch reads are separate)",
                 "slot_ram": "Parked DeepSeek conversations held in RAM",
                 "slot_disk": "Parked DeepSeek conversations held in the slot folder",
                 "slot_ram_bytes": "Parked DeepSeek conversation payload bytes in RAM",
                 "slot_ram_cap": "Parked DeepSeek conversation RAM budget, bytes; 0 means count-limited only",
                 "vram_used": "Expert slots in use in VRAM", "vram_slots": "Expert slots in VRAM",
                 "ram_used": "Expert slots in use in RAM", "ram_slots": "Expert slots in RAM"}
    PROM_STAT_COUNTERS = {"eg_lookup_rows", "eg_lookup_hits", "eg_lookup_reads"}

    def prometheus(self) -> str:
        """GET /metrics/prometheus: the totals and what is running in Prometheus' text format, under llama-server's
        metric names (llamacpp:*) so its dashboards work unchanged; the requests and the engine's expert tiers as
        maya:*."""
        with self.status_lock:
            s, t, active = dict(self.status), dict(self.totals), self.active
            guard = dict(self.loop_guard_counts)
        stat = dict(getattr(self.engine, "stat", None) or {})
        prompt_n = t["prompt_tokens"] - t["reused"]               # what was read, as llama.cpp counts it
        prompt_s, decode_s = t["prompt_ms"] / 1000, t["decode_ms"] / 1000
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", prompt_n),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the kept conversation already held",
             t["reused"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", prompt_s),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", t["output_tokens"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", decode_s),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt read throughput in tokens/s",
             prompt_n / prompt_s if prompt_s else 0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s",
             t["output_tokens"] / decode_s if decode_s else 0),
            ("llamacpp:requests_processing", "gauge", "Requests running", max(active, int(bool(s.get("busy"))))),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", s.get("queued") or 0),
            ("maya:requests_total", "counter", "Requests finished", t["requests"]),
            ("maya:engine_alive", "gauge", "1 while the engine process runs",
             int(self.engine.alive()) if hasattr(self.engine, "alive") else 1),
            ("maya:loop_guard_detected_total", "counter", "Loops the loop guard saw in answers", guard["detected"]),
            ("maya:loop_guard_penalised_total", "counter",
             "Loops broken by dropping the repeats and a short repetition penalty", guard["penalised"]),
            ("maya:loop_guard_thinking_closed_total", "counter", "Loops in the thinking that closed it",
             guard["thinking_closed"]),
            ("maya:loop_guard_stopped_total", "counter", "Answers the loop guard ended (a loop that came back)",
             guard["stopped"]),
        ]
        rows += [(f"maya:engine_{k}", "counter" if k in self.PROM_STAT_COUNTERS else "gauge", h, stat[k])
                 for k, h in self.PROM_STAT.items()
                 if isinstance(stat.get(k), (int, float)) and not isinstance(stat.get(k), bool)]
        for category, values in (("cache", self.responses.stats()), ("jobs", self.response_jobs.stats())):
            gauges = {"entries", "blocks", "bytes", "inflight_bytes", "peak_inflight_bytes", "max_entries",
                      "max_bytes", "ttl_s", "records", "running", "queued", "event_bytes", "readers"}
            rows += [(f"maya:responses_{category}_{key}", "gauge" if key in gauges else "counter",
                      f"Responses {category} {key}", value) for key, value in values.items()]
        return "".join(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name} {value}\n"
                       for name, kind, help_, value in rows)

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
            active = self.active
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}
        places = max(1, int(getattr(self.engine, "multi", 0) or 0))   # requests the engine answers at once

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.engine.max_context
        images = self.vision is not None
        return {
            "service": "maya", "model": self.model,
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": places, "requested": places},   # more than that wait their turn
            "dialects": ["/v1/chat/completions", "/v1/responses", "/v1/messages"],
            "responses": {"cache": self.responses.stats(), "jobs": self.response_jobs.stats(),
                          "structured_outputs": {"mode": "validate-before-publish", "schema_draft": "2020-12",
                                                 "constrained_decoding": False}},
            "vision": {"enabled": images, "available": images,
                       "error": None if images or (getattr(self.engine, "info", {}) or {}).get("vision") != "off"
                       else "the model's vision tower did not load (the server's log says why)"},
            # (a Windows APU: Windows' commit - the page file - kept experts off the GPU; either engine's NOTE says so)
            "memory_note": (lambda i: None if not i.get("gpu_commit_capped_gb") else
                            f"{i['gpu_commit_capped_gb']} GB of experts stay off the GPU: Windows' page file is too small; "
                            f"a page file of {i.get('pagefile_hint_gb')} GB keeps them on the GPU")(
                                getattr(self.engine, "info", {}) or {}),
            "activity": {"requests": totals["requests"] + max(int(busy), active),
                         "in_flight": max(int(busy), active) + int(s.get("queued") or 0),
                         # the FIFO's state: a queue whose first waiter is old while nothing runs is stuck, not busy
                         "queue": {"waiting": int(s.get("queued") or 0),
                                   "first_waited_s": round(self.fifo.oldest_wait(), 1),
                                   "places": self.fifo.capacity},
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def prepare(self, messages, tools, kwargs, max_new=None, cancel=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  A tool_choice that requires a call ("_force_tool": "" any, else the tool's name)
        starts the answer with the call's opening, thinking off; that text goes back in kwargs["_prefill"] for run()."""
        given, force = kwargs, kwargs.get("_force_tool")
        kwargs = {k: v for k, v in kwargs.items() if k != "_force_tool"}
        if force is not None and tools:
            kwargs["enable_thinking"] = False
        if self.default_effort and "reasoning_effort" not in kwargs and "enable_thinking" not in kwargs:
            kwargs = {**kwargs, **(deepseek_effort_kwargs(self.default_effort) if getattr(self.template, "deepseek", False)
                                  else effort_kwargs(self.default_effort))}   # the run config's thinking level
        prefill = ""
        if force is not None and tools:
            # GLM's form opens with the tool's name, Qwen's with <function=NAME>, DeepSeek's DSML with its calls block
            glm_calls = "<arg_key>" in self.template.source
            if getattr(self.template, "deepseek", False):
                prefill = "\n\n<｜DSML｜ calls>\n" + (f'<｜DSML｜ invoke name="{force}">\n' if force else "")
            elif getattr(self.template, "minimax", False):
                prefill = MINIMAX_NS + "<tool_call>\n" + (MINIMAX_NS + f'<invoke name="{force}">' if force else "")
            else:
                prefill = CALL_START + (force if glm_calls else "\n" + (f"<function={force}>\n" if force else ""))
        given["_prefill"] = prefill
        ids = self.render_ids(messages, tools, kwargs, prefill)
        self.embeddings.path = None
        media = media_of(messages)
        images = (deepseek_images(messages) if getattr(self.template, "deepseek", False)
                  else [src for kind, src in media if kind == "image"])
        if isinstance(self.vision, mm3_vision.Vision) and any(kind == "video" for kind, _ in media):
            raise ValueError("MiniMax-M3.1 reads pictures, not videos")
        if images and isinstance(self.vision, ENGINE_VISION):
            # DeepSeek: the engine encodes the pictures itself; each placeholder the template wrote (user text's are
            # plain text, see render_ids) becomes its picture's span, the resized pixels go in the GENI file
            pics = []
            for src in images:
                if cancel is not None and cancel.is_set():
                    raise ValueError("image preparation was cancelled")
                pics.append(self.vision.picture(src))
            if cancel is not None and cancel.is_set():
                raise ValueError("image preparation was cancelled")
            ids, starts = self.vision.expand(ids, pics)
            self.embeddings.path = self.vision.write(self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.dsi", pics,
                                                     starts)
        elif media:
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images or videos")
            # the template's media markers: GLM-5.3 writes <|begin_of_image|><|image|><|end_of_image|> and
            # <|begin_of_video|><|video|><|end_of_video|>, Qwen <|vision_start|><|image_pad|><|vision_end|> and
            # <|vision_start|><|video_pad|><|vision_end|>
            glm = "<|begin_of_image|>" in self.template.source
            image_pad, image_start = ("<|image|>", "<|begin_of_image|>") if glm else (IMAGE_PAD, VISION_START)
            video_pad, video_start = ("<|video|>", "<|begin_of_video|>") if glm else ("<|video_pad|>", VISION_START)
            pad = self.tok.encode(image_pad, parse_special=True)[0]
            start = self.tok.encode(image_start, parse_special=True)[0]
            vp = self.tok.encode(video_pad, parse_special=True)[0]
            vstart = self.tok.encode(video_start, parse_special=True)[0]
            # every frame is an independent image: the image's own begin / end markers around its pads
            iend = self.tok.encode("<|end_of_image|>" if glm else "<|vision_end|>", parse_special=True)[0]
            frame = (start, pad, iend)
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self._turn([False, False], cancel, units=None):
                if cancel is not None and cancel.is_set():
                    raise ValueError("image preparation was cancelled")
                self.ensure_engine()                     # (the encoder borrows the engine's memory: it must run)
                with self.vision.active():
                    enc = []
                    for kind, src in media:
                        if cancel is not None and cancel.is_set():
                            raise ValueError("image preparation was cancelled")
                        enc.append((kind, self.vision.encode(src) if kind == "image"
                                    else self.vision.encode_video(src)))
                if cancel is not None and cancel.is_set():
                    raise ValueError("image preparation was cancelled")
            literal = self.tok.encode(image_pad, parse_special=False)
            encoded = [e for kind, e in enc if kind == "image"]
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start and k < len(encoded):
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
            # each <|video|> marker -> its frames, every frame an independent image wrapped in the image's own
            # markers (GLM-5.3 reads a video as a row of pictures; no timestamp text between the frames)
            videos = [e for kind, e in enc if kind == "video"]
            if videos:
                vliteral = self.tok.encode(video_pad, parse_special=False)
                out, k = [], 0
                for j, t in enumerate(ids):
                    if t == vp and j > 0 and ids[j - 1] == vstart and k < len(videos):
                        for n in videos[k][1]:
                            out += [frame[0]] + [pad] * n + [frame[2]]
                        k += 1
                    elif t == vp:
                        out += vliteral
                    else:
                        out.append(t)
                if k != len(videos):
                    raise ValueError("the prompt and its videos do not match")
                ids = out
            combined = self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            with open(combined, "wb") as f:
                for _, (path, _) in enc:
                    f.write(path.read_bytes())
            self.embeddings.path = combined
        room = self.engine.max_context - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ContextOverflow(f"maximum context length is {self.engine.max_context} tokens; prompt "
                                      f"({len(ids)} tokens) leaves no room to answer in the context; requests are "
                                      "never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ContextOverflow(f"maximum context length is {self.engine.max_context} tokens; prompt "
                                      f"({len(ids)} tokens) + max tokens ({max_new}) exceeds the context; requests "
                                      "are never truncated")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def count_tokens(self, req: dict) -> dict:
        """POST /api/tokens: how many tokens this chat request's prompt takes - the same template, thinking level,
        tools and tokenizer as prepare(), with nothing run (no image encoder: a picture or a video counts as its
        marker, so the count is approximate then).  The web app's context meter."""
        messages, tools, kw = openai_to_messages(req, deepseek=getattr(self.template, "deepseek", False))
        kw = {k: v for k, v in kw.items() if k != "_force_tool"}
        if req.get("strata_mcp") is True and self.mcp is not None:
            own = {t.get("name") for t in tools or []}
            tools = (tools or []) + self.mcp.template_tools(exclude=own) or None
        if self.default_effort and "reasoning_effort" not in kw and "enable_thinking" not in kw:
            kw = {**kw, **(deepseek_effort_kwargs(self.default_effort) if getattr(self.template, "deepseek", False)
                                  else effort_kwargs(self.default_effort))}
        if getattr(self.template, "deepseek", False) or isinstance(self.vision, mm3_vision.Vision):
            # (DeepSeek, MiniMax: the engine's own vision tower - a picture's span is its resize's)
            sources = (deepseek_images(messages) if getattr(self.template, "deepseek", False)
                       else [src for kind, src in media_of(messages) if kind == "image"])
            n, approximate = len(self.render_ids(messages, tools, kw)), bool(sources)
            if sources and isinstance(self.vision, ENGINE_VISION):
                try:
                    n += sum(self.vision.picture(src).tokens - 1 for src in sources)
                    approximate = False
                except (ValueError, OSError):
                    pass
            return {"tokens": n, "images": len(sources), "approximate": approximate,
                    "max_context": self.engine.max_context}
        media = media_of(messages)
        return {"tokens": len(self.render_ids(messages, tools, kw)), "images": len(media),
                "approximate": bool(media), "max_context": self.engine.max_context}

    def _note(self, n, evs, stat):
        with self.status_lock:
            s = self.status
            s["generated"] = n
            now = time.time()
            if stat.first_token is None:
                stat.first_token = now
                self.current_requests.add(stat)
            stat.note(now, n)                           # this request's window over the last RATE_WINDOW_S
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = (s["tail"] + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status)
            r = self.latest_request if self.latest_request in self.current_requests else None
        el = now - s.get("started", now)
        if r is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[maya] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(1e-6, now - r.first_token)
            print(f"[maya] {s['phase']}: {s['generated']} of max {s.get('max_tokens')} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    @contextlib.contextmanager
    def _turn(self, hold, cancel=None, units=1):
        """The engine's turn for one request: the FIFO, given back at the end unless a fair slice already did.  The
        place is taken (and given up) inside the try: an exception while waiting - a FIFO that throws instead of
        waiting - left it in the line for ever, and every request behind it waited for a turn nothing would take."""
        place = None
        try:
            if hold[0]:
                pass                                    # (the turn taken already: _queue_wait's, with keep-alives)
            elif cancel is None:
                self.fifo.acquire(units=units)
                hold[0] = True
            else:
                place = self.fifo.enqueue(units=units)
                while not cancel.is_set():
                    if self.fifo.wait_turn(place, timeout=0.05):
                        hold[0] = True
                        break
            yield
        finally:
            if place is not None:
                self.fifo.leave(place)
            if hold[0]:
                hold[0] = False
                self.fifo.release(units=units)

    def _queue_wait(self, hold, cancel, units=1) -> Iterator[tuple[str, object]]:
        """A request's wait in the FIFO for its first turn, as ("ping", None) every QUEUE_PING_S - the streams write
        them as SSE comments (": keep-alive"), which every client's parser ignores (#483: a request queued behind a
        long answer sent nothing at all, and a client with a stream-idle timeout gave up on a turn that was only
        waiting).  hold[0] is True once the turn is this request's (_turn then holds it and gives it back); a client
        that goes away meanwhile - the GeneratorExit at a ping - gives its place back here."""
        if cancel is None:                              # (no way to leave the line: _turn waits as before)
            return
        every = queue_ping_seconds()
        place = self.fifo.enqueue(units=units)
        try:
            last = time.monotonic()
            while not cancel.is_set():
                if self.fifo.wait_turn(place, timeout=0.05):
                    hold[0] = True
                    return
                if time.monotonic() - last >= every:
                    last = time.monotonic()
                    yield "ping", None
        finally:
            if not hold[0]:
                self.fifo.leave(place)

    @staticmethod
    def _resumed_sampling(sampling, produced):
        """The sampling for an engine call that goes on after `produced`: the thinking budget counts from where it got
        to (and is gone once the thinking was closed)."""
        tb = (sampling or {}).get("_think_budget")
        if not (isinstance(tb, int) and tb > 0):
            return sampling
        smp = dict(sampling)
        if (sampling or {}).get("_think_end") in produced:
            smp.pop("_think_budget", None)
            smp.pop("_think_end", None)
        else:
            smp["_think_budget"] = max(1, tb - len(produced))
        return smp

    def _loop_guard_on(self, sampling) -> bool:
        """The loop guard for this request: its own "loop_guard" when it sends one, else the server's."""
        own = (sampling or {}).get("loop_guard")
        return own if isinstance(own, bool) else bool(self.loop_guard)

    def _loop_noted(self, act, written: int):
        """A loop the guard broke: a line in the server window, and the Monitor's count."""
        print(f"[maya] loop guard: {act.describe()}"
              f"{' (now ' + format(written, ',') + ' tokens)' if act.kind != loop_guard.STOP else ''}", flush=True)
        with self.status_lock:
            c = self.loop_guard_counts
            c["detected"] += 1
            c[{loop_guard.PENALTY: "penalised", loop_guard.CLOSE_THINKING: "thinking_closed",
               loop_guard.STOP: "stopped"}[act.kind]] += 1

    def _sliced(self, ids, max_new, sampling, cancel, emb, hold, stat, guard=None) -> Iterator[int | None]:
        """engine.generate(), in fair slices (STRATA_FAIR_SLICE_S): once this answer has been decoding that long and
        another request waits for the engine, the engine stops here, the FIFO goes to the next request and this one
        queues behind it.  Back in turn, it goes on with prompt + what it already wrote as the new prompt: the
        engine's conversation slots hold this prompt's state, so only the tokens written since are read again (a
        prompt read is ~30x a decode step's speed).  Concurrent chats then all move on instead of waiting behind the
        longest answer, and a short side request (an agent's title, a summary) no longer waits for a whole turn.
        The caller sees one stream of token ids (None: a heartbeat, also while queued again).  `hold[0]` says
        whether this request holds the FIFO now.  Not for image prompts: their state is never kept in a slot.

        With the loop guard (`guard`, serve/loop_guard.py) every id goes through it and the caller gets only what
        lies before the tail it holds back (what could still be a loop's repeats).  A loop restarts the engine the
        same way, at once and still holding the FIFO: prompt + what is kept (the repeats dropped; after a second
        loop in the thinking, the model's end-of-thinking token too) with a short repetition penalty, which a call
        of its own ends and the request's own sampling follows; or the answer ends there (LOOP_STOP).  Until then the
        engine calls and their sampling are the ones without it."""
        produced, cur, left, smp = [], list(ids), max_new, sampling
        segments = []                                   # each slice's DONE (engine.last), summed at the end
        sent = 0                                        # produced[:sent] went to the caller (the guard holds the rest)
        while True:
            last0 = getattr(self.engine, "last", None)
            smp_call, n_call = guard.sampling(smp, len(produced), left) if guard is not None else (smp, left)
            gen = self.engine.generate(cur, n_call, smp_call, cancel, embeddings=emb) if emb else \
                self.engine.generate(cur, n_call, smp_call, cancel)
            slice_s, t0, cut, pausing = fair_slice_seconds(), None, False, False
            act, got = None, 0                          # the loop guard's action; this call's tokens
            try:
                for t in gen:
                    if guard is None:
                        yield t
                    elif t is None:
                        yield None
                    if t is None:
                        continue
                    produced.append(t)
                    if guard is not None:
                        got += 1
                        act = guard.push(t, len(produced))
                        if act is not None:
                            break                       # (gen.close(): STOP + drain, still holding the FIFO)
                        while sent < len(produced) - guard.hold:
                            sent += 1
                            yield produced[sent - 1]
                    if t0 is None:
                        t0 = time.monotonic()
                    if slice_s > 0 and not pausing and not emb and not self.stopping and len(produced) < max_new \
                            and self.fifo.waiting() and time.monotonic() - t0 >= slice_s:
                        if getattr(self.engine, "can_pause", False):
                            # PAUSE: the engine keeps the state with what it wrote; the tokens it still writes
                            # before its DONE go to the client as usual, so the next prompt continues that state
                            self.engine.pause(keep=len(cur) == len(ids))   # its first slice
                            pausing = True
                        else:
                            cut = True
                            break
                if pausing:
                    done = getattr(self.engine, "last", None)
                    cut = len(produced) < max_new and (done is last0 or (done or {}).get("finish") == "cancel")
            finally:
                gen.close()                             # STOP + drain to this slice's DONE, still holding the FIFO
                fresh = getattr(self.engine, "last", None)
                if fresh is not None and fresh is not last0:
                    segments.append(fresh)
            if act is not None:                         # a loop: what is kept, then on from there - or the end
                del produced[max(sent, len(produced) - act.drop):]   # (what was sent stays: the hold-back covers it)
                if act.kind == loop_guard.CLOSE_THINKING:
                    produced.append(guard.think_end)
                guard.resume(produced, act)
                self._loop_noted(act, len(produced))
                if act.kind == loop_guard.STOP or len(produced) >= max_new or cancel.is_set() or self.stopping:
                    break
                cur, left = list(ids) + produced, max_new - len(produced)
                smp = self._resumed_sampling(sampling, produced)
                continue
            if not cut:
                if guard is not None and n_call < left and got >= n_call and produced[-1] not in self.stop_ids \
                        and not cancel.is_set() and not self.stopping:
                    # the guard's penalty ended that call: on with the request's own sampling, still holding the FIFO
                    cur, left = list(ids) + produced, max_new - len(produced)
                    smp = self._resumed_sampling(sampling, produced)
                    continue
                break
            with self.status_lock:
                mine = dict(self.status)
                stat.clear()                            # queued again: not decoding, so no part of the live rate
                self.active -= 1
                hold[1] = False
                self.status["busy"] = self.active > 0
                self.status["queued"] += 1
            waiting = self.fifo.waiting()
            place = self.fifo.enqueue(units=1)          # in line before letting go: behind whoever waits now
            hold[0] = False
            self.fifo.release(units=1)
            print(f"[maya] fair slice: {len(produced)} tokens written, the engine goes to the next request "
                  f"({waiting} waiting); this one continues after it", flush=True)
            try:
                while not self.fifo.wait_turn(place, timeout=0.1):
                    if cancel.is_set() or self.stopping:
                        return
                    yield None                          # keeps the client's watchdog calm while queued again
                hold[0] = True
            finally:
                if not hold[0]:                         # the client went away while queued again
                    self.fifo.leave(place)
                with self.status_lock:
                    self.status["queued"] -= 1
            with self.status_lock:
                self.active += 1
                hold[1] = True
                self.status.update({k: mine[k] for k in ("busy", "phase", "prompt_tokens", "started", "tool", "tail",
                                                         "max_tokens") if k in mine},
                                   generated=len(produced))
            if self.stopping or cancel.is_set():
                break
            if hasattr(self.engine, "alive") and not self.engine.alive():   # it died in the request in between
                print("[maya] the engine had stopped; starting it again (a minute or two) ...", flush=True)
                self.engine.restart()
            cur, left = list(ids) + produced, max_new - len(produced)
            smp = self._resumed_sampling(sampling, produced)   # (the thinking budget counts from where it got to)
        if len(segments) > 1:                           # the request's figures, over all its slices
            first, last = segments[0], segments[-1]
            merged = dict(last, prompt_tokens=first.get("prompt_tokens"), reused=first.get("reused"))
            for k in ("generated", "prompt_ms", "decode_ms", "drafts_accepted", "drafts_offered", "hits", "lookups"):
                if all(k in sg for sg in segments):
                    merged[k] = sum(sg[k] for sg in segments)
            self.engine.last = merged
        lastd = getattr(self.engine, "last", None)
        if isinstance(lastd, dict) and (lastd.get("prompt_tokens") or 0) > len(ids):
            self.engine.last = own_prompt_figures(lastd, len(ids))   # (the first call's DONE was not seen)
        if guard is not None and not cancel.is_set():   # the end: what the guard still held back
            while sent < len(produced):
                sent += 1
                yield produced[sent - 1]
            if guard.events and guard.events[-1].kind == loop_guard.STOP:
                yield LOOP_STOP                         # (run(): the answer ends here, finish "stop")

    def run(self, ids, thinking, tools, max_new, sampling, cancel) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..})."""
        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        responses = (sampling or {}).get("_responses") is True
        single_tool = responses and (sampling or {}).get("parallel_tool_calls") is False
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True,
                              dsml=getattr(self.template, "deepseek", False),
                              minimax=getattr(self.template, "minimax", False),
                              strict_tools=responses, single_tool=single_tool)
        prefill = (sampling or {}).get("_prefill") or ""   # the call's opening, already in the prompt (prepare)
        pre_events = parser.feed(prefill) if prefill else []
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        reasoning_tokens = 0
        held = ""                                       # text that may be the start of a stop marker (stop_texts)
        # the client's own stop strings (OpenAI "stop", Anthropic "stop_sequences"): the answer ends before one.  They
        # apply to the answer, not inside the reasoning; the model's own end-of-turn markers apply everywhere
        client_stops = client_stop_strings(sampling)
        matched = None                                  # the client stop string that ended the answer
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        guards = []                                     # this request's loop guard (the Monitor's row says what it did)
        emb = getattr(self.embeddings, "path", None)
        # Identity token: only a DONE line replaces engine.last, so a request that died, errored or was
        # disconnected must not have the PREVIOUS request's decode figures recorded as its own.
        engine_last0 = getattr(self.engine, "last", None)
        counted, my_started, my_first, queued = False, time.time(), None, False
        stat = RequestStat()                            # this request's live figures
        with self.status_lock:
            self.status["queued"] += 1
            queued = True                               # counted as waiting until its turn comes (below, or in finally)
        hold = [False, False]                           # holds the FIFO (fair slices let go of it); counts as active
        asked = time.monotonic()
        try:
            yield from self._queue_wait(hold, cancel)   # (#483: keep-alives while queued; the turn taken there)
            with self._turn(hold, cancel):
                with self.status_lock:
                    self.status["queued"] -= 1
                    queued = False
                waited = time.monotonic() - asked
                if waited >= QUEUE_LOG_S:               # a long wait is worth a line: it is what a wedged FIFO looked like
                    print(f"[maya] this request waited {waited:.0f} s for the engine "
                          f"({self.fifo.waiting()} request(s) behind it)", flush=True)
                if cancel.is_set():
                    yield "done", {"finish": "cancel", "completion_tokens": 0}
                    return
                if self.stopping:
                    raise ValueError("the server is stopping: send the request again once the model is back")
                if self.vision is not None and self.vision.owed:   # memory an encode borrowed, not back yet
                    self.vision.settle()
                self.ensure_engine()   # after an idle unload, or issue #27: it died in an earlier request
                with self.status_lock:
                    self.status.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids), generated=0,
                                       started=time.time(), tool=None, tail="", max_tokens=max_new)
                    self.latest_request = stat
                    self.active += 1
                    counted = hold[1] = True
                    my_started = self.status["started"]
                    self.last_request_at = time.time()
                before = getattr(self.engine, "last", None)
                last_print = time.time()
                # the thinking budget goes to the engine with the request: a reasoning block still open after that many
                # tokens is closed there (its next token is </think>) and the answer follows in the same decode
                in_think = bool(thinking) and self.think_end_id is not None and self.think_budget > 0
                # never let the reasoning use the whole output: a request whose max_tokens is reached while still
                # thinking ends with no answer at all. Reserve a quarter for the answer, with a preferred floor
                # of 1024 (STRATA_GLM_THINK_RESERVE). The floor cannot take more than half of a small output limit:
                # the old unconditional floor reduced every limit <=1024 to ONE reasoning token. The forced
                # </think> also consumes one token of max_new, so account for it before assigning the budget.
                think_budget = self.think_budget
                if in_think and max_new:
                    reserve_floor = max(0, int(os.environ.get("STRATA_GLM_THINK_RESERVE", "1024") or 1024))
                    reserve = max(int(max_new) // 4, min(reserve_floor, max(1, int(max_new) // 2)))
                    think_budget = max(1, min(think_budget, int(max_new) - reserve - 1))
                if in_think:
                    sampling = {**(sampling or {}), "_think_budget": think_budget, "_think_end": self.think_end_id}

                def attempts():
                    """The engine's tokens.  When its numbers went non-finite (a NaN or an infinity in a route: the
                    engine's ERR says "non-finite") before the first token, the engine has dropped that conversation's
                    state and the request runs once more, its prompt read from token 0."""
                    for again in (False, True):
                        guard = loop_guard.LoopGuard(self.think_end_id, parser.state == "reasoning") \
                            if guard_on else None       # (serve/loop_guard.py; a fresh one for a run again)
                        guards[:] = [guard] if guard is not None else []
                        sliced, got = self._sliced(ids, max_new, sampling, cancel, emb, hold, stat, guard), 0
                        try:
                            for t in sliced:
                                got += t is not None
                                yield t
                            return
                        except ValueError as e:
                            if again or got or NONFINITE not in str(e):
                                raise
                            print(f"[maya] the engine's numbers went non-finite before the answer started ({e}): "
                                  "running the request again from token 0", flush=True)
                        finally:
                            sliced.close()
                guard_on = self._loop_guard_on(sampling)
                gen = attempts()
                for ev in pre_events:                   # a required call's opening (prepare put it in the prompt)
                    yield "event", ev
                try:
                    for t in gen:
                        if t is None:                   # heartbeat while the engine is quiet
                            last_print = self._progress(last_print)
                            pr = getattr(self.engine, "progress", None)   # (read, total, reuse, ms), the PP parse
                            yield "ping", ({"total": pr[1], "cache": pr[2], "processed": pr[0],
                                            "time_ms": round(pr[3])} if pr and len(pr) >= 4 else None)
                            continue
                        if t is LOOP_STOP:              # the loop guard ended the answer (a loop that persisted)
                            finish = "stop"
                            break
                        n += 1
                        if my_first is None:
                            my_first = time.time()
                        if in_think and t == self.think_end_id:
                            in_think = False
                            if n == think_budget + 1:
                                print(f"[maya] thinking reached its budget ({think_budget} tokens): closed, "
                                      "answering", flush=True)
                        if t in self.stop_ids:
                            finish = "stop"
                            raw_ids.append(t)
                            break
                        if parser.state == "reasoning":
                            reasoning_tokens += 1
                        raw_ids.append(t)
                        text, stopped = detok.push(t), False
                        stops = self.stop_texts + (client_stops if parser.state != "reasoning" else [])
                        if stops:
                            held += text
                            hits = [(i, s) for s, i in ((s, held.find(s)) for s in stops) if i >= 0]
                            if hits:
                                cut, s = min(hits)
                                text, held, stopped = held[:cut], "", True
                                matched = s if s in client_stops and s not in self.stop_texts else None
                            else:               # keep back the longest tail that could still grow into a marker
                                k = max((j for s in stops for j in range(1, len(s))
                                         if held.endswith(s[:j])), default=0)
                                text, held = held[:len(held) - k], held[len(held) - k:]
                        evs = parser.feed(text)
                        self._note(n, evs, stat)
                        last_print = self._progress(last_print)
                        for ev in evs:
                            yield "event", ev
                            if single_tool and ev.kind == "tool_call" and ev.complete:
                                stopped = True
                                break
                        if stopped:
                            finish = "stop"
                            break
                    if held:                    # the run ended on what only looked like a marker's start
                        for ev in parser.feed(held):
                            yield "event", ev
                        held = ""
                    if cancel.is_set():
                        finish = "cancel"
                except EngineDied as e:
                    finish = "error"
                    note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
                    print(f"[maya] {e}. {note} The next request starts the engine again."
                          f"{' Its log: ' + self.engine.log_path if getattr(self.engine, 'log_path', None) else ''}",
                          flush=True)
                    raise
                except ValueError as e:                 # the engine's ERR line (it may have ended after it)
                    if NONFINITE in str(e) and n > 0:
                        # an answer already under way: it ends here (what was sent stands), and the engine has
                        # dropped the conversation's state, so the next turn reads it from token 0
                        finish = "length"
                        print(f"[maya] the engine's numbers went non-finite after {n} tokens ({e}): the answer ends "
                              "there", flush=True)
                        if held:
                            for ev in parser.feed(held):
                                yield "event", ev
                            held = ""
                    else:
                        finish = "error"
                        print(f"[maya] the engine reported an error: {e}", flush=True)
                        raise
                finally:
                    gen.close()                         # STOP+drain to THIS request's DONE while still holding the
                    #                                     fifo, so a stop-token break can't leave the shared engine
                    #                                     queue mid-drain for the next request to read as its own DONE
        except GeneratorExit:                           # the client disconnected mid-stream
            finish = "disconnect"
            raise
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
            book = None                                 # (the savings' count: after the lock, it writes a file)
            with self.status_lock:
                self.current_requests.discard(stat)
                if counted:
                    # only this request's DONE counts: same object means no DONE arrived (death, error, disconnect)
                    last = dict(getattr(self.engine, "last", {}) or {}) \
                        if getattr(self.engine, "last", None) is not engine_last0 else {}
                    started = my_started
                    loaded = str((getattr(self.engine, "info", {}) or {}).get("cvec", 0)) not in ("0", "", "None")
                    hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                    self.history.append({
                        "projection": (sampling or {}).get("experimental_speed_projection") is not False
                        if loaded else None,
                        "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                        "api": (sampling or {}).get("_api"),
                        "prefill_tok_s": round((len(ids) - (last.get("reused") or 0)) / (last["prompt_ms"] / 1000), 1)
                        if last.get("prompt_ms") and len(ids) > (last.get("reused") or 0) else None,
                        "prompt_tokens": len(ids), "reused": last.get("reused"), "output_tokens": n,
                        "engine_generated": last.get("generated"),
                        "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                        "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                        if n and last.get("generated") and last.get("decode_ms") else None,
                        "hit_rate": hit_rate,
                        "loop_guard": guards[0].summary() if guards and guards[0].events else None})
                    t = self.totals
                    t["requests"] += 1
                    t["prompt_tokens"] += len(ids)
                    t["reused"] += last.get("reused") or 0
                    t["output_tokens"] += n
                    book = (len(ids), last.get("reused") or 0, n)
                    t["prompt_ms"] += last.get("prompt_ms") or 0.0
                    t["decode_ms"] += last.get("decode_ms") or 0.0
                    fresh = getattr(self.engine, "last", None)
                    if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                        timings = request_timings(len(ids), n, last)
                        self.last_timings = dict(timings, at=int(time.time())) if timings else None
                    self.last_request_at = time.time()
                    now = time.time()
                    el = now - my_started
                    ft = my_first
                    # the engine's decode clock when there is one (the Monitor's row); else the tokens after the first
                    # over the time since it - `n / span` read a one-token answer as thousands of tok/s
                    rate = last["generated"] / (last["decode_ms"] / 1000) if n and last.get("generated") and \
                        last.get("decode_ms") else (n - 1) / (now - ft) if ft and n > 1 and now > ft else 0.0
                    hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                    print(f"[maya] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "
                          f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                    if os.environ.get("STRATA_DEBUG") and raw_ids:
                        print(f"[maya] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                if queued:                              # never started (a cancel, an error while waiting): not waiting now
                    self.status["queued"] -= 1
                if hold[1]:
                    hold[1] = False
                    self.active -= 1
                self.status["busy"] = self.active > 0
            if book is not None and self.savings is not None:
                self.savings.add(*book)
        for ev in parser.finish():
            yield "event", ev
        yield "done", {"finish": finish, "completion_tokens": n, "reasoning_tokens": reasoning_tokens,
                       "reused": (timings or {}).get("cache_n", 0),
                       "timings": timings, "stop_sequence": matched if finish == "stop" else None}


def own_prompt_figures(last: dict, n_prompt: int) -> dict:
    """A request's figures when its last engine call went on from what it had written (after the loop guard or a fair
    slice): that call's prompt was the request's prompt plus the tokens kept, all of them in the conversation cache,
    so its prompt_tokens and reused count those too - a guarded answer reported 5,919 cached tokens of a 246-token
    prompt.  The request's own: its prompt, and what the cache held of it (the call's reused less the kept tokens)."""
    extra = int(last.get("prompt_tokens") or 0) - n_prompt
    reused = max(0, min(n_prompt, int(last.get("reused") or 0) - max(0, extra)))
    return dict(last, prompt_tokens=n_prompt, reused=reused)


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[maya] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = int(hub.settings["max_rounds"])
    total, rounds, done = 0, 0, None
    messages = list(messages)
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, cancel):
            if kind == "done":
                done = x
                continue
            if kind == "event":
                ev: Event = x
                if ev.call is not None and ev.call.name in mcp_names:
                    if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                        yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                    elif ev.kind == "tool_call":
                        calls.append(ev.call)
                    continue                             # its argument pieces are not streamed to the client
                if ev.kind == "tool_call":
                    own_calls += 1
                elif ev.kind == "content":
                    text.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning.append(ev.text)
            yield kind, x
        total += done["completion_tokens"]
        run_them = calls and not own_calls and done["finish"] == "stop" and not cancel.is_set()
        if run_them and rounds >= max_rounds:
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
            run_them = False
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    if worker.is_alive():
                        yield "ping", None
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[maya] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r["text"])
            yield "mcp", {"event": "result", "id": c.id, **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        deepseek = getattr(svc.template, "deepseek", False)
        messages.append({"role": "assistant", "content": "".join(text) if deepseek else "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning) if deepseek else "".join(reasoning).strip()}
                            if reasoning else {}),
                         "tool_calls": [{"id": c.id, "function": {"name": c.name, "arguments": c.arguments}}
                                        for c in calls]})
        messages += [{"role": "tool", "tool_call_id": c.id, "content": r} for c, r in zip(calls, results)]
        # a required tool call applies to the first answer only: after the tools ran, the model answers freely
        kw = {k: v for k, v in kw.items() if k not in ("_force_tool", "_prefill")}
        sampling = {k: v for k, v in (sampling or {}).items() if k != "_prefill"}
        ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req)
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
def json_from_text(text: str) -> str:
    """The JSON in an answer that was asked for JSON: the answer itself when it parses, else the inside of its first
    code fence, else the first object or array in it; the text as it was when none parses."""
    s = text.strip()
    candidates = [s]
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", s, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    for c in candidates:
        try:
            json.loads(c)
            return c
        except ValueError:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", s):
        try:
            _, end = dec.raw_decode(s, m.start())
            return s[m.start():end]
        except ValueError:
            continue
    return text


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    # response_format json_object / json_schema: the answer is held and its JSON sent alone at the end (Strata #762)
    rf = req.get("response_format")
    json_mode = isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")
    held = []
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel):
        # ^ kind "ping": x carries the prompt's progress (total, cache, processed, time_ms) while the engine reads,
        #   or None while this request just waits (queued again, an MCP round's engine work)
        if kind == "ping":
            if req.get("return_progress") and req.get("stream") and isinstance(x, dict):
                c = chunk({})
                c["prompt_progress"] = x          # llama.cpp's field: the prompt read's progress, for a client's bar
                yield c
            else:
                yield None
        elif kind == "mcp":
            c = chunk({})
            c["strata_mcp"] = x
            yield c
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                if json_mode:
                    held.append(ev.text)
                    continue
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], **ev.call.to_openai(arguments="")}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                continue
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, **ev.call.to_openai()}]})
                calls += 1
        else:
            if held:
                yield chunk({"content": json_from_text("".join(held))})
            finish = "tool_calls" if calls and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                             "total_tokens": pt + x["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            yield last


def carry_embeddings(local, path, chunks):
    """`Service.embeddings` is thread-local (the request thread prepares the prompt and sets the image embeddings
    file).  A strata_resume answer runs its chunks on the job thread, which would see no file: the image's place
    in the prompt then went to the engine as plain pad tokens and the model answered about a black / blank picture."""
    local.path = path                                  # runs on the thread that first pulls from the generator
    yield from chunks


def watch_client(handler, cancel: threading.Event, every: float = 0.5) -> None:
    """A request's client watched on its own thread (#314): an API request writes nothing while it waits in the queue,
    and a non-streamed one nothing until its answer is done, so a client that timed out or went away was only noticed
    at its first write - queued requests nobody waited for any more piled up for hours, and each then ran its whole
    answer.  A connection the client closed (or reset) reads as end-of-file: `cancel` is set, the request leaves the
    queue (or its engine work stops).  Data the client sent meanwhile (its next request on a kept-alive connection) is
    not an end.  The watch ends with the request's cancel, the connection, or the next request on that connection
    (one watch a connection)."""
    import select
    import socket
    conn, token = handler.connection, object()
    handler._client_watch = token

    def watch():
        while not cancel.wait(every) and getattr(handler, "_client_watch", None) is token:
            try:
                ready, _, _ = select.select([conn], [], [], 0)
                if ready and conn.recv(1, socket.MSG_PEEK) == b"":
                    cancel.set()
                    return
            except ConnectionError:                     # (reset by the client: gone)
                cancel.set()
                return
            except (OSError, ValueError):               # (closed by this server: the request is done)
                return
    threading.Thread(target=watch, name="client-watch", daemon=True).start()


class StreamJob:
    """A web-app answer that outlives its connection (the request's `strata_resume`): a thread runs its chunks into
    `items` and any reader streams them from an index (GET /v1/strata/stream?id=&from=), so a phone whose screen
    sleeps mid-answer reconnects and carries on instead of losing it.  Only POST /v1/strata/cancel (the app's Stop)
    cancels it - a dropped connection does not."""

    ORPHAN_S = 300   # a running answer nobody has read for this long (the page was closed) is cancelled

    def __init__(self, jid: str, chunks, cancel: threading.Event):
        self.id, self.cancel = jid, cancel
        self.items: list = []
        self.done, self.ended = False, None
        self.readers, self.unread_since = 0, time.time()
        self.cv = threading.Condition()
        threading.Thread(target=self._run, args=(chunks,), daemon=True, name="strata-job").start()
        threading.Thread(target=self._watch, daemon=True, name="strata-job-watch").start()

    def reader(self, on: bool):
        with self.cv:
            self.readers += 1 if on else -1
            if self.readers == 0:
                self.unread_since = time.time()

    def _watch(self):
        while not self.done:
            time.sleep(5)
            with self.cv:
                orphan = not self.done and self.readers == 0 and time.time() - self.unread_since > self.ORPHAN_S
            if orphan:
                print(f"[maya] nobody read answer {self.id} for {self.ORPHAN_S} s: stopped", flush=True)
                self.cancel.set()
                return

    def _add(self, c):
        with self.cv:
            self.items.append(c)
            self.cv.notify_all()

    def _run(self, chunks):
        try:
            for c in chunks:
                if c is not None:                      # heartbeats: each reader writes its own
                    self._add(c)
        except EngineDied as e:
            self._add({"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
        except Exception as e:  # noqa: BLE001 - the engine's ERR after the stream started, or anything else
            self._add({"error": {"type": "server_error", "message": str(e)}})
        finally:
            with self.cv:
                self.done, self.ended = True, time.time()
                self.cv.notify_all()

    def wait(self, n: int, timeout: float):
        """The items from index n (waiting up to `timeout` for one) and whether the answer is complete."""
        with self.cv:
            if len(self.items) <= n and not self.done:
                self.cv.wait(timeout)
            return self.items[n:], self.done


JOBS: dict[str, StreamJob] = {}
JOBS_LOCK = threading.Lock()


def jobs_put(job: StreamJob) -> None:
    """Keeps the live answers and the finished ones of the last 15 minutes (at most 8)."""
    with JOBS_LOCK:
        now = time.time()
        for j in JOBS.values():                    # a new answer: one still running with nobody reading it goes
            if not j.done and j.readers == 0:
                j.cancel.set()
        for k in [k for k, j in JOBS.items() if j.done and now - (j.ended or now) > 900]:
            del JOBS[k]
        while len(JOBS) >= 8:
            old = min(JOBS.values(), key=lambda j: (not j.done, j.ended or now))
            del JOBS[old.id]
        JOBS[job.id] = job


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    for c in chunks:
        if c is None or c.get("prompt_progress"):  # a heartbeat, a prompt-progress chunk: not answer content
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            if tc.get("namespace") is not None:
                cur["namespace"] = tc["namespace"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    streamed = set()
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
        if kind == "ping":
            yield None
            continue
        if kind == "event":
            ev: Event = x
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield close()
            stop = "tool_use" if used_tool and x["finish"] == "stop" else \
                "stop_sequence" if x.get("stop_sequence") else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts, Anthropic's way: input_tokens leaves out what the conversation cache already held,
            # which is cache_read_input_tokens (message_start could only say the whole prompt)
            reused = min(x.get("reused") or 0, len(ids))
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop,
                                                                       "stop_sequence": x.get("stop_sequence")},
                                    "usage": {"input_tokens": len(ids) - reused, "cache_read_input_tokens": reused,
                                              "output_tokens": x["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            b["input"] = json.loads(b.pop("_json") or "{}")
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
def _backlog() -> int:
    try:
        return max(5, int(os.environ.get("STRATA_HTTP_BACKLOG") or 256))
    except ValueError:
        return 256


def body_limit() -> int:
    """The most a request body may hold, in bytes (it is read into memory): 256 MiB, a million-token conversation with
    room to spare; STRATA_MAX_BODY_MIB changes it.  A larger one is answered 413 before it is read (Strata #893)."""
    try:
        mib = int(os.environ.get("STRATA_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return (mib if mib > 0 else 256) << 20


class BadBody(Exception):
    """A request body that cannot be read (a bad Content-Length, a malformed or oversized chunked body)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def key_list(value) -> list[str]:
    """The API keys the server accepts (Strata #1344): llama.cpp's form, several separated by commas ("k1,k2"), or a
    list in the config ("api_key": ["k1", "k2"], for a key that contains a comma)."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def parse_cors_origins(value) -> list[str]:
    """The config's "cors_origins": a list of page origins such as "http://127.0.0.1:8090" (or ["*"]) -> the list
    without trailing slashes; missing or empty = none (no CORS headers at all).  A ValueError says what is wrong."""
    if value is None or value == "":
        return []
    if not isinstance(value, list) or not all(isinstance(o, str) and o.strip() for o in value):
        raise ValueError('"cors_origins" must be a list of origins like "http://127.0.0.1:8090" (or ["*"])')
    return [o.strip().rstrip("/") for o in value]


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection
        answer_started = False                              # a malformed request's 400 only replaces an answer not begun

        def log_message(self, fmt, *args):
            pass

        def send_response(self, code, message=None):
            self.answer_started = True
            super().send_response(code, message)

        def _body(self) -> bytes:
            """The request body: Content-Length checked first (a bad one is a 400, one over body_limit() a 413 before
            anything is read), or a Transfer-Encoding: chunked one from a relay that does not buffer it."""
            limit = body_limit()
            te = self.headers.get("Transfer-Encoding", "") or ""
            if te.split(",")[-1].strip().lower() == "chunked":
                parts, total = [], 0
                while True:
                    line = self.rfile.readline(1025)
                    try:
                        size = int(line.split(b";", 1)[0].strip(), 16)
                    except ValueError:
                        raise BadBody(400, "malformed chunked request body") from None
                    if size < 0 or not line.endswith(b"\n"):
                        raise BadBody(400, "malformed chunked request body")
                    if size == 0:
                        for _ in range(64):                  # the trailers, up to the blank line
                            if self.rfile.readline(8193).strip() == b"":
                                break
                        return b"".join(parts)
                    total += size
                    if total > limit:
                        self.close_connection = True
                        raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                           "(STRATA_MAX_BODY_MIB raises the limit)")
                    piece = self.rfile.read(size)
                    if len(piece) != size or self.rfile.read(2) != b"\r\n":
                        raise BadBody(400, "malformed chunked request body")
                    parts.append(piece)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True                 # not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(STRATA_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _cors_allow(self):
            """The Access-Control-Allow-Origin value for this request, or None: only /v1/* paths and only an origin the
            config's cors_origins lists (or "*")."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return None
            if "*" in svc.cors_origins:
                return "*"
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return origin if origin and origin in svc.cors_origins else None

        def end_headers(self):
            # every answer (JSON, a stream, an error) to an allowed page carries the CORS header - one place for all
            allow = self._cors_allow()
            if allow:
                self.send_header("Access-Control-Allow-Origin", allow)
                if allow != "*":
                    self.send_header("Vary", "Origin")
            super().end_headers()

        def do_OPTIONS(self):
            # a CORS preflight: no credentials come with it, so no API key; granted only for cors_origins
            allow = self._cors_allow()
            self.send_response(204 if allow else 403)
            if allow:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                asked = self.headers.get("Access-Control-Request-Headers")
                self.send_header("Access-Control-Allow-Headers", asked or "Content-Type, Authorization, x-api-key")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if given and any(hmac.compare_digest(given.encode(), k.encode()) for k in key_list(svc.api_key)):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _response_owner(self):
            if not svc.api_key:
                return "public"
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            return hashlib.sha256(given.encode()).hexdigest()

        def _response_lookup(self, owner, response_id):
            try:
                return svc.response_jobs.get(owner, response_id)
            except KeyError:
                return svc.responses.get(owner, response_id)

        def _background_stream(self, owner, response_id, after=-1):
            try:
                svc.response_jobs.check_cursor(owner, response_id, after)
            except KeyError:
                raise JobError(409, "background event history is no longer retained") from None
            with svc.response_jobs.reader(owner, response_id, after) as events:
                self._write_background_stream(events, after)

        def _write_background_stream(self, events, after):
            self.connection.settimeout(svc.response_limits["STREAM_TIMEOUT_S"])
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, payload = item
                        after = payload["sequence_number"]
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(payload, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except (JobError, KeyError) as exc:
                payload = {"type": "error", "code": "event_history_unavailable", "param": None,
                           "message": str(exc), "sequence_number": after + 1}
                try:
                    self.wfile.write(b"event: error\ndata: " + json.dumps(payload).encode() + b"\n\n")
                except OSError:
                    pass
            except OSError:
                pass
            finally:
                events.close()

        def _response_resource(self, delete=False):
            if not self._authorized():
                return
            parts = self.path.split("?")[0].rstrip("/").removeprefix("/v1/responses/").split("/")
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            if len(parts) > 2 or (len(parts) == 2 and (parts[1] != "input_items" or delete)):
                return self._json(404, {"error": {"message": "not found"}})
            try:
                owner = self._response_owner()
                if delete:
                    if query:
                        raise ValueError("response deletion does not support query options")
                    removed = False
                    for store in (svc.response_jobs, svc.responses):
                        try:
                            store.delete(owner, parts[0])
                            removed = True
                        except KeyError:
                            pass
                    if not removed:
                        raise KeyError(parts[0])
                    return self._json(200, {"id": parts[0], "object": "response.deleted", "deleted": True})
                stored = self._response_lookup(owner, parts[0])
                if len(parts) == 1 and query:
                    if (set(query) - {"stream", "starting_after"} or query.get("stream") != ["true"] or
                            any(len(value) != 1 for value in query.values())):
                        raise ValueError("retrieve accepts stream=true and optional starting_after")
                    after = int(query["starting_after"][0]) if "starting_after" in query else -1
                    if "starting_after" in query and after < 0:
                        raise ValueError("starting_after must be nonnegative")
                    return self._background_stream(owner, parts[0], after)
                result = input_page(stored["input"], query) if len(parts) == 2 else stored["response"]
                self._json(200, result)
            except KeyError:
                self._json(404, {"error": {"type": "invalid_request_error", "message": "stored response not found"}})
            except (JobError, StoreBusy) as exc:
                self._json(getattr(exc, "status", 503), {"error": {"type": "server_error", "message": str(exc)}})
            except ValueError as exc:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(exc)}})

        def _response_cancel(self, path):
            parts = path.removeprefix("/v1/responses/").split("/")
            if len(parts) != 2 or parts[1] != "cancel" or "?" in self.path:
                return self._json(404, {"error": {"message": "not found"}})
            owner = self._response_owner()
            try:
                out = svc.response_jobs.cancel(owner, parts[0])
            except KeyError:
                try:
                    out = self._response_lookup(owner, parts[0])["response"]
                except KeyError:
                    return self._json(404, {"error": {"message": "background response not found"}})
                if not out.get("background"):
                    raise ValueError("only background responses can be cancelled")
            self._json(200, out)

        def do_DELETE(self):
            if self.path.split("?")[0].startswith("/v1/responses/"):
                self._response_resource(delete=True)
            elif self._authorized():
                self._json(404, {"error": {"message": "not found"}})

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/v1/responses/"):
                self._response_resource()
                return
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: serve/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (serve/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics":
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/metrics/prometheus":
                if self._authorized():
                    body = svc.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "":
                body = (ROOT / "serve" / "web" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                # the run config's recommended settings: the web app's Chat defaults until its user picks their own
                rec = {k: v for k, v in svc.sampling_defaults.items() if k in ("temperature", "top_p", "top_k")}
                if svc.default_effort:   # (MiniMax's by its own five names)
                    rec["reasoning_effort"] = (effort_name if getattr(svc.template, "minimax", False) else effort_level)(
                        svc.default_effort) or svc.default_effort
                self._json(200, {"status": "updating" if svc.updater.busy() else
                                           "reloading" if svc.reloading() else "ok",
                                 "max_context": svc.engine.max_context, "model": svc.model,
                                 "model_label": svc.model_label,   # (the model first: "GLM-5.3 · Maya-S24")
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key), "defaults": rec,
                                 "version": MAYA_VERSION, "mcp_tools": bool(svc.mcp), "unloaded": svc.unloaded})
            elif path == "/api/context":
                # the dashboard's Context size: the range, the measured cost of the current size, a reload's state
                if self._authorized():
                    self._json(200, svc.context_limits())
            elif path == "/api/port":
                # the dashboard's Port: where the server listens, and whether a new one applies at once
                if self._authorized():
                    self._json(200, svc.port_info())
            elif path == "/api/models":
                # the Models page: the configs the setup wrote, the running one, each one's engine settings
                if self._authorized():
                    self._json(200, svc.models_info())
            elif path == "/api/savings":
                # the sidebar's Saved: what the model's own API would have charged for the same tokens, per model
                if self._authorized():
                    self._json(200, svc.savings.report() if svc.savings is not None else {"usd": None, "models": []})
            elif path == "/api/update":
                # About > Updates: the latest release (GitHub, cached six hours; ?force=1 asks now) and whether this
                # folder can update itself
                if self._authorized():
                    self._json(200, svc.updater.check(force="force=1" in self.path))
            elif path == "/api/report":
                # the dashboard's Copy report: what an issue needs (the run config's secrets left out)
                if self._authorized():
                    body = svc.report().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            elif path == "/status":
                with svc.status_lock:
                    s = dict(svc.status)
                    answering = svc.latest_request in svc.current_requests
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if answering:
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                s.pop("started", None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    model = {"id": svc.model, "object": "model",
                             "status": {"value": "loaded" if loaded else "unloaded"},
                             "meta": {"n_ctx": svc.engine.max_context},
                             # (videos: the llama.cpp encoder's - GLM's; DeepSeek's encoder reads pictures only)
                             "architecture": {"input_modalities": ["text"] if svc.vision is None else
                                              ["text", "image", "video"] if isinstance(svc.vision, Vision) else
                                              ["text", "image"],
                                              "output_modalities": ["text"]}}
                    self._json(200, {"object": "list", "data": [model] if loaded or svc.unloaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                    n_slots = max(1, int(getattr(svc.engine, "multi", 0) or 0))
                    with svc.status_lock:
                        active = svc.active
                    slots = [{"id": i, "n_ctx": svc.engine.max_context,
                              "is_processing": busy if n_slots == 1 else i < active} for i in range(n_slots)]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            elif path == "/v1/strata/stream":
                if self._authorized():
                    self._job_get()
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            try:
                if path.startswith("/v1/responses/") and path.endswith("/cancel"):
                    self._response_cancel(path)
                    return
                if path == "/settings":
                    self._settings()
                    return
                if path == "/v1/strata/cancel":
                    self._job_cancel()
                    return
                if path == "/api/context":
                    self._context()
                    return
                if path == "/api/port":
                    self._port()
                    return
                if path == "/api/update":
                    self._update()
                    return
                if path in ("/api/models/settings", "/api/models/switch"):
                    self._models(path.rsplit("/", 1)[1])
                    return
                if path == "/api/savings/reset":
                    # it clears every model's count: the app's own page only
                    self._body()
                    if not self._own_page("the savings can be reset"):
                        return
                    if svc.savings is None:
                        self._json(400, {"error": {"type": "invalid_request_error", "message": "nothing is counted here"}})
                        return
                    svc.savings.reset()
                    self._json(200, svc.savings.report())
                    return
                if path == "/api/savings/prices":
                    # Settings > Savings: a model's API prices ({"model", "price": {input, cached, output} or null:
                    # the built-in ones}); every total is counted again with them - the app's own page only
                    body = self._body()
                    if not self._own_page("the prices can be changed"):
                        return
                    try:
                        req = json.loads(body or b"{}")
                        if svc.savings is None:
                            raise ValueError("nothing is counted here")
                        if not isinstance(req, dict):
                            raise ValueError("the request body must be a JSON object")
                        svc.savings.set_price(req.get("model"), req.get("price"))
                    except (TypeError, ValueError) as e:
                        self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                        return
                    self._json(200, svc.savings.report())
                    return
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("the request body must be a JSON object")
                if path == "/api/tokens":                    # the context meter: a prompt's size, nothing run
                    self._json(200, svc.count_tokens(req))
                    return
                if path in ("/v1/chat/completions", "/v1/responses", "/v1/messages") and (svc.reloading() or svc.updater.busy()):
                    # a context change is reloading the model, or Maya is updating: say so, and when to try again
                    why = ("Project Maya is updating and starts again in a few minutes" if svc.updater.busy() else
                           "the model is reloading with a new context size; try again in a minute")
                    body = json.dumps({"error": {"type": "overloaded_error", "message": why}}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "30")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/responses":
                    self._responses(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except BadBody as e:
                self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except (JobError, StoreBusy) as e:
                self._json(getattr(e, "status", 503), {"error": {"type": "server_error", "message": str(e)}})
            except ValueError as e:                          # (a ContextOverflow with its code: #432)
                self._json(400, error_body(e))
            except EngineDied as e:                          # before the answer started (not streamed)
                self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
            except (KeyError, TypeError, AttributeError, IndexError) as e:
                # a field of the wrong shape (Strata: a 400 with the traceback in the log, not a dropped connection)
                traceback.print_exc()
                if not self.answer_started:
                    self._json(400, {"error": {"type": "invalid_request_error",
                                               "message": f"malformed request ({type(e).__name__}: {e})"}})

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
                self._json(404, {"error": {"message": "model not found"}})
                return
            if hasattr(svc.engine, "alive") and not svc.engine.alive() and not svc.unloaded:
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.engine.max_context, "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "multi", 0) or 0)), "model_alias": svc.model, "chat_template": svc.template.source,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": False,
                     "is_sleeping": False}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server grants only to the config's cors_origins and only on /v1/*) and no foreign Origin: a web page
            elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
                self._json(403, {"error": {"message": f"{what} only from Maya's own page"}})
                return False
            return True

        def _context(self):
            # It reloads the model for every client, so only the app's own page may ask for it
            body = self._body()
            if not self._own_page("the context size can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                n = int(req.get("max_context")) if isinstance(req, dict) else 0
                reload = svc.set_context(n)
            except (TypeError, ValueError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202, {"reload": reload})

        def _port(self):
            # It moves the server for every client, so only the app's own page may ask for it
            body = self._body()
            if not self._own_page("the port can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                n = int(req.get("port")) if isinstance(req, dict) else 0
                out = svc.set_port(n)
            except (TypeError, ValueError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202, out)

        def _models(self, what: str):
            # They change which model runs, or how, for every client: only the app's own page
            body = self._body()
            if not self._own_page("models can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("the request body must be a JSON object")
                if what == "switch":
                    out = svc.switch_model(req.get("file"))
                else:
                    out = svc.model_settings(req.get("file"), req.get("settings") or {}, bool(req.get("dry_run")))
            except (TypeError, ValueError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(200 if what == "settings" and req.get("dry_run") else 202, out)

        def _update(self):
            # It replaces the server's files and restarts it for every client: only the app's own page
            self._body()
            if not self._own_page("an update can be started"):
                return
            try:
                state = svc.updater.start(svc)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202, {"update": state})

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                shared = svc.set_shared(req.get("defaults") if isinstance(req, dict) else None)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[maya] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[maya] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _responses(self, req):
            owner = self._response_owner()
            try:
                previous = req.get("previous_response_id")
                if previous is not None:
                    if not isinstance(previous, str) or not previous:
                        raise ValueError("previous_response_id must be a nonempty string")
                    stored = self._response_lookup(owner, previous)
                    if stored["response"]["status"] not in ("completed", "incomplete"):
                        raise JobError(409, "previous response is not a completed or incomplete generation")
                    added = req.get("input", [])
                    if isinstance(added, str):
                        added = [{"role": "user", "content": added}]
                    if not isinstance(added, list):
                        raise ValueError("input must be a string or array of input items")
                    req = {**req, "input": stored["input"] + stored["response"]["output"] + added}
            except KeyError:
                return self._json(404, {"error": {"type": "invalid_request_error",
                                                 "message": "previous_response_id: stored response not found"}})
            sampling = normalize_request(req)
            validation = sampling.pop("_response_validation")
            history = input_items(req["input"]) if req.get("store") else None
            sampling = svc.with_shared(sampling, "openai")
            sampling["_api"] = "openai"
            sampling["_responses"] = True
            svc.check_request(sampling)
            if req.get("background"):
                initial = new_response(svc.model, req, status="queued")
                work = lambda cancel, rid: self._response_work(req, sampling, history, validation, owner, cancel, rid, False)
                svc.response_jobs.store = svc.responses
                out = svc.response_jobs.submit(owner, initial, history, work)
                if req.get("stream"):
                    return self._background_stream(owner, out["id"])
                return self._json(200, out)
            cancel = threading.Event()
            watch_client(self, cancel)
            events = self._response_work(req, sampling, history, validation, owner, cancel)
            try:
                if not req.get("stream"):
                    out = response_collect(events)
                    return self._json(500 if out["status"] == "failed" else 200, out)
                first = next(events)
                self._sse()
                for item in itertools.chain([first], events):
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, data = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()
                if not self.answer_started:
                    self._json(500, {"error": {"type": "server_error", "message": "could not prepare the response"}})
            finally:
                events.close()

        def _response_work(self, req, sampling, history, validation, owner, cancel, response_id=None, persist=True):
            sampling = dict(sampling)
            # Shared thinking defaults must be resolved by the same normalizer as request values.
            messages, tools, kw = openai_to_messages(sampling)
            max_new = int(sampling.get("max_completion_tokens") or sampling.get("max_tokens") or 0)
            events = None
            svc.embeddings.path = None
            try:
                if cancel.is_set():
                    raise ValueError("generation was cancelled before preparation")
                ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, cancel=cancel)
                if kw.get("_prefill"):
                    sampling["_prefill"] = kw["_prefill"]
                effective = {**svc.sampling_defaults, **svc.shared, **sampling}
                config = {**req, **{k: effective[k] for k in ("temperature", "top_p") if k in effective}}
                run = svc.run(ids, thinking, tools, max_new, sampling, cancel)
                save = (lambda out: svc.responses.put(owner, out, history)) if req.get("store") and persist else None
                events = response_events(svc.model, config, run, len(ids), persist=save, response_id=response_id,
                                         validation=validation, cancel=cancel)
                yield from events
            finally:
                if events is not None:
                    events.close()
                # A disconnect before the generator starts, or failed preparation, cannot run Service.run's cleanup.
                emb = getattr(svc.embeddings, "path", None)
                if emb:
                    Path(emb).unlink(missing_ok=True)

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            req["_api"] = "web" if req.get("strata_resume") is True else "openai"   # the Monitor's request table
            svc.check_request(req)                             # a bad sampling field or logit_bias: a 400, now
            messages, tools, kw = openai_to_messages(req, deepseek=getattr(svc.template, "deepseek", False))
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            use_mcp = req.get("strata_mcp") is True and svc.mcp is not None      # the web app's opt-in (serve/mcp.py)
            own = {t.get("name") for t in tools or []}
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                extra = svc.mcp.template_tools(exclude=own)       # the request's own tools win a name clash
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new)
            if kw.get("_prefill"):                             # a required tool call starts the answer
                req = {**req, "_prefill": kw["_prefill"]}
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            if not (req.get("stream") and req.get("strata_resume") is True):   # (the web app's runs on, by design)
                watch_client(self, cancel)
            run = run_with_mcp(svc, svc.mcp, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run)
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            if req.get("strata_resume") is True:
                # the web app: the answer runs on if the connection drops; the page reconnects to it by id
                first = next(chunks)
                job = StreamJob(first["id"], itertools.chain(
                    [first], carry_embeddings(svc.embeddings, getattr(svc.embeddings, "path", None), chunks)), cancel)
                jobs_put(job)
                self._sse()
                self._stream_job(job, 0)
                return
            self._sse()
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                cancel.set()                                 # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _stream_job(self, job: StreamJob, n: int):
            """A StreamJob's chunks from index n as SSE until it completes; a reader that goes away leaves it running."""
            job.reader(True)
            try:
                while True:
                    items, done = job.wait(n, 10.0)
                    if not items and not done:
                        self.wfile.write(b": keep-alive\n\n")
                    for c in items:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    n += len(items)
                    self.wfile.flush()
                    if done and n >= len(job.items):
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
            except OSError:
                return
            finally:
                job.reader(False)

        def _job_get(self):
            q = parse_qs(urlsplit(self.path).query)
            job = JOBS.get(q.get("id", [""])[0])
            if job is None:
                self._json(404, {"error": {"message": "no such answer (finished more than 15 minutes ago?)"}})
                return
            try:
                start = max(0, int(q.get("from", ["0"])[0]))
            except ValueError:
                start = 0
            self._sse()
            self._stream_job(job, start)

        def _job_cancel(self):
            body = self._body()
            if not self._own_page("answers can be stopped"):
                return
            try:
                jid = (json.loads(body or b"{}") or {}).get("id", "")
            except ValueError:
                jid = ""
            job = JOBS.get(jid)
            if job is not None:
                job.cancel.set()
            self._json(200, {"cancelled": job is not None})

        def _anthropic(self, req):
            req = svc.with_shared(req, "anthropic")
            req["_api"] = "anthropic"
            svc.check_request(req)                             # a bad sampling field: a 400, now
            messages, tools, kw = anthropic_to_messages(req, deepseek=getattr(svc.template, "deepseek", False))
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new)
            if kw.get("_prefill"):                             # a required tool call starts the answer
                req = {**req, "_prefill": kw["_prefill"]}
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            watch_client(self, cancel)
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel)
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: an agent or a load of clients opening 30-40 connections at once got
    # "connection reset by peer" on the first ones (Strata 0.1.41); the requests wait in the server's queue instead
    # (STRATA_HTTP_BACKLOG overrides it)
    request_queue_size = _backlog()

    def server_close(self):
        service = getattr(self, "response_service", None)
        try:
            if service is not None:
                service.response_jobs.close()
        finally:
            super().server_close()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)


SERVER_ENV = ("STRATA_ENGINE_STALL_S", "STRATA_REQUEST_QUIET_S", "STRATA_DRAIN_S", "STRATA_HTTP_BACKLOG",
              "STRATA_MAX_BODY_MIB", "STRATA_FAIR_SLICE_S", "STRATA_DS4_KEEP_THINKING",
              "STRATA_SERVER_LOG", "STRATA_GLM_THINK_RESERVE", "STRATA_LOOP_GUARD") + RESPONSE_ENV_KEYS


def apply_server_env(cfg: dict) -> None:
    """The config's "env" entries this server reads itself (`--env` puts every setting there; the rest go to the
    engine): in effect before it starts."""
    global ENGINE_STALL_S, REQUEST_QUIET_S, DRAIN_S
    for k in SERVER_ENV:
        v = (cfg.get("env") or {}).get(k)
        if v is not None:
            os.environ[k] = str(v)
    ENGINE_STALL_S = _stall_seconds()
    REQUEST_QUIET_S = _quiet_seconds()
    DRAIN_S = _drain_seconds()
    Server.request_queue_size = _backlog()


def warn_tight_ram(arena_mib) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux ends the engine mid-answer when memory runs out (issue #27) and Windows
    pages to disk; say so at start instead of after a lost answer."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[maya] WARNING: RAM is tight - the model's experts take {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + ("Linux may stop the engine in the middle of an answer. " if os.name != "nt" else
                 "Windows will slow down (paging to disk). ")
              + "Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", flush=True)


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    svc.listen = (host, httpd.server_address[1])         # (the dashboard's Port setting starts from it)
    httpd.response_service = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


SHARED_KEYS = ("reasoning_effort", "temperature", "top_p", "top_k", "seed", "max_tokens", "experimental_speed_projection")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":   # (kept by its own name: medium / xhigh are MiniMax's, GLM's High / Max else)
            if effort_level(value) is None:
                raise ValueError(f"reasoning_effort: {EFFORT_WORDS}")
            value = effort_name(value)
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "min_p":                            # the Chat's Min-p (the Creative preset's 0.05)
            if not number or not 0 <= value < 1:
                raise ValueError("min_p: 0 <= min_p < 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise ValueError("top_k: an integer 1..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p", "min_p") else value
    return out


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty, frequency_penalty, penalty_last_n, seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd config should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[maya] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[maya] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[maya] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[maya] config sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[maya] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[maya] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[maya] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < -1:
                raise SystemExit(f"[maya] config sampling.penalty_last_n={value!r}: expected an integer >= -1 (0 = no "
                                 "penalties, -1 = the whole context)")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[maya] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[maya] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[maya] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def request_stop(stop: threading.Event, why: str) -> None:
    """The server's stop, once: main() wraps up the request in progress and ends the engine."""
    if not stop.is_set():
        stop.set()
        print(f"[maya] stopping ({why}): the engine ends", flush=True)


def stop_signals(stop: threading.Event) -> None:
    """SIGTERM, SIGINT (Ctrl+C), SIGHUP (the terminal closed) and Windows' SIGBREAK (Ctrl+Break) stop the server
    (request_stop).  A later one is ignored while it stops - but a second Ctrl+C raises KeyboardInterrupt: at once.  A
    signal that is ignored (nohup) stays ignored."""
    import signal

    def on(signum, frame):
        if stop.is_set():
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            return
        request_stop(stop, signal.Signals(signum).name)
    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        try:
            if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
                signal.signal(sig, on)
        except (OSError, RuntimeError, ValueError):
            pass


def watch_parent(stop, stdin=None, getppid=None, poll: float = 1.0) -> None:
    """--parent-pipe: stdin is a pipe from the program that started the server (maya.py), which writes nothing and
    holds it open while it runs.  Its end - however that program ended: a crash, SIGKILL, Task Manager - closes it,
    and the EOF stops the server (`stop(why)`) as Ctrl+C does.  On POSIX a new parent process does too (the parent
    ended and init or a subreaper took the server).  Both watch from threads of their own."""
    src = stdin if stdin is not None else getattr(sys.stdin, "buffer", None)

    def closed_polling():
        # Windows: the CRT keeps a descriptor's lock for a whole blocking read(), and the first `import numpy` (a
        # picture, in a request's thread) waits on stdin's: the server froze (#444).  Polling the pipe holds no lock.
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = wintypes.HANDLE(msvcrt.get_osfhandle(src.fileno()))
        avail = wintypes.DWORD()
        while k32.PeekNamedPipe(h, None, 0, None, ctypes.byref(avail), None):
            if avail.value:
                src.read1(avail.value)
            time.sleep(0.25)
        return ctypes.get_last_error() == 109      # ERROR_BROKEN_PIPE: the parent's end is closed

    def eof():
        try:
            if not (os.name == "nt" and stdin is None and closed_polling()):
                while src.read(1 << 12):
                    pass
        except (OSError, ValueError, AttributeError):
            pass
        stop("the program that started it ended")
    if src is not None:
        threading.Thread(target=eof, name="parent-pipe", daemon=True).start()
    if os.name == "nt" and getppid is None:
        return
    getppid = getppid or os.getppid
    first = getppid()

    def orphan():
        while getppid() == first:
            time.sleep(poll)
        stop("its parent process ended")
    threading.Thread(target=orphan, name="parent-pid", daemon=True).start()


class ServerLogTee:
    """stdout as it was, and the same text appended to the run's log (where the engine already writes its own lines).
    This server only printed to the terminal, so a run whose window was closed - or that was started by a script - left
    no record of what it did: a request that waited minutes for its turn, an engine restarted mid-session, a queue that
    stopped moving.  That record is what a report of a stuck server needs."""

    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            try:
                self.handle.write(text)
            except (OSError, ValueError):                # the log went away: the terminal still gets the line
                pass
        return self.stream.write(text)

    def flush(self):
        try:
            self.stream.flush()
        except (OSError, ValueError):
            pass
        with self.lock:
            try:
                self.handle.flush()
            except (OSError, ValueError):
                pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_server_log(path: str) -> None:
    """This server's own lines (`[maya] ...`) go to `path` as well as this terminal (STRATA_SERVER_LOG=0: terminal
    only).  The engine's stderr is already appended there, so the file tells both sides of a request."""
    path = os.path.abspath(path)
    if not path or isinstance(sys.stdout, ServerLogTee):
        return
    try:
        handle = open(path, "a", encoding="utf-8", buffering=1)
    except OSError as e:
        print(f"[maya] this server's lines cannot also go to {path} ({e}): they stay on this terminal", flush=True)
        return
    print(f"[maya] this server's lines go to {path} too", flush=True)
    sys.stdout = ServerLogTee(sys.stdout, handle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices "
                         "on your network (set an API key); also \"host\" in the config")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help='unload the model after this many seconds without requests, so other programs can use the GPU '
                         'memory and the RAM; the next request loads it again (also "idle_unload_s" in the config; '
                         '0: never)')
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(default: reject with 400, like llama.cpp; also \"fit_max_tokens\": true in the config)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); several separated by "
                         "commas; also $STRATA_API_KEY")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--parent-pipe", action="store_true",
                    help="stdin is a pipe from the program that started this server (maya.py), open while it runs: "
                         "when it closes - that program ended, however it ended - the server stops as on Ctrl+C "
                         "(on Linux also when its parent process ends)")
    a = ap.parse_args()
    maya_names.alias_env(os.environ)   # (MAYA_* settings under the names the code and the engine read)
    cfg = maya_names.env_in(json.loads(Path(a.config).read_text(encoding="utf-8-sig"))) if a.config else {}   # (a BOM: Notepad)
    apply_server_env(cfg)
    # the server's own lines go to the run's log as well (the engine writes its stderr there): a run whose terminal
    # was closed leaves the record a stuck queue needs (STRATA_SERVER_LOG=0 keeps them on the terminal only)
    _srv_log = os.environ.get("STRATA_SERVER_LOG")
    if _srv_log is None:
        _srv_log = str(cfg.get("log") or "")
    if _srv_log.strip().lower() not in ("", "0", "off", "no", "false", "none"):
        tee_server_log(_srv_log)
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        ap.error(f"port {a.port} is already in use - is Maya (or another server) already running? "
                 f"Close it, or start this one with a different --port")
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_dir(tpath)              # with the pack's own pre-tokenizer (GLM: glm4, #27)
    hub = hub_from_config(cfg, a.mcp_config)            # before the minutes of loading: a bad entry stops here
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        vision = None
        env = child_env(cfg)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[maya] sampling defaults from the config: {pretty}", flush=True)
        vcfg = cfg.get("vision") or None
        vlog = (open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None) if vcfg else None
        venv = dict(env)
        if vcfg and vcfg.get("gpu") and gpu_list(cfg) and cfg.get("backend") != "hip":
            venv["CUDA_VISIBLE_DEVICES"] = str(gpu_list(cfg)[0])   # the encoder on the first GPU only
        # the GLM engine lends the encoder the tail of its expert cache while it encodes, sized by what the encoder
        # measured on this GPU ("resident": true keeps the encoder running beside the model, holding its VRAM)
        on_demand = bool(vcfg) and vcfg.get("gpu") and not vcfg.get("resident")
        engine_env = env
        if on_demand:
            fp = vision_footprint(vcfg, venv, Path(a.config).resolve().with_name("vision-memory.json"),
                                  (gpu_list(cfg) or [0])[0])
            if fp is None:
                print("[maya] images: the vision encoder could not be measured on the GPU - it runs on the CPU",
                      flush=True)
                vcfg, on_demand = dict(vcfg, gpu=False), False
            else:
                vcfg = dict(vcfg, max_tokens=fp[0])
                engine_env = dict(env, STRATA_GLM_VISION_LEND_MB=str((fp[1] >> 20) + 1))
                print(f"[maya] images: the vision encoder needs {fp[1] / 2**30:.2f} GB on this GPU at up to {fp[0]} "
                      "image tokens", flush=True)
        if vcfg and not on_demand:
            print("loading the vision encoder ...", flush=True)
            vision = Vision(vcfg, log=vlog, env=venv)
        print("loading the model (the first start takes a minute or two) ...", flush=True)
        if len(gpu_list(cfg)) > 1:
            print(f"[maya] layer split across GPUs {gpu_list(cfg)} ({cfg.get('layer_split') or 'auto'})", flush=True)
        t0 = time.time()
        engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=engine_env)
        engine.load_s = round(time.time() - t0, 1)      # (the dashboard's estimate of a reload: this start's time)
        warn_tight_ram(engine.info.get("arena_mib"))
        if vision is None and not vcfg and engine.info.get("vision") == "ds4":
            # DeepSeek-V4.1: the engine's own vision tower (the checkpoint's); the server resizes the pictures
            vision = ds4_vision.Vision(ds4_vision.Geometry.from_info(engine.info),
                                       Path(tempfile.mkdtemp(prefix="maya-ds4-pictures-")), Vision.load)
            room = engine.info.get("vision_room")
            where = (f"{engine.info.get('vision_mib')} MiB kept" if room in (None, "resident") else
                     f"none between pictures, ~{engine.info.get('vision_room_mib')} MiB while one is read, lent by "
                     + {"chunk": "the prompt buffers", "tail": "expert slots", "free": "free memory"}.get(room, room))
            print(f"[maya] images: the model's vision tower runs in the engine (GPU {engine.info.get('vision_gpu')}: "
                  f"{where}), up to {vision.g.max_tokens} tokens a picture", flush=True)
        elif vision is None and not vcfg and engine.info.get("vision") == "mm3":
            # MiniMax-M3.1-Flash: the engine's own vision tower; the server resizes the pictures
            vision = mm3_vision.Vision(mm3_vision.Geometry.from_info(engine.info),
                                       Path(tempfile.mkdtemp(prefix="maya-mm3-pictures-")), Vision.load)
            print(f"[maya] images: the model's vision tower runs in the engine, up to {vision.g.max_pixels} pixels "
                  f"({vision.g.max_tokens} tokens) a picture", flush=True)
        elif vision is None and not vcfg and engine.info.get("vision") == "off":
            # (DeepSeek-V4.1: its tower did not load - the engine's NOTE line above says why; no pictures offered)
            print("[maya] images: OFF - pictures are refused until the vision tower loads (see the line above)",
                  flush=True)
        if on_demand:
            lent = int(engine.info.get("vision_lend") or 0)
            if lent > 0:
                vision = Vision(vcfg, log=vlog, env=venv, start=False)
                vision.lend = lambda: engine.command("VLEND", "VLENT")
                vision.reclaim = lambda: engine.command("VRECLAIM", "VRECLAIMED")
                print(f"[maya] images: the vision encoder starts when a picture arrives, in {lent / 2**30:.2f} GB "
                      "the model lends it", flush=True)
            else:   # the model's GPU memory is too small to lend that much (or an engine without lending)
                print("[maya] images: the model cannot lend the vision encoder its GPU memory here - the encoder "
                      "runs on the CPU", flush=True)
                vision = Vision(dict(vcfg, gpu=False), log=vlog, env=venv)
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."]), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's.  A GLM-5.3-Flash pack's
    # copy is the one its GGUF carried (August's): the template zai-org publishes now renders the same prompts and
    # matches tool results to their calls without nested loops (zai-org/GLM-5.3-Flash PR #63: 100 parallel calls
    # 137 -> 6 ms a render), so the server uses the copy shipped here (serve/glm5_chat_template.jinja) for a GLM-5
    # pack.  MAYA_PACK_TEMPLATE=1 keeps the pack's own.
    tpl = tpath / "chat_template.jinja"
    tpl = tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"
    glm5 = ROOT / "serve/glm5_chat_template.jinja"
    if glm5.exists() and os.environ.get("MAYA_PACK_TEMPLATE") != "1" and tpl != glm5:
        head = tpl.read_text(encoding="utf-8", errors="replace")
        if head.startswith("[gMASK]<sop>") and "reasoning_effort in ['low', 'high']" in head:
            tpl = glm5
    svc = Service(engine, tok, ChatTemplate(tpl),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or cfg.get("fit_max_tokens") is True)
    if getattr(engine, "multi", 0) > 1:                 # n conversations at once: n places in the FIFO
        svc.fifo.set_capacity(engine.multi)
        print(f"[maya] {engine.multi} conversations at once (--parallel)", flush=True)
    if cfg.get("reasoning_effort") or (svc.template.deepseek and cfg.get("reasoning_effort") is not None):
        try:
            (deepseek_effort_kwargs if svc.template.deepseek else effort_kwargs)(cfg["reasoning_effort"])
        except ValueError as e:
            raise SystemExit(f"[maya] config reasoning_effort: {e}")
        svc.default_effort = cfg["reasoning_effort"]
        # (#363: DeepSeek's 1..100 budget is a level of its own - it was shown as "none")
        level = (effort_name if svc.template.minimax else effort_level)(svc.default_effort) or \
            (f"{svc.default_effort} of 100" if type(svc.default_effort) is int
                                                     else "none")
        print(f"[maya] thinking level when a request names none: {level}"
              f"{'' if level in (svc.default_effort, f'{svc.default_effort} of 100') else f' (the config says {svc.default_effort})'}",
              flush=True)
    svc.think_budget = int(cfg.get("thinking_budget", 32768) or 0) if svc.think_end_id is not None else 0
    svc.idle_unload_s = max(0.0, a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0))
    if svc.think_budget:
        print(f"[maya] thinking budget: {svc.think_budget} tokens (then the answer)", flush=True)
    svc.loop_guard = loop_guard_setting(cfg)            # (MAYA_LOOP_GUARD=0 or "loop_guard": false: off)
    if not svc.loop_guard:
        print("[maya] loop guard: off (a request's \"loop_guard\": true turns it on for that request)", flush=True)
    svc.api_key = a.api_key or cfg.get("api_key", "")
    try:
        svc.cors_origins = parse_cors_origins(cfg.get("cors_origins"))
    except ValueError as e:
        raise SystemExit(f"config: {e}")
    if svc.cors_origins:
        print("[maya] browser pages allowed to call /v1/* (CORS): " + ", ".join(svc.cors_origins), flush=True)
    svc.gpu_index = (gpu_list(cfg) or [0])[0]           # the Monitor reads the card the engine runs on (issue #51)
    svc.gpu_indices = gpu_list(cfg)                     # ... or every card of a layer split (issue #112)
    svc.backend = cfg.get("backend")                    # the Monitor's GPU reader: sysfs on AMD/HIP, NVML otherwise
    svc.model_label = maya_quants.config_label(cfg)     # (configs of every release: by their installer record)
    if a.config:
        svc.config_path = str(Path(a.config).resolve())  # a context change from the dashboard is saved there
        p = Path(svc.config_path)                        # the tokens it serves, and what its API would have charged
        svc.savings = savings.Ledger(p.parent / savings.FILE, p.name, svc.model, models.quant_label(cfg, p))
    args = [str(x) for x in cfg.get("args") or []]
    if "--glm-pack" in args[:-1]:                       # the model's folder: the pack's parent holds its GGUF files
        svc.model_dir = Path(args[args.index("--glm-pack") + 1]).resolve().parent
        threading.Thread(target=lambda: setattr(svc, "trained_context", trained_context(svc.model_dir)),
                         daemon=True).start()
    elif "--pack" in args[:-1]:                         # DeepSeek's pack (its ds4.txt), MiniMax's (mm3.txt); its folder
        svc.trained_context = trained_context_ds4(args[args.index("--pack") + 1])
        svc.model_dir = Path(args[args.index("--pack") + 1]).resolve().parent
        # (DeepSeek's engine drafts unless --dspark off / none - "none" does not even load the blocks - when the pack
        #  has them; MiniMax's says so itself, INFO dspark=)
        if models.family(cfg) == "deepseek":
            svc.dspark = models.flag(args, "--dspark", "on") not in ("off", "none") and models.ds4_has_drafter(cfg)
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            svc.shared = clean_shared_defaults(json.loads(Path(svc.shared_path).read_text(encoding="utf-8")))
            if svc.shared:
                print("[maya] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
        except (OSError, ValueError):
            svc.shared = {}
    if hub is not None:
        import atexit
        svc.mcp = hub
        print(f"[maya] starting {len(hub.servers)} MCP server{'s' * (len(hub.servers) != 1)} for the web app's "
              f"chat: {', '.join(hub.servers)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    httpd = serve(svc, host=a.host, port=a.port)
    svc.start_idle_unload()
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{a.port}/v1  (OpenAI: /v1/chat/completions, /v1/responses, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{a.port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{a.port}/   (API: http://{ip}:{a.port}/v1)", flush=True)
        if not ips:
            print("       from other devices: http://<this PC's IP address>:" + str(a.port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" "
                  "to the config (clients send it as their API key; the web page asks for it)", flush=True)
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Maya " + str(a.port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(a.port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    # SIGTERM (llama-swap's cmdStop, systemd), SIGHUP (the terminal closed), Ctrl+C, Windows' Ctrl+Break and - with
    # --parent-pipe - the end of maya.py (watch_parent) end the server alike: the engine gets QUIT and time to keep its
    # conversation.  A second signal while that runs (systemd signals the whole group, then the supervisor again) is
    # ignored instead of cutting it short - except a second Ctrl+C, which ends the server at once (the engine then ends
    # when its input closes).
    stop = threading.Event()
    stop_signals(stop)
    if a.parent_pipe:
        watch_parent(lambda why: request_stop(stop, why))
    try:
        while not stop.wait(0.5):                      # (a timed wait: signal handlers run on Windows too)
            pass
    except KeyboardInterrupt:
        request_stop(stop, "Ctrl+C")
    try:
        httpd.shutdown()
        # a request in progress is wrapped up, not dropped: its thinking is closed and it answers, for up to
        # STRATA_ENGINE_WRAP_S seconds (40), so the client still gets an answer and the turn ends; past that it is cut.
        # Requests still queued are refused. Then the engine ends (and keeps its conversation).
        svc.stopping = True
        wrap_s = float(os.environ.get("STRATA_ENGINE_WRAP_S", "40") or 0)
        if wrap_s > 0 and hasattr(engine, "wrap") and svc.status.get("busy"):
            print(f"[maya] stopping: the request in progress closes its thinking and answers (up to {wrap_s:.0f} s)",
                  flush=True)
            engine.wrap()
            if svc.fifo.acquire(timeout=wrap_s):
                svc.fifo.release()
                print("[maya] stopping: the request in progress finished", flush=True)
            else:
                print(f"[maya] stopping: the answer was not done after {wrap_s:.0f} s - cut there", flush=True)
                engine.stop()
                # the cut request ends at its next step (DONE, or an error if the engine died): only then QUIT, so its
                # last lines are read by the request and not by close()
                with svc.fifo:
                    pass
        if hasattr(engine, "close"):
            engine.close()
        if vision:
            vision.close()
        if hub is not None:
            hub.close()
    except KeyboardInterrupt:                          # a second Ctrl+C: at once
        print("[maya] stopped at once: the engine ends as its input closes", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
