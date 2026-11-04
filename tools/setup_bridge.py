"""The setup's side of its screen (tools/setup_tui.py): setup.UI while the screen runs.  setup.py's output functions
hand a Bridge what they would print, ask and run; it queues all of it for the screen in order, runs the commands with
their output on the screen as it comes, and keeps what the terminal gets when the screen closes (summary) and
maya-setup.log.  Nothing here draws: the screen reads `events`, and answers questions through their Future.
LoadWatch reads how far Maya's start is from its engine log, for the dashboard's loading box (setup_dash.py).
"""
from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path

TAIL = 40                                               # the output lines a summary keeps of a step that stopped
# the GLM engine's start, in its log: each GPU's expert tiers warmed up in turn - "glm fast: CUDA2 warming the expert
# tiers 40% (5.4 of 13.4 GB, 11 s)", then "glm fast: CUDA2 tiers warm: 1152 experts (13.4 GB) in 29 s"
WARMING = re.compile(r"glm fast: (\S+?)(\d+) warming the expert tiers \d+% \(([\d.]+) of ([\d.]+) GB")
WARMED = re.compile(r"glm fast: (\S+?)(\d+) tiers warm: \d+ experts \(([\d.]+) GB\)")


def curl_meter(line: str):
    """(share done, what to show) from a line of curl's progress meter, or None.  Its columns: % Total, Total,
    % Received, Received, % Xferd, Xferd, Dload, Upload, Time Total, Spent, Left, Current - e.g.
    ' 12 31.2G   12 3900M    0     0  85.1M      0  0:06:15  0:00:45  0:05:30 88.2M'."""
    f = line.split()
    if len(f) != 12 or not (f[0].isdigit() and f[2].isdigit()):
        return None
    return int(f[0]) / 100, f"{f[3]}B of {f[1]}B, {f[11]}B/s, {f[10]} left"


def describe(cmd: list) -> str:
    """What a command does, for the card."""
    name = Path(cmd[0]).stem.lower()
    if name == "curl" and "-o" in cmd[:-1]:
        return f"Downloading {Path(cmd[cmd.index('-o') + 1]).name}"
    if name == "cmake":
        return "Compiling" if "--build" in cmd else "Configuring the build"
    if name.startswith("python") and cmd[1:3] == ["-m", "pip"]:
        return "Installing Python packages"
    return f"Running {Path(cmd[1]).name if name.startswith('python') and len(cmd) > 1 else Path(cmd[0]).name}"


class LoadWatch:
    """How far the model's start is, from the engine log (read from where it stood when the start began): the warm-up
    of each GPU's expert tiers - the bulk of a start, one GPU after the other - in GB.  `gpus`: the start's, in the
    order the engine numbers its parts (its CUDA0 is the first); `vram`: each one's GB of VRAM, by its number."""

    def __init__(self, path, gpus, vram=None):
        self.path, self.gpus, self.vram = path, list(gpus) or [0], dict(vram or {})
        try:
            self.pos = os.path.getsize(path) if path else 0
        except OSError:
            self.pos = 0
        self.reset()

    def reset(self) -> None:
        """A start (or a reload) begins: nothing warm yet."""
        self.parts, self.t0 = {}, time.monotonic()      # the part's number -> [GB warm, GB in all]

    def read(self) -> None:
        """The log's new lines (on the screen's thread, once a second: a few KB)."""
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                chunk = f.read()
        except (OSError, TypeError):
            return
        cut = chunk.rfind(b"\n") + 1
        self.pos += cut
        for line in chunk[:cut].decode("utf-8", "replace").splitlines():
            if m := WARMING.search(line):
                self.parts[int(m.group(2))] = [float(m.group(3)), float(m.group(4))]
            elif m := WARMED.search(line):
                self.parts[int(m.group(2))] = [float(m.group(3))] * 2

    def total(self) -> tuple:
        """(the GB the start warms in all, whether it is known): each part's as the engine said; one not started yet
        as the parts so far per GB of their VRAM, times its own (a 16 GB card after the 3090: 2/3 of its part, not
        all of it) - without the VRAMs, as their mean."""
        known = {k: whole for k, (_, whole) in self.parts.items()}
        waiting = [k for k in range(len(self.gpus)) if k not in known]
        vram = [self.vram.get(self.gpus[k]) if k < len(self.gpus) else None for k in [*known, *waiting]]
        if not waiting or not known:
            return sum(known.values()), not waiting
        if all(vram):
            per = sum(known.values()) / sum(vram[:len(known)])
            return sum(known.values()) + per * sum(vram[len(known):]), False
        return sum(known.values()) * (1 + len(waiting) / len(known)), False

    def fraction(self):
        """The share of the start done (None: no engine log to tell)."""
        if not self.path:
            return None
        if not self.parts:
            return 0.0
        return min(1.0, sum(done for done, _ in self.parts.values()) / max(self.total()[0], 1e-9))


class Bridge:
    """setup.UI while the screen runs: what the setup's thread prints, asks and runs, queued for the screen in order.
    Each method returns something false when it does not take it - the screen has closed and the caller is not the
    setup's thread - and the caller prints it as without the screen.  After a stop the setup's thread ends at its
    next line (KeyboardInterrupt)."""

    def __init__(self, log_path: Path | None):
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.entries: list = []                         # (step, kind, text): what a summary is made of
        self.step_no = 0
        self.failure = None                             # fail()'s (message, hint)
        self.log = open(log_path or os.devnull, "w", encoding="utf-8", buffering=1)   # (None: a start, no setup log)
        self.thread: threading.Thread | None = None     # the setup's
        self.proc: subprocess.Popen | None = None       # the command running now: a stop ends it
        self.interrupted: subprocess.Popen | None = None   # the one a stop interrupted
        self.stopped = threading.Event()
        self.closed = False

    def _here(self) -> bool:
        if threading.current_thread() is self.thread:
            if self.stopped.is_set():
                raise KeyboardInterrupt
            return True
        return not self.closed

    def _note(self, kind: str, text: str) -> None:
        self.entries.append((self.step_no, kind, text))
        self.log.write(("\n" if kind == "step" else "") + text + "\n")

    def _wait(self, *event):
        fut = Future()
        self.events.put((*event, fut))
        return fut.result()                             # the screen's answer; a stop raises KeyboardInterrupt here

    def _emit(self, event: tuple, text: str | None = None) -> bool:
        """`event` for the screen, `text` (the line printed without it) for the log and the summary."""
        if not self._here():
            return False
        if text is not None:
            self._note(event[0], text)
        self.events.put(event)
        return True

    def say(self, msg=""):
        return self._emit(("say", str(msg)), str(msg))

    def step(self, n, title):
        if self._here():
            self.step_no = n
        return self._emit(("step", n, title), f"=== Step {n}: {title} ===")

    def ok(self, msg):
        return self._emit(("ok", str(msg)), f"  [ok] {msg}")

    def warn(self, msg):
        return self._emit(("warn", str(msg)), f"  [!]  {msg}")

    def progress(self, msg=None, frac=None):
        return self._emit(("progress", msg, frac))

    def fail(self, msg, hint=None):
        """Shown until it is read; then the setup ends (SystemExit), and run() prints it once the screen is gone."""
        if not self._here():
            return False
        self.failure = (msg, hint)
        self.log.write(f"\n  [X]  {msg}\n" + (f"       {hint}\n" if hint else ""))
        self._wait("fail", str(msg), hint)
        raise SystemExit(1)

    def ask(self, question, choices, default):
        if not self._here():
            return None
        question = question.strip()
        answer = self._wait("ask", question, list(choices), default)
        self._note("answer", f"  {question} {answer}")
        return answer

    def choose(self, question, intro, options, default, outro=()):
        if not self._here():
            return None
        pick = self._wait("choose", question, list(intro), list(options), default, list(outro))
        self._note("answer", f"  {question} {pick + 1}) {options[pick][0]}")
        return pick

    def run(self, cmd, cwd=None, env=None, keep=True, parent_pipe=False):
        """The command with its output on the screen as it comes (stdin closed: nothing can wait for a key).  keep:
        its output also into the log and the summary (not the server's: it runs for days - "live" on the screen).
        parent_pipe: Maya's server - its stdin a pipe from here instead (--parent-pipe: it stops when this process
        ends, however it ends), and on Windows in a job object that ends it with this process (setup.end_with_me)."""
        if not self._here():
            return None
        self.events.put(("run", cmd))
        own = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        if parent_pipe:
            cmd = list(cmd) + ["--parent-pipe"]
        with subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE if parent_pipe else subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, **own) as p:
            if parent_pipe:
                import setup
                setup.end_with_me(p)
            self.proc = p
            rest = b""
            while chunk := p.stdout.read(1 << 16):
                rest += chunk
                # a line ends at \n (or \r\n); a lone \r rewrites it (curl's meter); a last \r may be half of a \r\n
                cut = len(rest) - rest.endswith(b"\r")
                parts = re.split(rb"(\r\n|\n|\r)", rest[:cut])
                rest = parts.pop() + rest[cut:]
                for text, end in zip(parts[::2], parts[1::2]):
                    self._output(text.decode("utf-8", "replace"), end == b"\r", keep)
            if rest.strip(b"\r\n"):
                self._output(rest.decode("utf-8", "replace").rstrip("\r\n"), False, keep)
        self.proc = None
        self.events.put(("ran", p.returncode))
        if self.stopped.is_set():
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(cmd, p.returncode)

    def _output(self, line: str, rewrite: bool, keep: bool = True) -> None:
        if rewrite:
            self.events.put(("status", line))
            return
        if keep:
            self._note("out", line)
        self.events.put(("out" if keep else "live", line))

    def serve(self, cmd, env, info: dict):
        """Maya's server, after the setup, on the same screen until it ends: its exit code.  `info`: what the screen
        shows of it (maya.py, server_command)."""
        if not self._here():
            return None
        self.events.put(("serve", info))
        return self.run(cmd, env=env, keep=False, parent_pipe=True).returncode

    def stop(self) -> None:
        """Stopped on the screen: the command running now gets Ctrl+C's SIGINT, with its children (the server closes
        the engine; finish() ends what is left), anything else this process started ends (the engine the tuning loads:
        without the screen Ctrl+C reached it too), and the setup's thread ends at its next line."""
        self.stopped.set()
        p = self.interrupted = self.proc
        if p is not None and p.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
                else:
                    os.killpg(p.pid, signal.SIGINT)
            except OSError:
                pass
        try:
            import psutil                               # (step 3 installs it; the tuning's engine starts after that)
        except ImportError:
            return
        try:
            mine = {p.pid, *(c.pid for c in psutil.Process(p.pid).children(recursive=True))} if p is not None else set()
        except psutil.Error:
            mine = set()
        for child in psutil.Process().children(recursive=True):
            try:
                if child.pid not in mine:
                    child.terminate()
            except psutil.Error:
                pass

    def finish(self, timeout: float = 20) -> None:
        """After a stop, with the screen gone: the command it interrupted gets `timeout` seconds to end, then it ends
        by force (with its children)."""
        p = self.interrupted
        if p is None or os.name == "nt":                # (taskkill /F ended it already)
            return
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass

    def summary(self, whole: bool) -> list:
        """The steps with their results and answers; `whole`: also the last step's messages and the end of its
        output (the setup ended there)."""
        out = [i for i, (n, kind, _) in enumerate(self.entries) if n == self.step_no and kind == "out"]
        keep = set(out[-TAIL:])
        lines = []
        for i, (n, kind, text) in enumerate(self.entries):
            if kind == "step":
                lines += ["", text]
            elif kind in ("ok", "warn", "answer") or whole and n == self.step_no and (kind == "say" or i in keep):
                lines.append(text)
        return lines
