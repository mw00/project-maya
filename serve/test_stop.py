"""Every way of stopping maya.py stops Maya's server and its engine (v1.1.18): the server's stop signals, --parent-pipe
(its stdin a pipe from maya.py: EOF when maya.py ends, however it ends), the orphan check, and maya.run_server - with
the real server and serve/fake_multi_engine.py as its engine (a process that ends on QUIT or when its input closes), in
real processes.  No GPU needed; psutil (setup installs it) finds the processes."""
from __future__ import annotations

import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as SV  # noqa: E402

try:
    import psutil
except ImportError:                                     # (the CI's and setup's environments have it)
    psutil = None
try:
    import regex  # noqa: F401  (the GLM tokenizer's, which --engine maya loads: else the mock engine, no process)
    FAKE_ENGINE = True
except ImportError:
    FAKE_ENGINE = False
WIN = os.name == "nt"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_config(d: Path) -> Path:
    """A config whose engine is serve/fake_multi_engine.py (one conversation) and a tokenizer of single bytes."""
    tok = d / "tokenizer"
    tok.mkdir()
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("\xa1"), ord("\xac") + 1), *range(ord("\xae"), ord("\xff") + 1)]
    cs = bs[:] + [256 + n for n in range(256 - len(bs))]
    bs += [b for b in range(256) if b not in bs]
    vocab = {chr(c): b for b, c in zip(bs, cs)}         # (byte-level BPE's 256 byte tokens, as GPT-2 maps them)
    vocab.update({"<|im_end|>": 256, "</think>": 257})
    (tok / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tok / "merges.txt").write_text("a b", encoding="utf-8")
    engine = ROOT / "serve" / "fake_multi_engine.py"
    if WIN:                                             # (Windows starts no .py by itself: a .bat around it)
        exe = d / "fake_engine.bat"
        exe.write_text(f'@"{sys.executable}" "{engine}" %*\r\n', encoding="utf-8")
    else:
        exe = engine
    cfg = d / "maya-fake.json"
    cfg.write_text(json.dumps({"exe": str(exe), "args": ["1"], "tokenizer": str(tok), "model_name": "fake",
                               "log": str(d / "engine.log")}), encoding="utf-8")
    return cfg


def server_cmd(cfg: Path, port: int) -> list:
    engine = ["--engine", "maya", "--config", str(cfg)] if FAKE_ENGINE else ["--engine", "mock"]
    return [sys.executable, str(ROOT / "serve" / "server.py"), *engine, "--port", str(port)]


class Proc:
    """A process with its output read line by line (until() waits for a line)."""

    def __init__(self, cmd, **kw):
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1, **kw)
        self.lines: queue.Queue = queue.Queue()
        self.seen: list = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.lines.put(line.rstrip())
        self.lines.put(None)

    def until(self, text: str, timeout: float = 60) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                return False
            self.seen.append(line)
            if text in line:
                return True
        return False

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait(timeout=30)
        for f in (self.p.stdin, self.p.stdout):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass

    def rest(self, timeout: float = 5) -> str:
        """The output that is left once the process ended."""
        self.until("\0never", timeout)
        return "\n".join(self.seen)


def gone(pids, timeout: float = 30) -> list:
    """The processes of `pids` still running after `timeout` seconds (zombies count as ended)."""
    def alive(pid):
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    end = time.monotonic() + timeout
    left = list(pids)
    while left and time.monotonic() < end:
        left = [p for p in left if alive(p)]
        if left:
            time.sleep(0.2)
    return left


def kill_all(pids) -> None:
    for q in pids:
        try:
            psutil.Process(q).kill()
        except psutil.Error:
            pass


def descendants(pid: int) -> list:
    try:
        return [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        return []


class WatchParent(unittest.TestCase):
    """The server's watch of its parent (--parent-pipe), in-process."""

    def test_eof_on_the_pipe_stops(self):
        got = []
        SV.watch_parent(got.append, stdin=io.BytesIO(b"x" * 10000), getppid=lambda: 1234)
        end = time.monotonic() + 5
        while not got and time.monotonic() < end:
            time.sleep(0.01)
        self.assertEqual(got[:1], ["the program that started it ended"])

    def test_a_new_parent_stops(self):
        got, ppid = [], [1234]
        quiet = SimpleNamespace(read=lambda n: time.sleep(3600))   # (a pipe that stays open: the EOF watch waits)
        SV.watch_parent(got.append, stdin=quiet, getppid=lambda: ppid[0], poll=0.01)
        time.sleep(0.1)
        self.assertEqual(got, [])
        ppid[0] = 1                                     # (the parent ended: init took the server)
        end = time.monotonic() + 5
        while not got and time.monotonic() < end:
            time.sleep(0.01)
        self.assertEqual(got, ["its parent process ended"])

    def test_stop_once(self):
        stop = threading.Event()
        SV.request_stop(stop, "a")
        SV.request_stop(stop, "b")
        self.assertTrue(stop.is_set())


@unittest.skipIf(psutil is None, "psutil is not installed")
class ServerStops(unittest.TestCase):
    """The real server with the fake engine: each way of stopping it ends it and its engine, after QUIT."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = fake_config(Path(self.tmp.name))

    def start(self, extra=(), **kw) -> Proc:
        p = Proc(server_cmd(self.cfg, free_port()) + list(extra), **kw)
        self.addCleanup(p.close)
        self.assertTrue(p.until("ready:"), "the server did not start:\n" + "\n".join(p.seen))
        self.engine = descendants(p.p.pid)
        self.assertTrue(self.engine or not FAKE_ENGINE, "no engine process")
        return p

    def ended(self, p: Proc, why: str):
        self.assertEqual(p.p.wait(timeout=60), 0)
        out = p.rest()
        self.assertIn(f"[maya] stopping ({why}): the engine ends", out)
        self.assertEqual(gone(self.engine), [], "the engine outlived the server")

    @unittest.skipUnless(FAKE_ENGINE, "the tokenizer's regex module")
    def test_old_engine_name(self):
        """--engine strata (the old name, in older run scripts) starts the configured engine, not the mock one."""
        cmd = server_cmd(self.cfg, free_port())
        cmd[cmd.index("maya")] = "strata"
        p = Proc(cmd, stdin=subprocess.DEVNULL)
        self.addCleanup(p.close)
        self.assertTrue(p.until("ready:"), "the server did not start:\n" + "\n".join(p.seen))
        self.assertTrue(descendants(p.p.pid), "no engine process: --engine strata ran the mock engine")

    def test_parent_pipe_closed(self):
        p = self.start(["--parent-pipe"], stdin=subprocess.PIPE)
        p.p.stdin.close()
        self.ended(p, "the program that started it ended")

    def test_stdin_closed_without_parent_pipe_changes_nothing(self):
        p = self.start(stdin=subprocess.PIPE)
        p.p.stdin.close()
        time.sleep(2)
        self.assertIsNone(p.p.poll())                   # (llama-swap, systemd, a run script: as before)

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sigterm(self):
        p = self.start(stdin=subprocess.DEVNULL)
        p.p.send_signal(signal.SIGTERM)
        self.ended(p, "SIGTERM")

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sighup(self):
        p = self.start(stdin=subprocess.DEVNULL)
        p.p.send_signal(signal.SIGHUP)
        self.ended(p, "SIGHUP")


DRIVER = """
import sys
sys.path.insert(0, {root!r})
sys.argv = ["maya.py"]
import maya
maya.stop_signals()
print("maya pid", flush=True)
sys.exit(maya.run_server({cmd!r}, wait_s=60))
"""


@unittest.skipIf(psutil is None, "psutil is not installed")
class MayaStops(unittest.TestCase):
    """maya.run_server as maya.py runs it: a stop of maya.py alone - SIGTERM, SIGHUP, even SIGKILL / Task Manager -
    ends the server and its engine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg = fake_config(Path(self.tmp.name))
        code = DRIVER.format(root=str(ROOT), cmd=server_cmd(cfg, free_port()))
        self.maya = Proc([sys.executable, "-c", code], stdin=subprocess.DEVNULL)
        self.addCleanup(self.cleanup)
        self.assertTrue(self.maya.until("ready:"), "the server did not start:\n" + "\n".join(self.maya.seen))
        self.tree = descendants(self.maya.p.pid)        # the server (and venv launchers) and the engine
        self.assertGreaterEqual(len(self.tree), 2 if FAKE_ENGINE else 1, self.tree)

    def cleanup(self):
        kill_all([self.maya.p.pid] + self.tree)
        self.maya.close()

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sigterm_to_maya_alone(self):
        self.maya.p.send_signal(signal.SIGTERM)
        self.assertEqual(self.maya.p.wait(timeout=60), 0)          # (it waited for the server, which ended well)
        self.assertIn("[maya] stopping (the program that started it ended)", self.maya.rest())
        self.assertEqual(gone(self.tree), [])

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sighup_to_maya_alone(self):
        self.maya.p.send_signal(signal.SIGHUP)
        self.assertEqual(self.maya.p.wait(timeout=60), 0)
        self.assertEqual(gone(self.tree), [])

    def test_maya_killed(self):
        """SIGKILL (POSIX: the pipe's EOF stops the server, which ends its engine) or TerminateProcess (Windows: the
        job object ends them at once)."""
        self.maya.p.kill()
        self.maya.p.wait(timeout=30)
        self.assertEqual(gone(self.tree), [])
        if not WIN:
            self.assertIn("[maya] stopping (the program that started it ended)", self.maya.rest())


SCREEN = """
import sys, threading
sys.path[:0] = [{root!r}, {tools!r}]
from setup_bridge import Bridge
b = Bridge(None)
b.thread = threading.current_thread()
sys.exit(b.serve({cmd!r}, None, {{}}))
"""


def listening(port: int, timeout: float = 60) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


@unittest.skipIf(psutil is None, "psutil is not installed")
class ScreenStops(unittest.TestCase):
    """The server on the setup's screen (Bridge.serve: in a session / process group of its own, out of reach of the
    terminal's signals) ends with maya.py too when maya.py is ended by force."""

    def test_maya_killed(self):
        with tempfile.TemporaryDirectory() as d:
            port = free_port()
            code = SCREEN.format(root=str(ROOT), tools=str(ROOT / "tools"),
                                 cmd=server_cmd(fake_config(Path(d)), port))
            maya = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL)
            try:
                self.assertTrue(listening(port), "the server did not start")
                tree = descendants(maya.pid)
                self.assertGreaterEqual(len(tree), 2 if FAKE_ENGINE else 1, tree)
                maya.kill()
                maya.wait(timeout=30)
                self.assertEqual(gone(tree), [])
            finally:
                kill_all(descendants(maya.pid) + [maya.pid])


TREE = """
import subprocess, sys, time
sys.path.insert(0, {root!r})
import setup
inner = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)"
p = subprocess.Popen([sys.executable, "-c", inner])
print("job", setup.end_with_me(p), flush=True)
time.sleep(120)
"""


@unittest.skipUnless(WIN and psutil is not None, "Windows' job objects (and psutil)")
class WindowsJob(unittest.TestCase):
    """setup.end_with_me: a process and the processes it starts - grandchildren too - end the moment the process
    that set it up is ended by force (TerminateProcess: taskkill /F, Task Manager)."""

    def test_the_tree_ends_with_its_owner(self):
        owner = Proc([sys.executable, "-c", TREE.format(root=str(ROOT))])
        self.addCleanup(owner.close)
        self.addCleanup(lambda: kill_all(descendants(owner.p.pid) + [owner.p.pid]))
        self.assertTrue(owner.until("job True", 30), owner.seen)
        end = time.monotonic() + 20
        while len(descendants(owner.p.pid)) < 2 and time.monotonic() < end:   # (the grandchild started)
            time.sleep(0.1)
        tree = descendants(owner.p.pid)
        self.assertGreaterEqual(len(tree), 2, tree)
        owner.p.kill()
        owner.p.wait(timeout=30)
        self.assertEqual(gone(tree, 15), [])


if __name__ == "__main__":
    unittest.main()
