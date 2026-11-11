# CUDA checkpoint code adapted from Xolotl; see LICENSE.xolotl-checkpoint.
"""CUDA process checkpoint used by Maya's opt-in sleep endpoint.

The driver stores device allocations in host memory and releases CUDA state;
restore returns them at the same addresses. The caller must exclude all model
requests and other lifecycle operations during checkpoint and restore.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import queue
import signal
import sys
import threading
import time

MODES = ("auto", "checkpoint", "off")
SLEEP_TIMEOUT_S = 300.0      # native: the engine's device-to-host copies and its context release
WAKE_TIMEOUT_S = 600.0       # native: a new context and the engine's uploads
LOCK_TIMEOUT_MS = 30000      # checkpoint: the driver waits this long for the engine's CUDA calls to finish

# the driver's CUprocessState
RUNNING, LOCKED, CHECKPOINTED, FAILED = 0, 1, 2, 3


class SleepRefused(ValueError):
    """The engine did not go to sleep (it stays awake), or cannot sleep at all."""


class WakeRefused(RuntimeError):
    """The engine did not wake (it stays asleep): most often the GPU is not free."""


class EngineGone(RuntimeError):
    """The engine ended or stopped answering during a sleep or a wake."""


def mode_from_config(cfg: dict) -> str:
    """Sleep is opt-in; auto currently selects the CUDA checkpoint backend."""
    mode = cfg.get("deep_sleep", "off")
    if mode is None or mode is False:
        return "off"
    if mode is True:
        mode = "checkpoint"
    if mode not in MODES:
        raise ValueError(f'deep_sleep={mode!r}: expected one of {", ".join(MODES)}')
    return "checkpoint" if mode == "auto" else mode


def figures(line: str) -> dict:
    """The key=value fields of an engine line (the first word is the line's kind)."""
    out = {}
    for kv in line.split()[1:]:
        key, _, value = kv.partition("=")
        if not value:
            continue
        try:
            out[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            out[key] = value
    return out


class CheckpointError(RuntimeError):
    pass


class CudaCheckpoint:
    """ctypes on libcuda's cuCheckpointProcess* (in the server process: cuInit makes no GPU context)."""
    _shared = None
    _lock = threading.Lock()

    def __init__(self, lib):
        self.lib = lib
        for name in ("cuCheckpointProcessGetState", "cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint",
                     "cuCheckpointProcessRestore", "cuCheckpointProcessUnlock"):
            getattr(lib, name).restype = ctypes.c_int
        rc = lib.cuInit(0)
        if rc != 0:
            raise CheckpointError(f"cuInit: {self._name(rc)}")

    @classmethod
    def get(cls):
        """The driver, or None when this host has no CUDA driver with process checkpoints (cached)."""
        if sys.platform != "linux":
            cls.unavailable = "CUDA checkpoint sleep is supported only on Linux"
            return None
        with cls._lock:
            if cls._shared is None:
                try:
                    path = ctypes.util.find_library("cuda") or "libcuda.so.1"
                    cls._shared = cls(ctypes.CDLL(path))
                except (OSError, AttributeError, CheckpointError) as e:
                    cls._shared = False
                    cls.unavailable = str(e)
            return cls._shared or None

    def _name(self, rc: int) -> str:
        s = ctypes.c_char_p()
        try:
            if self.lib.cuGetErrorName(rc, ctypes.byref(s)) == 0 and s.value:
                return f"{s.value.decode()} ({rc})"
        except AttributeError:
            pass
        return f"CUresult {rc}"

    def _call(self, what: str, fn, pid: int, *args):
        rc = fn(int(pid), *args)
        if rc != 0:
            raise CheckpointError(f"{what} of process {pid}: {self._name(rc)}")

    def state(self, pid: int) -> int | None:
        """The process's CUprocessState, or None when the driver does not know it as a CUDA process."""
        st = ctypes.c_int(-1)
        return st.value if self.lib.cuCheckpointProcessGetState(int(pid), ctypes.byref(st)) == 0 else None

    def lock(self, pid: int, timeout_ms: int):
        class LockArgs(ctypes.Structure):
            _fields_ = [("timeoutMs", ctypes.c_uint), ("reserved0", ctypes.c_uint), ("reserved1", ctypes.c_uint64 * 7)]
        self._call("lock", self.lib.cuCheckpointProcessLock, pid, ctypes.byref(LockArgs(int(timeout_ms))))

    def checkpoint(self, pid: int):
        self._call("checkpoint", self.lib.cuCheckpointProcessCheckpoint, pid, None)

    def restore(self, pid: int):
        self._call("restore", self.lib.cuCheckpointProcessRestore, pid, None)

    def unlock(self, pid: int):
        self._call("unlock", self.lib.cuCheckpointProcessUnlock, pid, None)


driver = CudaCheckpoint.get      # tests replace it


def _process_start(pid: int) -> int | None:
    """Linux process birth time; a saved PID alone is not a process identity."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="ascii") as f:
            # comm may contain spaces and parentheses. Fields after its final ')'
            # start at state (3); starttime is field 22.
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class ProcessId(int):
    """A CUDA PID and the process birth time captured before checkpointing."""
    def __new__(cls, pid):
        value = int.__new__(cls, pid)
        value.start = _process_start(value)
        return value

    def matches(self):
        return self.start is not None and _process_start(self) == self.start


def _require_process(pid):
    if not isinstance(pid, ProcessId) or not pid.matches():
        raise _Stuck(f"checkpoint process {int(pid)} no longer matches the original process")


def _children(pid: int) -> list[int]:
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{tid}/children", encoding="ascii") as f:
                out += [int(x) for x in f.read().split()]
    except (OSError, ValueError):
        pass
    return out


def cuda_processes(pid: int, drv) -> list[int]:
    """The engine's process and its descendants that the driver knows as running CUDA processes (a launcher script
    that does not exec the engine leaves the engine as its child)."""
    todo, seen, out = [int(pid)], set(), []
    while todo:
        p = todo.pop(0)
        if p in seen:
            continue
        seen.add(p)
        identity = ProcessId(p)
        if identity.matches() and drv.state(p) == RUNNING:
            out.append(identity)
        todo += _children(p)
    return out


def _rss_mib(pids) -> int | None:
    total = 0
    try:
        for p in pids:
            with open(f"/proc/{p}/status", encoding="ascii") as f:
                total += next(int(x.split()[1]) for x in f if x.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    return total // 1024


def _kill_processes(pids):
    """End each tracked CUDA process, including a launcher's descendants. A
    checkpointed process cannot handle QUIT until restored."""
    for pid in reversed(pids):
        if not isinstance(pid, ProcessId) or not pid.matches():
            continue  # exited/reused or never tracked: never signal an unrelated process
        try:
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                try:
                    fd = os.pidfd_open(pid)
                except OSError as e:
                    if e.errno not in (errno.ENOSYS, errno.EINVAL):
                        raise
                    # Older kernels may expose the Python API without the syscall.
                    if pid.matches():
                        os.kill(pid, signal.SIGKILL)
                    continue
                try:
                    if pid.matches():  # check again after obtaining the race-free signal handle
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class _Stuck(RuntimeError):
    """A checkpoint or restore failed and could not be undone: the processes must be ended."""


def _checkpoint(drv, pids) -> float:
    """Lock and checkpoint every process, all or none -> when the locks were taken.  CheckpointError: nothing changed
    (the ones done are restored and unlocked again); _Stuck: that undo failed too."""
    for p in pids:
        _require_process(p)
    locked, saved = [], []
    try:
        for p in pids:
            _require_process(p)
            drv.lock(p, LOCK_TIMEOUT_MS)
            locked.append(p)
        t1 = time.perf_counter()
        for p in pids:
            _require_process(p)
            drv.checkpoint(p)
            saved.append(p)
        return t1
    except CheckpointError as e:
        try:
            for p in saved:
                _require_process(p)
                drv.restore(p)
            for p in locked:
                _require_process(p)
                drv.unlock(p)
        except CheckpointError as e2:
            raise _Stuck(f"the checkpoint failed ({e}) and could not be undone ({e2})") from None
        raise


def _restore(drv, pids) -> float:
    """Restore and unlock every process, all or none -> when the restores were done.  CheckpointError: they stay
    checkpointed; _Stuck: the undo failed."""
    for p in pids:
        _require_process(p)
    done = []
    try:
        for p in pids:
            _require_process(p)
            drv.restore(p)
            done.append(p)
    except CheckpointError as e:
        try:
            for p in done:                       # all or none: the ones restored go back to their checkpoint
                _require_process(p)
                drv.checkpoint(p)
        except CheckpointError as e2:
            raise _Stuck(f"the restore failed ({e}) and could not be undone ({e2})") from None
        raise
    t1 = time.perf_counter()
    try:
        for p in pids:
            _require_process(p)
            drv.unlock(p)
    except CheckpointError as e:
        # Earlier processes may already be running. Never advertise this mixed
        # state as safely asleep, or retry restore on an unlocked process.
        raise _Stuck(f"the restored processes could not all be unlocked ({e})") from None
    return t1


def helper_pid(vision) -> int | None:
    """The process of a vision helper (the common Vision's, or a client's), when it runs."""
    proc = getattr(vision, "proc", None) or getattr(getattr(vision, "client", None), "process", None)
    try:
        return proc.pid if proc is not None and proc.poll() is None else None
    except (AttributeError, OSError):
        return None


def freeze(pid: int) -> list[int] | None:
    """Checkpoint a helper process (and its CUDA descendants) as an engine is: its VRAM goes, the same bytes come back
    with thaw().  None: the driver cannot (no driver, not a CUDA process); CheckpointError: it failed, nothing changed."""
    drv = driver()
    pids = cuda_processes(pid, drv) if drv is not None else []
    if not pids:
        return None
    try:
        _checkpoint(drv, pids)
    except _Stuck as e:
        _kill_processes(pids)
        raise CheckpointError(str(e)) from None
    return pids


def thaw(pids: list[int]) -> None:
    """freeze()'s processes back.  CheckpointError: they could not be restored (the caller ends them)."""
    try:
        _restore(driver(), pids)
    except _Stuck as e:
        raise CheckpointError(str(e)) from None


def method(engine) -> str | None:
    """How this engine would sleep now: "native", "checkpoint" or None (it cannot)."""
    if getattr(engine, "sleeping", False):
        return getattr(engine, "sleep_via", None)
    mode = getattr(engine, "deep_sleep", "off")
    if mode == "off" or not engine.alive():
        return None
    if mode in ("auto", "native") and str((getattr(engine, "info", None) or {}).get("sleep", "0")) == "1":
        return "native"
    if mode in ("auto", "checkpoint"):
        drv = driver()
        proc = getattr(engine, "proc", None)
        if drv is not None and proc is not None and cuda_processes(proc.pid, drv):
            return "checkpoint"
    return None


def _command(engine, command: str, answer: str, timeout: float):
    """One control line between requests: (answer line or None, the ERR text or None, key=value figures of the
    answer and of the INFO lines before it)."""
    engine._send(command)
    deadline, got = time.monotonic() + timeout, {}
    while True:
        try:
            line = engine.lines.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            raise EngineGone(f"the engine did not answer {command}") from None
        if line is None:
            raise EngineGone("the engine ended")
        line = line.strip()
        if line.startswith("ERR"):
            return None, line[4:].strip() or f"the engine refused {command}", got
        if line.startswith("INFO "):
            got.update(figures(line))
        elif line.startswith(answer):
            return line, None, {**got, **(figures(line) if answer != "READY" else {})}
        if time.monotonic() > deadline:
            raise EngineGone(f"the engine did not answer {command}")


def _checkpoint_control(engine, command: str, answer: str) -> dict:
    """Negotiated idle control operation. A lost acknowledgement is fatal: never
    expose a running process with an uncertain checkpoint-clock state."""
    try:
        line, error, got = _command(engine, command, answer, WAKE_TIMEOUT_S)
        if error:
            raise EngineGone(f"{command} failed: {error}")
        if not line or line.split()[0] != answer:
            raise EngineGone(f"{command} returned an invalid acknowledgement")
        if command == "CHECKPOINT_RESUME":
            # Require telemetry on the ACK itself, not an earlier INFO line.
            telemetry = figures(line)
            for key in ("paused_us", "graph_execs"):
                if type(telemetry.get(key)) is not int or telemetry[key] < 0:
                    raise EngineGone(f"{command} returned invalid {key}")
            got.update(telemetry)
        return got
    except (EngineGone, OSError) as error:
        _kill_processes(getattr(engine, "sleep_pids", []))
        _kill_processes([getattr(engine, "sleep_launcher", None)])
        engine.sleeping, engine.sleep_via = False, None
        engine.checkpoint_paused = False
        engine.info.pop("asleep", None)
        raise EngineGone(f"{error}; the checkpoint-controlled engine was ended") from None


def _resume_checkpoint_clock(engine) -> dict:
    got = _checkpoint_control(engine, "CHECKPOINT_RESUME", "CHECKPOINT_RESUMED")
    engine.checkpoint_paused = False
    return got


def sleep(engine) -> dict:
    """Put the engine to sleep (the caller holds the request turn; no request runs) -> figures.  SleepRefused: it
    stays awake.  EngineGone: it ended."""
    how = method(engine)
    if how is None:
        raise SleepRefused("this engine cannot sleep (no native sleep, and the CUDA driver cannot checkpoint it)")
    if engine.sleeping:
        return {}
    t0 = time.perf_counter()
    if how == "native":
        _, error, got = _command(engine, "SLEEP", "SLEPT", SLEEP_TIMEOUT_S)
        if error:
            raise SleepRefused(error)
    else:
        drv = driver()
        pids = cuda_processes(engine.proc.pid, drv)
        rss0 = _rss_mib(pids)
        engine.sleep_pids = pids
        engine.sleep_launcher = ProcessId(engine.proc.pid)
        _require_process(engine.sleep_launcher)
        if str((getattr(engine, "info", None) or {}).get("checkpoint_clock", "0")) == "1":
            # The engine owns a monotonic timestamp: exclude the inactive interval
            # from graph TTL without discarding captures, tensors or cache state.
            _checkpoint_control(engine, "CHECKPOINT_PAUSE", "CHECKPOINT_PAUSED")
            engine.checkpoint_paused = True
        try:
            t1 = _checkpoint(drv, pids)
        except CheckpointError as e:
            # _checkpoint rolled back and unlocked every process. Its clock must
            # resume too; a failed restore, in contrast, retains the original pause.
            if getattr(engine, "checkpoint_paused", False):
                _resume_checkpoint_clock(engine)
            raise SleepRefused(f"the CUDA checkpoint failed: {e}") from None
        except _Stuck as e:
            _kill_processes(pids)
            _kill_processes([engine.sleep_launcher])  # half checkpointed: never served again
            raise EngineGone(f"{e}; the engine was ended") from None
        rss1 = _rss_mib(pids)
        got = {"seconds": round(time.perf_counter() - t0, 3), "lock_seconds": round(t1 - t0, 3),
               "checkpoint_seconds": round(time.perf_counter() - t1, 3), "processes": len(pids)}
        if rss0 is not None and rss1 is not None:
            got["host_copy_mib"] = rss1 - rss0   # the device memory the driver now keeps in the engine's RAM
        engine.sleep_pids = pids
    got.setdefault("seconds", round(time.perf_counter() - t0, 3))
    engine.sleeping, engine.sleep_via = True, how
    engine.info["asleep"] = {"method": how, **got}
    return {"method": how, **got}


def wake(engine) -> dict:
    """Wake a sleeping engine -> figures ({} when it was awake).  WakeRefused: it stays asleep."""
    if not engine.sleeping:
        return {}
    t0 = time.perf_counter()
    if engine.sleep_via == "native":
        _, error, got = _command(engine, "WAKE", "READY", WAKE_TIMEOUT_S)
        if error:
            raise WakeRefused(error)
    else:
        try:
            t1 = _restore(driver(), list(getattr(engine, "sleep_pids", [])))
        except CheckpointError as e:
            raise WakeRefused(f"the CUDA restore failed: {e}") from None
        except _Stuck as e:
            _kill_processes(getattr(engine, "sleep_pids", []))
            _kill_processes([getattr(engine, "sleep_launcher", None)])
            raise EngineGone(f"{e}; the engine was ended") from None
        got = {"restore_seconds": round(t1 - t0, 3), "unlock_seconds": round(time.perf_counter() - t1, 3)}
        if getattr(engine, "checkpoint_paused", False):
            got.update(_resume_checkpoint_clock(engine))
    got.setdefault("wake_seconds", round(time.perf_counter() - t0, 3))
    engine.sleeping, engine.sleep_via = False, None
    engine.info.pop("asleep", None)
    engine.info["wake"] = got
    return got


def end(engine) -> bool:
    """Before an engine that sleeps by checkpoint is closed: a checkpointed process cannot free its device memory
    itself (its CUDA calls wait for a restore), so it is killed; the driver frees what it kept.  True when killed."""
    if getattr(engine, "sleeping", False) and getattr(engine, "sleep_via", None) == "checkpoint":
        _kill_processes(getattr(engine, "sleep_pids", []))
        _kill_processes([getattr(engine, "sleep_launcher", None)])
        engine.sleeping, engine.sleep_via = False, None
        return True
    return False
