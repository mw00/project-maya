"""Two GPUs of different makers in one PC as one DeepSeek-V4.1 model (on by itself where it is found: a DeepSeek
config on an AMD GPU (HIP), Linux, an NVIDIA card that nvidia-smi lists and a CUDA build of maya-ds4 beside the HIP one;
"mixed_gpus": false in the config / maya.py --mixed-gpus off turns it off, a layer count sets the card's share).  An AMD APU's GPU (the config's HIP engine) runs the trunk's later layers,
the head, the sampling and the drafter; an NVIDIA card beside it (a CUDA build of maya-ds4) the first layers - as many
as its free VRAM holds with every expert resident, measured at each start (maya-ds4 --mixed-plan).  A prompt's rows go
from one engine to the other through shared pinned memory.

One start: the server starts both engines (the HIP one as the "part", the CUDA one as the "front" the server talks to)
and watches both.  If either ends - the NVIDIA card or its link gone, a crash, a kill - the request in progress fails
with a clear line, and the next request starts the model on the HIP GPU alone (it loads the first layers then, which
the part never held: no RAM spent on them while both run); the next Maya start tries both GPUs again.

Answer-changing (on by default as the owner's exception): the NVIDIA card computes its layers in FP16, so answers can
be worded differently from the HIP GPU alone (prompt answers match in tests).  A request with pictures,
log-probabilities or refusal ablation runs on the HIP GPU alone (the engines are swapped for it, a load of a minute or
so), the next request on both again; a config with --parallel above 1 runs on the HIP GPU alone.  Linux only for now
(Windows has no shared-memory path for it yet): Windows and PCs without a second maker's card run as before."""
from __future__ import annotations

import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

PLAN_TIMEOUT_S = 180
PLAN_SETTLE_S = 90   # (back on both GPUs: how long the NVIDIA card may take to free the front's memory)
# found two GPUs and no "mixed_gpus" in the config: on by itself only where it is faster at decode too (the owner's
# rule) - Radeon 8065S + V100, Maya-XS: prompts +11/+39/+75 % at 2K/8K/32K but decode -1..-4 %, so not yet: the setup and
# the log say how to turn it on.  A prompt-reading-only mode (decode on the AMD GPU alone) would make it on by itself.
AUTO = False
# flags the two-GPU run leaves out: abliteration's projection is not carried through the link (the engines would
# disagree), and the AMD GPU's VRAM reserve is not the NVIDIA card's
ABLATE_FLAGS = {"--ablate": 1, "--ablate-default": 1, "--ablate-layer": 1, "--ablate-scale": 1}


class Unavailable(Exception):
    """Why the two-GPU run cannot start: the model then runs on the HIP GPU alone (again: a later start may work)."""

    def __init__(self, why: str, again: bool = True):
        super().__init__(why)
        self.again = again


def nvidia_listed() -> bool:
    """nvidia-smi lists a card (Linux; no nvidia-smi: none)."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and "GPU 0" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def windows() -> bool:
    return os.name == "nt"


def cuda_exe(cfg: dict) -> str:
    return cfg.get("mixed_exe") or str(Path(cfg["exe"]).resolve().parents[1] / "build-cuda" / Path(cfg["exe"]).name)


def auto_found(cfg: dict) -> bool:
    """Two GPUs found for a config that does not set "mixed_gpus": Linux, a DeepSeek config on the HIP engine, an
    NVIDIA card listed and the CUDA build of maya-ds4 there (the card's layer room is measured at the start)."""
    return (not windows() and cfg.get("backend") == "hip" and "maya-ds4" in Path(str(cfg.get("exe", ""))).name
            and Path(cuda_exe(cfg)).is_file() and nvidia_listed())


def config_of(cfg: dict) -> dict | None:
    """The config's two-GPU setting ("mixed_gpus": true / a layer count / false; not set: on when auto_found), None
    when off or not for this config (a DeepSeek config on the HIP engine)."""
    v = cfg.get("mixed_gpus")
    if v is None:
        if not auto_found(cfg):
            return None   # (Windows, no NVIDIA card or no CUDA build: as before, not a line said)
        if not AUTO:
            print("[maya] two GPUs: an NVIDIA card beside the AMD GPU - `maya --mixed-gpus on` runs DeepSeek on both "
                  "(long prompts read much faster, decode about the same; answers can be worded slightly differently)",
                  flush=True)
            return None
        print("[maya] two GPUs: an NVIDIA card beside the AMD GPU - DeepSeek runs on both (on by itself; "
              "`maya --mixed-gpus off` or \"mixed_gpus\": false in the config turns it off)", flush=True)
        v = True
    if v is False or v == 0 or str(v).lower() in ("off", "false", "0", ""):
        return None
    if cfg.get("backend") != "hip" or "maya-ds4" not in Path(str(cfg.get("exe", ""))).name:
        print("[maya] two GPUs (mixed_gpus): only a DeepSeek config on an AMD GPU (HIP) with an NVIDIA card beside it "
              "- this config runs as before", flush=True)
        return None
    layers = "auto" if v is True or str(v).lower() in ("on", "true", "auto") else int(v)
    return {"layers": layers, "exe": cuda_exe(cfg), "gpu": int(cfg.get("mixed_gpu", 0))}


def strip_flags(args: list[str], flags: dict[str, int]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] in flags:
            i += 1 + flags[args[i]]
            continue
        out.append(args[i])
        i += 1
    return out


def set_flag(args: list[str], flag: str, value: str, keep: bool = False) -> list[str]:
    """args with `flag value` (replaced; keep: a value already given stays)."""
    if flag in args[:-1]:
        if not keep:
            args[args.index(flag) + 1] = value
        return args
    return args + [flag, value]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def end_with_parent():
    """(preexec_fn) Linux: the part ends when the server does (SIGTERM on the parent's death)."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not Linux: the server's stop ends it
        pass


def plan(exe: str, pack: str, env: dict, cwd: str | None) -> tuple[int, str]:
    """maya-ds4 --mixed-plan on the NVIDIA card: (layers that fit, the numbers it went by)."""
    try:
        r = subprocess.run([exe, "--pack", pack, "--mixed-plan"], cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=PLAN_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Unavailable(f"the NVIDIA card's engine did not answer ({e})") from None
    m = re.search(r"^MIXED_PLAN (\d+) (.*)$", r.stdout, re.M)
    if r.returncode != 0 or not m:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise Unavailable(f"the NVIDIA card's engine could not plan its layers ({tail[0]})")
    return int(m.group(1)), m.group(2)


def engine_class(base, wait_gpu_release):
    """The server's engine class for a two-GPU config: `base` (StrataEngine) with the second engine beside it."""

    class MixedEngine(base):
        mixed_off = None   # why the two-GPU run stopped (then: the HIP GPU alone until Maya starts again)

        def __init__(self, exe, args, cwd=None, log=None, env=None, mixed=None):
            if mixed is not None:
                self.mixed_cfg = mixed
            self.mixed_on, self.part, self._closed = False, None, False
            spawn = (exe, list(args), cwd, log, env)
            if getattr(self, "mixed_cfg", None) and not self.mixed_off:
                try:
                    self._start_mixed(spawn)
                    self.spawn = spawn   # (restart / set_context: the HIP engine's command, both engines start from it)
                    return
                except Unavailable as e:
                    p = getattr(self, "proc", None)   # (a front that started: ended before the AMD GPU loads alone)
                    if p is not None and p.poll() is None:
                        p.kill()
                        p.wait(timeout=20)
                        wait_gpu_release(p.pid)
                    self._stop_part()
                    self._fall(str(e), e.again)
            base.__init__(self, *spawn)

        def _fall(self, why: str, again: bool = True):
            self.mixed_off = why
            print(f"[maya] two GPUs (--mixed-gpus): {why} - the model runs on the AMD GPU alone" +
                  (" from here (it loads all its layers now; Maya's next start tries both GPUs again)" if again else ""),
                  flush=True)

        def _start_mixed(self, spawn):
            exe, args, cwd, log, env = spawn
            m = self.mixed_cfg
            if os.name == "nt":
                raise Unavailable("not on Windows yet (no shared-memory path between the two engines there)", False)
            if "--parallel" in args[:-1] and int(args[args.index("--parallel") + 1]) > 1:
                raise Unavailable("it runs one conversation at a time, and this config has --parallel "
                                  f"{args[args.index('--parallel') + 1]}", False)
            if not Path(m["exe"]).is_file():
                raise Unavailable(f"no CUDA build of maya-ds4 at {m['exe']} (mixed_exe in the config names another)",
                                  False)
            if "--pack" not in args[:-1]:
                raise Unavailable("the config names no --pack", False)
            pack = args[args.index("--pack") + 1]
            common = set_flag(strip_flags(list(args), ABLATE_FLAGS), "--threads", "on")
            ablate = "--ablate" in args[:-1]   # (a request that ablates runs on the AMD GPU alone, which loads it)
            fenv = dict(env or os.environ)
            fenv.pop("HIP_VISIBLE_DEVICES", None)
            fenv["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            fenv["CUDA_VISIBLE_DEVICES"] = str(m["gpu"])
            k, why = plan(m["exe"], pack, fenv, cwd)
            # back on both GPUs after a request ran on the AMD GPU alone: the card frees the front's memory in its own
            # time, so plan again until it fits as many layers as the first start found (or PLAN_SETTLE_S passed)
            best, t0 = getattr(self, "k_seen", 0), time.time()
            while k < best and time.time() - t0 < PLAN_SETTLE_S:
                time.sleep(2.0)
                k, why = plan(m["exe"], pack, fenv, cwd)
            if k < best:
                print(f"[maya] two GPUs: the NVIDIA card has less room than at the first start after "
                      f"{time.time() - t0:.0f} s", flush=True)
            elif best and time.time() - t0 > 1:
                print(f"[maya] two GPUs: the NVIDIA card had its memory back after {time.time() - t0:.0f} s", flush=True)
            self.k_seen = max(best, k)
            print(f"[maya] two GPUs: {why}", flush=True)
            if m["layers"] != "auto":
                k = int(m["layers"])
                print(f"[maya] two GPUs: {k} layers on the NVIDIA card, as the config says (mixed_gpus)", flush=True)
            if k < 1:
                raise Unavailable("the NVIDIA card has no room for one layer with its experts")
            port = free_port()
            part_log = (str(Path(log).with_suffix("")) + ".part.log") if log else os.devnull
            self.part_log = part_log
            with open(part_log, "a", encoding="utf-8") as plog:
                self.part = subprocess.Popen([exe, *common, "--part-from", str(k), "--part-serve", str(port)], cwd=cwd,
                                             stdin=subprocess.DEVNULL, stdout=plog, stderr=subprocess.STDOUT, env=env,
                                             preexec_fn=end_with_parent if os.name != "nt" else None)
            front = strip_flags(common, {"--reserve-mib": 1})
            front = set_flag(front, "--remote-part", f"127.0.0.1:{port}")
            front = set_flag(front, "--mmap-experts", "on")   # (its own experts all in VRAM: none copied into RAM)
            front = set_flag(front, "--prefill-chunk", "4096", keep=True)   # (measured: 1024 read slower through it)
            fenv["MAYA_MIXED_FRONT"] = str(k)
            print(f"[maya] two GPUs: layers 0-{k - 1} on the NVIDIA card, the rest on the AMD GPU (its log: {part_log})",
                  flush=True)
            ready = threading.Event()
            threading.Thread(target=self._watch_load, args=(ready,), daemon=True).start()
            try:
                base.__init__(self, m["exe"], front, cwd, log, fenv)
            except RuntimeError as e:
                why = "the AMD GPU's engine stopped while loading" if self.part.poll() is not None else str(e)
                raise Unavailable(f"{why} (logs: {log}, {part_log})") from None
            finally:
                ready.set()
            if self.part.poll() is not None:
                raise Unavailable(f"the AMD GPU's engine stopped (exit {self.part.returncode}; its log: {part_log})")
            self.mixed_on = True
            self.info["mixed_gpus"] = k
            if ablate:   # (INFO ablate as the AMD GPU alone has it: a request may ask, and then runs there)
                d = args[args.index("--ablate-default") + 1] if "--ablate-default" in args[:-1] else "off"
                self.info["ablate"] = 1
                self.info["ablate_default"] = 1 if d == "on" else 0
            print("[maya] two GPUs: on - `maya --mixed-gpus off` (or \"mixed_gpus\": false in the config) runs the "
                  "model on the AMD GPU alone", flush=True)

        def use(self, alone: str | None):
            """Before a request (the caller holds the engine): alone (why) - the AMD GPU alone for it (the two engines
            stopped, it loads every layer); None - both GPUs again if this one ran alone for an earlier request."""
            spawn = getattr(self, "spawn", None)
            if spawn is None or self.mixed_off or self._closed:
                return
            keep = (self._rids, self._routes, self._routes_lock)   # (as restart(): the request numbers go on)
            try:
                self._use(alone, spawn)
            finally:
                self._rids, self._routes, self._routes_lock = keep

        def _use(self, alone, spawn):
            if alone and self.mixed_on:
                print(f"[maya] two GPUs: this request {alone} - it runs on the AMD GPU alone (loading it, a minute or "
                      "so); the next request without it runs on both again", flush=True)
                base.stop(self)
                self._stop_part()
                self.ended = False
                self.mixed_on = False
                self.alone_for = alone
                base.__init__(self, *spawn)
            elif not alone and getattr(self, "alone_for", None) and not self.mixed_on:
                print("[maya] two GPUs: back on both GPUs (loading them, a minute or so)", flush=True)
                base.stop(self)
                self.ended = False
                self.alone_for = None
                try:
                    self._start_mixed(spawn)
                    self.spawn = spawn   # (as at the first start: the HIP engine's command)
                except Unavailable as e:
                    self.spawn = spawn
                    p = getattr(self, "proc", None)
                    if p is not None and p.poll() is None:
                        p.kill()
                        p.wait(timeout=20)
                        wait_gpu_release(p.pid)
                    self._stop_part()
                    self._fall(str(e), e.again)
                    base.__init__(self, *spawn)

        def _watch_load(self, ready: threading.Event):
            """While the front loads (it waits for the part): a part that ends ends the front too, at once."""
            while not ready.wait(1.0):
                if self.part is not None and self.part.poll() is not None:
                    p = getattr(self, "proc", None)
                    if p is not None and p.poll() is None:
                        p.kill()
                    return

        def _part_alive(self) -> bool:
            return self.part is not None and self.part.poll() is None

        def _stop_part(self):
            p, self.part = self.part, None
            if p is None:
                return
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=20)
            wait_gpu_release(p.pid)

        def death_note(self) -> str:
            if not self.mixed_on:
                return base.death_note(self)
            side = "the NVIDIA card's engine" if self._part_alive() else "the AMD GPU's engine"
            return (f"Two GPUs (--mixed-gpus): {side} ended, so this request could not finish; the next request runs the "
                    f"model on the AMD GPU alone (logs: {self.log_path}, {getattr(self, 'part_log', '')}).")

        def alive(self) -> bool:
            return base.alive(self) and (not self.mixed_on or self._part_alive())

        def restart(self):
            if self.mixed_on:
                if not self._closed and not (base.alive(self) and self._part_alive()):
                    who = "the NVIDIA card's engine" if self._part_alive() else "the AMD GPU's engine"
                    if self._part_alive() and getattr(self, "proc", None) is not None:
                        code = self.proc.poll()
                        who += f" ended (exit {code})" if code is not None else " ended"
                    else:
                        who += " ended"
                    self._fall(who)
                self._stop_part()
            self._closed = False
            base.restart(self)

        def stop(self, timeout: float = 60.0):
            base.stop(self, timeout)
            self._stop_part()

        def close(self):
            self._closed = True
            base.close(self)
            self._stop_part()

    return MixedEngine
