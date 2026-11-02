"""Downloading models from the Models page and the API (/api/pull).  Nothing here downloads or verifies a file
itself: the download is the setup's own (`maya.py --setup --model <id> --download-model --yes --plain --no-start`: curl
with resume, sha256 check, the pack, the config), run as a child process whose output is read for its progress line.
One download at a time."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import maya_quants as Q  # noqa: E402  (tools/) the quants the setup offers

from serve import models  # noqa: E402

# the setup's progress line: "12.3 / 96.5 GB (13%) · 40 MB/s · 4 files at once"
PROGRESS = re.compile(r"([\d.]+) / ([\d.]+) GB(?: \((\d+)%\))?(?: · ([\d.]+) MB/s)?")


def catalog(root: Path, glm_sizes: dict | None = None) -> list:
    """The models the setup can download: [{"id" (the setup's --model id), "family", "name" (as users see it),
    "installed" (a config here runs it), "size_gb" (None when the setup tells it at its start)}]."""
    have = set()
    for p in models.config_paths(root):
        cfg = models.read(p)
        if isinstance(cfg.get("installer"), dict):
            have.add((Q.family_of(cfg), str(Q.config_key(cfg))))
    sizes = glm_sizes if glm_sizes is not None else _glm_sizes()
    out = []
    for fam, key, _tier, ident in Q.QUANTS:
        out.append({"id": ident, "family": fam, "name": Q.shown(fam, key), "installed": (fam, key) in have,
                    "size_gb": sizes.get(key)})
    out.append({"id": "cyber-frost", "family": "qwen", "name": Q.shown("qwen", "q5_k_m"),
                "installed": ("qwen", "q5_k_m") in have, "size_gb": None})
    return out


def _glm_sizes() -> dict:
    try:
        sys.path.insert(0, str(ROOT))
        import maya                                    # (the setup's own table; nothing runs on import)
        return {k: m.get("download_gb") for k, m in maya.MODELS.items()}
    except Exception:  # noqa: BLE001 - a size is a nicety
        return {}


def setup_command(model_id: str, cfg: dict | None) -> list:
    """The setup run that downloads `model_id` the way this install was set up (its models folder and backend)."""
    inst = (cfg or {}).get("installer") or {}
    cmd = [sys.executable, str(ROOT / "maya.py"), "--setup", "--model", model_id, "--download-model", "--yes",
           "--plain", "--no-start"]
    if inst.get("models_dir"):
        cmd += ["--models-dir", str(inst["models_dir"])]
    if (cfg or {}).get("backend") in ("hip", "metal"):
        cmd += ["--backend", cfg["backend"]]
    return cmd


class Pull:
    """One download: its child process, the last progress and the last lines of the setup's output."""

    def __init__(self, model_id: str, command: list, cwd: Path = ROOT):
        self.id = model_id
        self.started = time.time()
        self.state = "starting"                        # starting, downloading, success, error, cancelled
        self.completed = self.total = 0
        self.speed = None
        self.error = None
        self.log: list = []
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, daemon=True).start()

    def _line(self, text: str):
        text = text.strip()
        if not text:
            return
        m = PROGRESS.search(text)
        with self._lock:
            if m:
                self.state = "downloading"
                self.completed, self.total = int(float(m.group(1)) * 1e9), int(float(m.group(2)) * 1e9)
                self.speed = float(m.group(4)) * 1e6 if m.group(4) else None
            else:
                self.log = (self.log + [text])[-40:]

    def _read(self):
        buf = b""
        while True:
            b = self.proc.stdout.read(1)               # (the setup rewrites its progress line with \r)
            if not b:
                break
            if b in (b"\r", b"\n"):
                self._line(buf.decode("utf-8", "replace"))
                buf = b""
            else:
                buf += b
        self._line(buf.decode("utf-8", "replace"))
        rc = self.proc.wait()
        with self._lock:
            if self.state == "cancelled":
                return
            if rc == 0:
                self.state = "success"
            else:
                self.state = "error"
                self.error = next((ln for ln in reversed(self.log) if ln.strip()), f"the setup stopped (code {rc})")

    def cancel(self):
        with self._lock:
            if self.state in ("success", "error", "cancelled"):
                return
            self.state = "cancelled"
        self.proc.terminate()

    def snapshot(self) -> dict:
        with self._lock:
            pct = int(100 * self.completed / self.total) if self.total else None
            return {"model": self.id, "state": self.state, "completed": self.completed, "total": self.total,
                    "percent": pct, "bytes_per_s": self.speed, "error": self.error, "started": self.started,
                    "message": self.log[-1] if self.log else "", "log": list(self.log[-12:])}

    def active(self) -> bool:
        return self.state in ("starting", "downloading")


class Puller:
    """The server's one download slot."""

    def __init__(self, root: Path, command=None):
        self.root = Path(root)
        self.command = command                         # tests: a function (model_id, cfg) -> argv
        self.current: Pull | None = None
        self._lock = threading.Lock()

    def start(self, ident: str, cfg: dict | None) -> Pull:
        known = {c["id"] for c in catalog(self.root, {})}
        ident = str(ident or "").strip().lower()
        ident = ident.removesuffix(":latest")
        if ident not in known:
            raise ValueError(f"{ident!r} is not in Maya's catalog (GET /api/catalog lists the ids)")
        with self._lock:
            if self.current is not None and self.current.active():
                raise ValueError(f"{self.current.id} is downloading; wait for it or cancel it")
            self.current = Pull(ident, (self.command or setup_command)(ident, cfg))
            return self.current

    def snapshot(self) -> dict | None:
        return self.current.snapshot() if self.current is not None else None

    def busy(self) -> bool:
        return self.current is not None and self.current.active()

    def cancel(self) -> bool:
        if self.busy():
            self.current.cancel()
            return True
        return False


def ndjson_events(pull: Pull, poll: float = 1.0):
    """The /api/pull stream for `pull`: {"status": ...} lines, with total / completed while it downloads, then
    "success" (or an {"error": ...} line)."""
    yield {"status": "pulling " + pull.id}
    last = None
    while True:
        s = pull.snapshot()
        if s["state"] == "downloading" and s["total"]:
            ev = {"status": f"downloading {pull.id}", "digest": pull.id, "total": s["total"], "completed": s["completed"]}
        elif s["state"] in ("starting", "downloading"):
            ev = {"status": s["message"] or "preparing"}
        elif s["state"] == "success":
            if s["total"]:                              # (a fast one finishes between two polls: the full bar anyway)
                yield {"status": f"downloading {pull.id}", "digest": pull.id, "total": s["total"], "completed": s["total"]}
            yield {"status": "verifying sha256 digest"}
            yield {"status": "success"}
            return
        else:
            yield {"error": s["error"] or s["state"]}
            return
        if ev != last:
            yield ev
            last = ev
        time.sleep(poll)
