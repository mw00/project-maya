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
from serve import glm_dsa_product as G53  # noqa: E402  GLM-5.3 (744B, glm-dsa)

# the setup's progress line: "12.3 / 96.5 GB (13%) · 40 MB/s · 4 files at once"
PROGRESS = re.compile(r"([\d.]+) / ([\d.]+) GB(?: \((\d+)%\))?(?: · ([\d.]+) MB/s)?")


# The other files the setup offers, each asked for by its own flag (maya.py: --ds4-quant, --mm3-quant, a GLM download
# by its name): (catalog id, family, the setup's key, who made them).  The Models page lists them under "Other quants".
# Left out as the setup's own list leaves them out: DeepSeek's UD-Q4_K_XL (maya.DS4_BY_HAND) and GLM's
# EXL3-sc_4.10bpw (maya.MODELS_BY_HAND) - a command line still takes them.
DS4_OTHER = (("exl3", "LibertAIDAI"), ("exl3-3.5", "LibertAIDAI"), ("gguf-q8", "unsloth"), ("gguf-iq1_s", "unsloth"),
             ("gguf-iq1_m", "unsloth"), ("gguf-iq2_xxs", "unsloth"), ("gguf-q2_k_xl", "unsloth"),
             ("gguf-iq3_xxs", "unsloth"), ("gguf-iq3_s", "unsloth"), ("gguf-q3_k_xl", "unsloth"),
             ("abliterated-128g", "orcarouter"), ("abliterated-256g", "orcarouter"))
GLM_EXL3 = ("EXL3-3.05bpw", "EXL3-4.05bpw")
# (Qwen3.8-Flash-Next itself: unsloth's UD-IQ2_XS only - its other UD quants run on the same code but are not listed;
#  the setup id: maya_quants.QWEN_UD_IDS[0])
QWEN_OTHER = (("qwen-ud-iq2_xs", "qwen", "ud-iq2_xs", "unsloth"),)
OTHER = tuple([(f"deepseek-{k}", "deepseek", k, by) for k, by in DS4_OTHER] + [("minimax-exl3", "minimax", "exl3", "Mia-AiLab")] +
              [(f"glm-{k.lower()}", "glm", k, "turboderp") for k in GLM_EXL3] + list(QWEN_OTHER))
GATED = ("abliterated-128g", "abliterated-256g")     # (orcarouter's repo is gated: the setup sends the user's HF token)
EXL3_MIN_CC = 70                                     # (maya.DS4_MIN_CC / mm3_min_cc: the EXL3 quants from V100 on)
QWEN_MIN_CC = 70                                     # (maya.QWEN_MIN_CC: the Qwen path from V100 on)


def _offered(fam: str, key: str, backend: str, ccs: list) -> bool:
    """Whether the setup offers these files on this PC (maya.py's own checks): DeepSeek and MiniMax not on Metal; an
    EXL3 quant and DeepSeek's GGUFs on NVIDIA from V100 on (no GPU readable: offered, the setup's PC check says) and on
    AMD; GLM's EXL3 quants anywhere but Metal."""
    if backend == "metal":
        return False
    if fam in ("deepseek", "minimax") and backend != "hip" and ccs:   # (their EXL3 quants and DeepSeek's GGUFs:
        return max(ccs) >= EXL3_MIN_CC                                  # maya-ds4's / maya-mm3's kernels, V100 on)
    if fam == "qwen" and backend != "hip" and ccs:                     # (unsloth's Qwen GGUF: the setup's check_pc)
        return max(ccs) >= QWEN_MIN_CC
    return True


def _note(fam: str, key: str, backend: str) -> str:
    out = []
    if key in GATED:
        out.append("gated repo: needs your Hugging Face token (HF_TOKEN or `hf auth login`)")
    return "; ".join(out)


def _other_name(fam: str, key: str) -> str:
    if fam == "glm":                                   # ("EXL3-3.05bpw" -> "GLM-5.3-Flash · EXL3 3.05 bpw")
        m = re.match(r"EXL3-([\d.]+)bpw$", key)
        return Q.SHOWN["glm"] + Q.SEP + (f"EXL3 {m.group(1)} bpw" if m else key)
    return Q.shown(fam, key)


_ccs_cache: list = []


def _nvidia_ccs(backend: str) -> list:
    """This PC's NVIDIA compute capabilities (70 = V100), asked once a minute; [] on AMD or when none is readable."""
    if backend in ("hip", "metal"):
        return []
    if _ccs_cache and time.time() - _ccs_cache[0] < 60:
        return _ccs_cache[1]
    found = []
    try:
        sys.path.insert(0, str(ROOT))
        import setup as SU                             # (the project's setup.py: nvidia-smi)
        found = [int(g["arch"]) for g in SU.gpus() or []]
    except Exception:  # noqa: BLE001 - no NVIDIA tools: the setup's PC check says
        found = []
    _ccs_cache[:] = [time.time(), found]
    return found


def catalog(root: Path, glm_sizes: dict | None = None, cfg: dict | None = None, ccs: list | None = None,
            other_sizes: dict | None = None) -> list:
    """The models the setup can download: [{"id" (the setup's --model id, or one of OTHER's), "family", "name" (as
    users see it), "group" ("maya": Maya's own quants and CYBER-FROST; "other": the other files the setup offers, by
    their makers), "by", "note", "installed" (a config here runs it), "size_gb" (None when the setup tells it at its
    start)}].  cfg: the running model's config (its backend picks the other files offered); ccs: the NVIDIA GPUs'
    compute capabilities (None: read)."""
    have = set()
    for p in models.config_paths(root):
        c = models.read(p)
        if isinstance(c.get("installer"), dict):
            have.add((Q.family_of(c), str(Q.config_key(c))))
    sizes = glm_sizes if glm_sizes is not None else _glm_sizes()
    out = []
    for fam, key, _tier, ident in Q.QUANTS:
        out.append({"id": ident, "family": fam, "name": Q.shown(fam, key), "group": "maya", "by": "Maya", "note": "",
                    "installed": (fam, key) in have, "size_gb": sizes.get(key)})
    out.append({"id": "cyber-frost", "family": "qwen", "name": Q.shown("qwen", "q5_k_m"), "group": "maya",
                "by": "peasantsmith", "note": "", "installed": ("qwen", "q5_k_m") in have, "size_gb": None})
    backend = (cfg or {}).get("backend") or "cuda"
    ccs = _nvidia_ccs(backend) if ccs is None else ccs
    extra = other_sizes if other_sizes is not None else (_other_sizes() if glm_sizes is None else {})
    for ident, fam, key, by in OTHER:
        if not _offered(fam, key, backend, ccs):
            continue
        out.append({"id": ident, "family": fam, "name": _other_name(fam, key), "group": "other", "by": by,
                    "note": _note(fam, key, backend), "installed": (fam, key) in have,
                    "size_gb": sizes.get(key) if fam == "glm" else extra.get(ident)})
    # GLM-5.3 (744B, glm-dsa): its downloads (maya.G53_MODELS) - NVIDIA from V100 on and AMD, not
    # on Metal; the id is the setup's --model name in lower case (Puller.start lowers what it is given); Maya's own
    # quant of it ("maya": GLM-5.3-Maya-XS) among Maya's, the others by their makers
    if backend != "metal" and (backend == "hip" or not ccs or max(ccs) >= G53_MIN_CC):
        for key, m in _g53_downloads().items():
            own = bool(m.get("maya"))
            out.append({"id": key.lower(), "family": "glm", "name": G53.LABEL + Q.SEP + Q.glm53_quant(key),
                        "group": "maya" if own else "other", "by": "Maya" if own else m["repo"].split("/")[0],
                        "note": "text only",
                        "installed": ("glm", key) in have, "size_gb": m.get("download_gb")})
    return out


G53_MIN_CC = 70                                      # (GLM-5.3's kernels: V100 on - the owner's scope, 2026-11-07)


def _g53_downloads() -> dict:
    """GLM-5.3's downloads (maya.G53_MODELS)."""
    try:
        sys.path.insert(0, str(ROOT))
        import maya                                    # (the setup's own table; nothing runs on import)
        return dict(maya.G53_MODELS)
    except Exception:  # noqa: BLE001 - a catalog entry is a nicety
        return {}


_other_gb: dict = {}


def _other_sizes() -> dict:
    """GB each of OTHER's downloads takes, as the setup's list says them (its manifests; Maya's n-gram tables in
    place of unsloth's table shard, with orcarouter's too); read once - an entry not readable is left out."""
    if _other_gb:
        return _other_gb
    try:
        sys.path.insert(0, str(ROOT))
        import maya                                    # (the setup's own tables; nothing runs on import)
    except Exception:  # noqa: BLE001 - a size is a nicety
        return {}
    for ident, fam, key, _by in OTHER:
        try:
            if fam == "deepseek" and key == "exl3":
                gb = 414                               # (the setup's own figure for it)
            elif fam == "deepseek" and key in maya.DS4_EXL3:
                gb = maya.ds4_model(key)["download_gb"]
            elif fam == "deepseek" and key in maya.DS4_ABL:
                gb = maya.ds4_gguf_model(key)["download_gb"] + maya.ds4_abl_engram_gb()
            elif fam == "deepseek":
                mk, t = maya.ds4_gguf_model(key), maya.ds4_own_table(key)
                gb = mk["download_gb"] - mk["files"].get(t, 0) / 1e9 + maya.ds4_abl_engram_gb() if t else mk["download_gb"]
            elif fam == "minimax":
                gb = maya.mm3_models()[0]["download_gb"]
            elif fam == "qwen":                        # (its files and the vision tower the setup fetches with them)
                vision = maya.qwen_vision_model()
                gb = maya.qwen_entry(key)["download_gb"] + (vision["download_gb"] if vision else 0)
            else:
                continue                               # (GLM's: the MODELS table, as Maya's own)
            _other_gb[ident] = round(float(gb), 1)
        except (Exception, SystemExit):  # noqa: BLE001 - a manifest not readable: no size shown
            continue
    return _other_gb


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
    other = next(((fam, key) for i, fam, key, _ in OTHER if i == model_id), None)
    if other is None:
        which = ["--model", model_id]
    elif other[0] == "glm":                            # (a GLM download by its name)
        which = ["--model", other[1]]
    elif other[0] == "qwen":                           # (its setup id: maya_quants.QWEN_UD_IDS)
        which = ["--model", model_id]
    else:                                              # (the model, its files by --ds4-quant / --mm3-quant)
        which = ["--model", Q.MODEL[other[0]], "--ds4-quant" if other[0] == "deepseek" else "--mm3-quant", other[1]]
    cmd = [sys.executable, str(ROOT / "maya.py"), "--setup", *which, "--download-model", "--yes",
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
        known = {c["id"] for c in catalog(self.root, {}, cfg, other_sizes={})}
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
