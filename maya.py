#!/usr/bin/env python3
"""Project Maya - set up and start GLM-5.3-Flash on your own GPU(s).
CUDA: Linux; Windows (experimental). HIP: experimental gfx1100/gfx1201/gfx1151, Linux and Windows, text only.

    ./maya.sh                 the first run sets everything up and starts the dashboard; later runs just start it
    ./maya.sh --setup         set up again (other GPUs, another context length, another model folder)
    ./maya.sh --check         only check this PC
    ./maya.sh --backend hip --gpu 0 --check    check an RX 7900 XT / XTX with system ROCm 7

On Windows START-MAYA.bat takes the same options.  Both make the private Python environment (.venv, the way Strata's
setup.sh does) and run this file.  It reuses Strata's installer (setup.py) for the PC checks,
pip, llama.cpp's source and resumable downloads.

In a terminal the setup runs on a screen of its own (tools/setup_tui.py): the steps, what runs now with a progress
bar, and the questions as menus - then Maya runs on the same screen, and every later start too: its dashboard (the
Monitor's numbers from the engine's GET /metrics), its server's output, Ctrl+C stops it.  Its library, Textual, comes
with Maya (third_party/wheels) and goes into .venv from there - nothing is downloaded.  --plain or a pipe: plain text,
and Maya in the terminal as before; --yes: the setup in plain text.

What the first run does (each step is skipped when it is already done):

  1. checks the PC: NVIDIA GPU(s) of compute capability 7.0+, driver, CUDA toolkit (nvcc), the C++ compiler (g++;
     on Windows Visual Studio 2022's Build Tools), CMake, RAM, CPU; HIP checks AMD gfx1100/gfx1201/gfx1151 and ROCm 7 instead
     (on Windows AMD's ROCm SDK wheels in .venv, offered when they are not there)
  2. asks: which GPUs (one, or several that split the layers), how much context, which model to download (Maya-S,
     Maya-S24, Maya-M or Maya-L)
  3. Python packages into .venv, llama.cpp's source at the pinned commit (it lists them and asks first)
  4. compiles the engine (`build/strata`, or `build-hip/strata`) for your GPU(s): 10-30 minutes, once
  5. the model: GGUF files you already have (--gguf-dir), or a download it shows you first - the exact commands
     and the size - and starts only after you answer y (or pass --download-model)
  6. builds the pack (the engine's index of the GGUF files) inside the model folder
  7. images: compiles the vision encoder (`build-vision/bin/strata-vision`) and fetches the model's vision files
     (1.1 GB, shown and asked first like the model; --no-vision and HIP skip it)
  8. writes maya-<model>.json and run-maya-<model>.sh (.bat on Windows) - with the settings tuned for this PC
     earlier, or it offers the tuning (./maya.sh --calibrate does it any time) - and starts the dashboard on
     http://127.0.0.1:8080

The tuning (tools/calibrate_glm.py, like Strata's --calibrate): decode speed measured with a few splits of the RAM-tier
experts between the CPU and the PCIe link, and with fewer CPU threads, in one engine run (~10-15 minutes, the model
loads first); a setting is kept when it is more than 3% faster than the engine's own choice.  The result goes into the
config's "env" (STRATA_GLM_PCIE_SHARE, STRATA_GLM_CPU_LANE) and into ~/.config/project-maya/calibration.json for this
PC, model and context, so a setup again keeps it.

Nothing is installed system-wide: a missing tool is reported with the command that installs it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
import setup as S  # noqa: E402  Strata's installer: PC checks, pip, llama.cpp, downloads (nothing runs on import)
from setup import ask, fail, ok, run, say, step, warn  # noqa: E402
import calibrate_glm as CAL  # noqa: E402  the CPU lane's tuning for this PC (./maya.sh --calibrate)

WIN = S.WIN
ME = "START-MAYA.bat" if WIN else "./maya.sh"      # how this is started, for the messages
UPDATE_EXIT = 75                                   # the server's exit after the dashboard updated Maya (serve/update.py)
ROOT = S.ROOT
BUILD = ROOT / "build"
EXE = BUILD / ("strata.exe" if WIN else "strata")
STAMP = BUILD / "MAYA-BUILD.json"                  # what the engine in build/ was compiled from and for
VBUILD = ROOT / "build-vision"
VEXE = VBUILD / "bin" / ("strata-vision.exe" if WIN else "strata-vision")  # the image encoder (llama.cpp's mtmd)
VSTAMP = VBUILD / "MAYA-BUILD.json"
MIN_CC = 70                                        # Volta (V100) and newer (the GLM path; see arch_setting)
MAX_GPUS = 16                                      # the layer split's parts at most (glm_model.hpp, kMaxParts)
HIP_ARCHS = ("gfx1100", "gfx1201", "gfx1151")        # Maya also supports Strix Halo (unified memory)
PY_PACKAGES = list(S.PY_PACKAGES)                  # (pillow: pictures in formats other than JPEG/PNG/BMP/GIF)
CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_CONTEXT = 32768
MODEL_NAME = "glm-5.3-flash"
SAMPLING = {"temperature": 1.0, "top_p": 0.95}     # the dashboard's and the API's defaults for requests that set none
EFFORT = "high"                                    # thinking level for requests that name none (GLM's High)
HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
# The models the installer can download: Project Maya's own quants, made from Z.ai's FP8 release (their model card
# has the measurements against it).
# Another glm5-next GGUF can be used with --gguf-dir (experimental).  "folder": the files' folder in the repo ("" =
# its top).  "file": the names on Hugging Face, with the quant label its file list groups (and adds up) them by -
# one label per model; "was": the names they had there before (v1.0.18 and earlier: no label; v1.0.19: Maya-S24 as
# IQ2_XXS, which Hugging Face added to Maya-S's) - a download under them is used as it is.  "sha256" per file name (the
# Hugging Face one): every download is verified; a list holds every version published (v1.0.28: the first shards' header
# names the architecture "glm5-next", as llama.cpp does - the data is the same, so a download of either is good).  "vision": the image encoder's files (the mmproj,
# made from the official vision tower, and the tokenizer it reads its markers with) - from "repo" / "revision" when
# it names them, else the model's own repo.
MODELS = {
    "Maya-S-v2-IQ2_XXS": {
        "about": "Maya-S, Project Maya's compact quant: error-feedback-rounded IQ2_XXS gate/up experts, "
                 "IQ2_S/IQ3_XXS down projections, Q6_K attention, the MTP draft block, made from Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S-v2-IQ2_XXS",
        "file": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 96.5,
        "sha256": {
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf":
                ["e56ef50efd71d81ddf6249e08bd2ef5af96ed348c2b7a004491d46168d0ecaa7",    # "glm5-next" (v1.0.28)
                 "a507f2b7b25e04624ee55c631d3b281caea7cfffc747e5247970cd5a7ea91b3f"],   # "glm5next", before
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00002-of-00003.gguf":
                "2d65d88a69f8dc124c8d24bd33b33161ddab218918b4253ad4f500aeede84df5",
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00003-of-00003.gguf":
                "a6a981c4fee7a53d78bdbd97d8f465d48ddf439cf938ff90617f395e0351b5a6"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-S24": {
        "about": "Maya-S24, Maya-S with Q4_K attention and shared experts: about 1.5 GB less that stays on the GPU, so "
                 "24 GB cards hold more experts (decode about 14% faster there, 11% on 32 GB), close to Maya-S in "
                 "quality",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S24",
        "file": "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-S24-IQ2_XXS-{i:05d}-of-{n:05d}.gguf",
                "GLM-5.3-Flash-Maya-S24-{i:05d}-of-{n:05d}.gguf"], "shards": 3, "download_gb": 94.7,
        "sha256": {
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00001-of-00003.gguf":
                ["8ebd5c74bcf65ece2a8699b7b6fd2b8d177f350ed252cee252fdd99b86a0138c",
                 "3dc347757686c1435eae36c4872f5c151191cc5063bc188ce699b97700aae076"],
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00002-of-00003.gguf":
                "4bd445da3a0128a9c5b32228c924e0a622aa4a132c7a8dc9a2c207a4beea8e34",
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00003-of-00003.gguf":
                "a6fd4e88007ac49b431c7d02be34e43ab7a09224d53a0c52782327613cb41917"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-M": {
        "about": "Maya-M, Project Maya's larger quant: error-feedback-rounded IQ2_S gate/up experts, IQ3_XXS down "
                 "projections (IQ3_S in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-M",
        "file": "GLM-5.3-Flash-Maya-M-IQ2_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-M-{i:05d}-of-{n:05d}.gguf"], "shards": 3, "download_gb": 116.0,
        "sha256": {
            "GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf":
                ["3f740d1b1022b4a0ddad50cd861bb071dd415bde94afa1292d0476641186c638",
                 "3ac0f066ec45af3432d59b33de49bdfb29156432b627240b35769c7d02cc6c02"],
            "GLM-5.3-Flash-Maya-M-IQ2_S-00002-of-00003.gguf":
                "285951d2afa0cd98285b03d0dc4aa68d83daf1a6f4a2594fe40b0cdd27ade485",
            "GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf":
                "ebf1ce713f71207747e10eeebed87597969d8b5e1d9dd817420d2c2f7ca51e0e"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-L": {
        "about": "Maya-L, Project Maya's largest quant: error-feedback-rounded IQ3_S gate/up experts, IQ4_XS down "
                 "projections (Q5_K in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-L",
        "file": "GLM-5.3-Flash-Maya-L-IQ3_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-L-{i:05d}-of-{n:05d}.gguf"], "shards": 4, "download_gb": 156.3,
        "sha256": {
            "GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf":
                ["c41aeaf3a3e0022150b7c355e7f4df6c92ba68c4794e3e827e02e56b54886b9e",
                 "5fc82a6c9af4c6898e8d45cf32964f9a7b10640caf51e82be735d2d50e9f3d45"],
            "GLM-5.3-Flash-Maya-L-IQ3_S-00002-of-00004.gguf":
                "351d59366afb7be7dfc3fb7af91281f7ccf00760b17dc45f82090325a7b34f64",
            "GLM-5.3-Flash-Maya-L-IQ3_S-00003-of-00004.gguf":
                "3685962fdfeaf130b74f98d3ce703ee33d818aa9991725932cb305b7c48db8af",
            "GLM-5.3-Flash-Maya-L-IQ3_S-00004-of-00004.gguf":
                "1af62cd72de6460c85369d988abbb790d6ec3a97507df51faf9bf32aa2b600d0"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
}
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
GLM_ARCHS = ("glm5-next", "glm5next")             # llama.cpp's spelling and unsloth's
PACK_FILES = ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/vocab.json", "tokenizer/merges.txt",
              "tokenizer/token_type.json", "tokenizer/chat_template.jinja")


# ------------------------------------------------------------------------------------------------ small helpers
def hf_url(repo: str, revision: str, folder: str, file: str) -> str:
    """A file's download URL on Hugging Face (folder "" = the repo's top)."""
    return HF.format(repo=repo, revision=revision, path=f"{folder}/{file}" if folder else file)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def configs() -> list:
    """The installed models' configs, the most recently used first - not the dashboard's settings the server keeps
    beside each (<config>.shared-settings.json), which a restart took for a config of the CUDA backend."""
    return sorted((p for p in ROOT.glob("maya-*.json") if not p.name.endswith(".shared-settings.json")),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def mem_gb() -> tuple:
    """(total, available) RAM in GiB - the engine sizes its RAM tier from MemAvailable (on Windows ullAvailPhys)."""
    if WIN:
        m = S._memory_status()
        return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024 / 2**30
    except (OSError, ValueError):
        pass
    return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)


def existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (where it would be created)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def rotational(path: Path) -> bool:
    """True when `path` is on a spinning disk (best effort: findmnt + lsblk; on Windows the drive's MediaType)."""
    if WIN:
        drive = existing(path).drive[:1]
        return drive.isalpha() and S.out(
            ["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Where-Object DeviceId -eq "
             f"(Get-Partition -DriveLetter {drive}).DiskNumber | Select-Object -ExpandProperty MediaType"]).strip() == "HDD"
    src = S.out(["findmnt", "-no", "SOURCE", "--target", str(existing(path))]).strip().split("[")[0]
    rota = S.out(["lsblk", "-ndo", "ROTA", src]).split() if src.startswith("/dev/") else []
    return bool(rota) and rota[0] == "1"


def tool_version(exe: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)", S.out([exe, "--version"]))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def venv_tool(name: str):
    p = Path(sys.executable).parent / (name + (".exe" if WIN else ""))
    return str(p) if p.exists() else None


def pick_cmake():
    """CMake 3.24 or newer: the one pip put into .venv first, then the system's."""
    for c in (venv_tool("cmake"), shutil.which("cmake")):
        if c and tool_version(c) >= (3, 24):
            return c
    return None


def choose(question: str, intro: list, options: list, default: int, yes: bool, outro: list = ()) -> int:
    """The answer's index among `options`, each (label, note or None, about or None): a menu on the setup's screen
    (tools/setup_tui.py; `about` shows under the highlighted one), else the numbered list and the number typed."""
    if S.UI is not None and not yes:
        pick = S.UI.choose(question, intro, options, default, outro)
        if pick is not None:
            return pick
    say()
    for line in intro:
        say("  " + line)
    for i, (label, note, _) in enumerate(options, 1):
        say(f"  {i}) {label}" + (f"   ({note})" if note else ""))
    for line in outro:
        say("     " + line)
    return int(ask(question, [str(i) for i in range(1, len(options) + 1)], str(default + 1), yes)) - 1


def gpu_label(g) -> str:
    if g.get("vendor") == "amd":
        return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB, {g['arch']})"
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def select_build_backend(backend: str) -> None:
    """Separate build folders and stamps prevent reusing a binary from the other backend."""
    global BUILD, EXE, STAMP
    BUILD = ROOT / ("build-hip" if backend == "hip" else "build")
    EXE = BUILD / ("strata.exe" if WIN else "strata")
    STAMP = BUILD / "MAYA-BUILD.json"


def hip_unified_memory(g) -> bool:
    # gfx1151 is Strix Halo / Gorgon Halo, even when KFD/sysfs has no product name or reports
    # only a tiny firmware VRAM carve-out. This is independent of vram_gb.
    return g.get("arch") == "gfx1151" or bool(g.get("integrated")) or bool(
        re.search(r"Ryzen AI Max|Radeon(?:\s*\(TM\))?\s+80[4-6]\dS", g.get("name", ""), re.I))


def hip_clang(root: Path) -> Path | None:
    """ROCm's clang++: <root>/llvm/bin (a system ROCm on Linux), <root>/lib/llvm/bin (TheRock's wheels), or <root>/bin
    (AMD's HIP SDK for Windows)."""
    exe = "clang++.exe" if WIN else "clang++"
    return next((p for p in (root / "llvm" / "bin" / exe, root / "lib" / "llvm" / "bin" / exe, root / "bin" / exe)
                 if p.exists()), None)


def rocm_sdk() -> str | None:
    """TheRock's rocm-sdk (AMD's ROCm wheels for Windows): ROCM_VENV's, this .venv's, else the one on PATH."""
    venv = os.environ.get("ROCM_VENV")
    for c in (venv and str(Path(venv) / "Scripts" / "rocm-sdk.exe"), venv_tool("rocm-sdk"), shutil.which("rocm-sdk")):
        if c and Path(c).exists():
            return c
    return None


def windows_rocm() -> tuple:
    """(ROCm's root, the folder of its DLLs) on Windows, or (None, None): ROCM_PATH, else TheRock's wheels, else AMD's
    HIP SDK for Windows (HIP_PATH, which its installer sets, else the newest in C:\\Program Files\\AMD\\ROCm)."""
    if os.environ.get("ROCM_PATH"):
        root = Path(os.environ["ROCM_PATH"]).resolve()
        return root, root / "bin"
    sdk = rocm_sdk()
    if sdk is not None:
        # the first `rocm-sdk init` unpacks the compiler, the headers and the libraries (about 3 GB, minutes) and links
        # the device wheels in; later ones only check. Before any `rocm-sdk path`: that would unpack it too, unseen and
        # within S.out's minute, and start over on every run (as llama.cpp's Windows CI does: init, then path)
        say("  AMD's ROCm SDK: rocm-sdk init (the first time it unpacks about 3 GB) ...")
        run([sdk, "init"], check=False)
        path = lambda what: (S.out([sdk, "path", what]).strip().splitlines() or [""])[-1].strip()
        root = path("--root")
        if root:
            return Path(root), Path(path("--bin") or Path(root) / "bin")
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"
    sdks = sorted(base.glob("[0-9]*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)], reverse=True)
    for root in ([Path(os.environ["HIP_PATH"])] if os.environ.get("HIP_PATH") else []) + sdks:
        if hip_clang(root) is not None:
            return root.resolve(), root.resolve() / "bin"
    return None, None


def rocm_devices_missing(archs) -> list:
    """Of these architectures, those whose library kernels (TheRock's rocm-sdk-device-<arch> wheel) are not in this
    .venv; [] for a ROCm SDK elsewhere (ROCM_PATH, ROCM_VENV, PATH: not checked)."""
    if os.environ.get("ROCM_PATH") or rocm_sdk() is None or rocm_sdk() != venv_tool("rocm-sdk"):
        return []
    from importlib import invalidate_caches, metadata
    invalidate_caches()
    missing = []
    for arch in dict.fromkeys(archs):
        try:
            metadata.version(f"rocm-sdk-device-{arch}")
        except metadata.PackageNotFoundError:
            missing.append(arch)
    return missing


def hip_info(bindir) -> str | None:
    """ROCm's hipInfo (on Windows: lists the GPUs as HIP numbers them, with their architecture)."""
    p = Path(bindir) / "hipInfo.exe" if bindir else None
    return str(p) if p and p.exists() else shutil.which("hipInfo")


THEROCK_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"   # AMD's ROCm for Windows: TheRock's pip wheels
THEROCK_VERSION = "7.14.1"                                     # 7.14: the first with Gorgon Halo (Ryzen AI Max 400)


def rocm_pip(archs) -> list:
    """pip's command that puts AMD's ROCm SDK for these GPUs into this Python (.venv): the compiler, the libraries and
    their kernels for each architecture."""
    devices = ",".join(f"device-{x}" for x in sorted(set(archs)))
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--index-url", THEROCK_INDEX,
            f"rocm[libraries,devel,{devices}]=={THEROCK_VERSION}"]


def install_rocm(a, chosen) -> None:
    """Windows: AMD's ROCm SDK with the kernels for these GPUs into Maya's .venv (asked first; nothing goes outside
    this folder), else the command."""
    archs = sorted({g["arch"] for g in chosen})
    cmd = rocm_pip(archs)
    what = f"AMD's ROCm SDK for Windows with the kernels for {', '.join(archs)}"
    if getattr(a, "check", False) or sys.prefix == sys.base_prefix:
        fail(f"{what} is not installed", f"install it into Maya's Python environment: {cmdline(cmd)}"
             f" - then run {ME} again (or set ROCM_PATH / ROCM_VENV to a ROCm 7 that has one)")
    if ask(f"Install {what} ({THEROCK_VERSION}) into .venv now (pip, from repo.amd.com)?",
           ["y", "n"], "y", getattr(a, "yes", False)) != "y":
        fail("Maya's HIP engine is compiled with AMD's ROCm SDK", f"install it with: {cmdline(cmd)}")
    run(cmd)
    if windows_rocm()[0] is None:
        fail("pip installed AMD's ROCm SDK, but rocm-sdk does not answer", f"run {cmdline(cmd)} again and check its "
             "messages")


def check_hip_pc(a, installed=False) -> dict:
    if not WIN and (not sys.platform.startswith("linux") or S.is_wsl()):
        fail("Maya's experimental HIP backend runs on native Linux or Windows")
    if WIN and S.find_vcvars(cuda=False) is None:      # (before ROCm's download: ROCm's clang needs its headers)
        fail("Visual Studio Build Tools (2022 or 2026) are needed: ROCm's compiler uses their headers and libraries",
             "winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add "
             "Microsoft.VisualStudio.Workload.VCTools --includeRecommended\" - or "
             "https://visualstudio.microsoft.com/visual-cpp-build-tools/ with 'Desktop development with C++'")
    root, bindir = windows_rocm() if WIN else (None, None)
    hipinfo = hip_info(bindir) if WIN else None
    found = S.amd_gpus(hipinfo) if WIN else S.amd_gpus()
    usable = [g for g in found if g["arch"] in HIP_ARCHS]
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if g in usable else "not supported by Maya's HIP build"))
    if not usable:
        fail("no supported AMD GPU found", "this port targets RX 7900 XT / XTX (gfx1100), RX 9070 / AI PRO R9700 "
             "(gfx1201), and Strix Halo / Gorgon Halo, Radeon 8050S / 8060S / 8065S (gfx1151)")
    if WIN and hipinfo and all(g.get("source") == "registry" for g in found):
        warn(f"ROCm's hipInfo sees no AMD GPU (the list above is Windows'): the engine cannot start until it does - "
             f"update AMD's driver (AMD Software: Adrenalin Edition, https://www.amd.com/en/support), restart, and "
             f"check with {hipinfo}")
    elif WIN and len(found) > 1 and any(g.get("source") == "registry" for g in found):
        warn("these GPU numbers are Windows' order (ROCm's hipInfo was not found): HIP's own can differ")
    if a.gpus:
        # two cards split the layers (each caches the experts of its own half); the larger card goes first, as it
        # takes the bigger first half - the order measured on an R9700 + RX 7900 XT
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes two GPU numbers as --check shows them, e.g. --gpus 0,1, not {a.gpus!r}")
        if len(want) != 2 or want[0] == want[1]:
            fail("Maya's HIP port runs on one GPU or splits the model across two: --gpu 0, or --gpus 0,1")
        picked = [next((g for g in usable if g["index"] == i), None) for i in want]
        for i, g in zip(want, picked):
            if g is None:
                fail(f"GPU {i} is not a supported AMD card")
        apu = next((g for g in picked if hip_unified_memory(g)), None)
        if apu is not None:
            fail(f"GPU {apu['index']} ({apu['name']}) is an APU with unified memory, which stays single-GPU",
                 "use --gpu N for it, or pick two discrete cards with --gpus")
        chosen = sorted(picked, key=lambda g: -g["vram_gb"])
    else:
        one = next((g for g in usable if g["index"] == a.gpu), None) if a.gpu is not None else max(
            usable, key=lambda g: g["vram_gb"])
        if one is None:
            fail(f"GPU {a.gpu} is not a supported AMD card")
        chosen = [one]
    if WIN:
        # no ROCm SDK, or Maya's without the library kernels (rocm-sdk-device-<arch>) for a chosen GPU: offered once,
        # then checked again with ROCm there (its hipInfo numbers the GPUs as HIP does)
        missing = sorted({g["arch"] for g in chosen}) if root is None else rocm_devices_missing(g["arch"] for g in chosen)
        if missing and not installed and (root is None or not getattr(a, "check", False)):   # (--check: says so below)
            install_rocm(a, chosen)
            return check_hip_pc(a, installed=True)
        if root is None:
            fail("AMD's ROCm SDK for Windows was not found after its install", f"run {ME} again")
        if missing:
            warn(f"the ROCm SDK still has no library kernels for {', '.join(missing)} (rocm-sdk-device-...): hipBLAS "
                 f"fails on that GPU - {cmdline(rocm_pip(g['arch'] for g in chosen))}")
        if hip_clang(root) is None or not (list(root.glob("lib/hipblas*.lib")) or list(bindir.glob("hipblas*.dll"))):
            fail(f"ROCm's HIP compiler and hipBLAS are not in {root}",
                 f"install AMD's ROCm SDK into Maya's .venv: {cmdline(rocm_pip(g['arch'] for g in chosen))} - or set "
                 "ROCM_PATH to the root of a ROCm 7 for Windows")
        hipcc = next((p for p in (bindir / "hipcc.exe", root / "bin" / "hipcc.exe") if p.exists()), None)
        if hipcc is not None and tool_version(str(hipcc)) < (7, 0):
            fail("Maya's HIP port requires ROCm 7 or newer")
    else:
        root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
        if not os.environ.get("ROCM_PATH") and not root.exists():
            # versioned installs without the /opt/rocm link (e.g. /opt/rocm-7.2.2): the newest one
            versions = sorted(Path("/opt").glob("rocm-[0-9]*"),
                              key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])
            if versions:
                root = versions[-1]
        root = root.resolve()
        if not (root / "llvm/bin/clang++").exists() or not list((root / "lib").glob("libhipblas.so*")):
            fail("ROCm's HIP compiler and hipBLAS are required", "install ROCm 7, or set ROCM_PATH to its root")
        if tool_version(str(root / "bin/hipcc")) < (7, 0):
            fail("Maya's HIP port requires ROCm 7 or newer")
        if not shutil.which("c++"):
            fail("a C++ compiler is needed", "Ubuntu/Debian: sudo apt install build-essential")
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        fail(f"the CPU ({cpu}) needs AVX2 for the expert lane")
    total, avail = mem_gb()
    ok(f"using {' + '.join(gpu_label(g) for g in chosen)}; experimental HIP, "
       f"{'two GPUs (layer split)' if len(chosen) == 2 else 'one GPU'}, text only")
    ok(f"ROCm: {root}; CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'})")
    ok(f"RAM: {total:.0f} GB, {avail:.0f} GB available now")
    if any(hip_unified_memory(g) for g in chosen) and WIN:
        # Windows gives the APU's GPU a fixed carve-out, which it does not count as RAM: the engine sizes the pool
        # from it as on a discrete card (src/core/glm_fast_path.cu, fast_setup)
        vram = sum(g["vram_gb"] for g in chosen)
        ok(f"APU: {vram:.0f} GB of the memory is the GPU's (Windows' share for it): the engine fills it with experts "
           "and keeps the next ones in RAM")
        if vram < total:
            warn(f"the GPU has {vram:.0f} GB and Windows {total:.0f} GB: set Variable Graphics Memory higher (AMD "
                 "Software: Performance > Tuning, or the iGPU memory size in the BIOS), e.g. 96 GB on a 128 GB PC, "
                 "and restart - the experts on the GPU are the fastest")
    elif any(hip_unified_memory(g) for g in chosen):
        ok("APU / unified memory: the engine sizes the GPU expert pool from available system RAM")
    select_build_backend("hip")
    return {"backend": "hip", "gpus": chosen, "archs": list(HIP_ARCHS), "rocm": str(root),
            **({"rocm_bin": str(bindir)} if WIN else {})}


def nvcc_range(archs) -> tuple:
    """The CUDA toolkit versions that can build for these GPUs: (lowest, first too new or None)."""
    lo = (12, 8) if max(archs) >= 100 else (12, 0)  # Blackwell needs 12.8
    hi = (13, 0) if min(archs) < 75 else None       # CUDA 13 no longer compiles for Volta (sm_70)
    return lo, hi


def find_nvcc(archs, given=None) -> tuple:
    """(nvcc, version) for these GPUs: --nvcc, else the newest toolkit that can build for them (Strata's find_nvcc
    takes the newest of all, but CUDA 13 cannot build for Volta), else the newest found (reported as unfit), else
    (None, None)."""
    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if given:
        return given, version(given)
    return pick_nvcc([(c, version(c)) for c in dict.fromkeys(c for c in nvcc_candidates() if c and Path(c).exists())],
                     archs)


def nvcc_candidates() -> list:
    """Every place a CUDA toolkit's nvcc may be (some do not exist): PATH, CUDA_PATH, the default install folders."""
    exe = "nvcc.exe" if WIN else "nvcc"
    cands = [shutil.which("nvcc"), os.environ.get("CUDA_PATH") and str(Path(os.environ["CUDA_PATH"]) / "bin" / exe)]
    if WIN:
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        return cands + [str(p / "bin" / exe) for p in sorted(base.glob("v*"))]
    cands += [str(p / "bin" / exe) for p in sorted(Path("/usr/local").glob("cuda*"))]
    return cands + [str(p / "bin" / exe) for p in sorted(Path("/opt").glob("cuda*"))] + ["/usr/bin/nvcc"]


def pick_nvcc(found, archs) -> tuple:
    """Of [(nvcc, version)], the newest that can build for `archs`, else the newest, else (None, None)."""
    found = sorted([f for f in found if f[1]], key=lambda f: f[1], reverse=True)
    lo, hi = nvcc_range(archs)
    fit = [f for f in found if f[1] >= lo and (hi is None or f[1] < hi)]
    return (fit or found or [(None, None)])[0]


def toolkit_hint(archs) -> str:
    if WIN:
        return ("NVIDIA's CUDA Toolkit 12.8 for Windows (the driver can stay as it is): "
                "https://developer.nvidia.com/cuda-12-8-0-download-archive" +
                (" - Volta (V100) needs 12.x, CUDA 13 dropped it" if min(archs) < 75 else ""))
    if min(archs) < 75:
        return ("Volta (V100) needs CUDA 12.x - Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), "
                "or NVIDIA's cuda-toolkit-12-8 package: https://developer.nvidia.com/cuda-12-8-0-download-archive")
    if max(archs) >= 100:
        return "NVIDIA's cuda-toolkit-12-8 (or newer) package: https://developer.nvidia.com/cuda-downloads"
    return ("Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), or NVIDIA's packages: "
            "https://developer.nvidia.com/cuda-downloads")


def cuda_lib_dirs(nvcc: str) -> list:
    """The toolkit's library folders for the engine's LD_LIBRARY_PATH (none for a distribution's /usr/bin/nvcc); on
    Windows its DLL folders (bin, and bin\\x64 in CUDA 13) for the engine's PATH."""
    if WIN:
        b = Path(nvcc).parent
        return [str(p) for p in (b, b / "x64") if p.is_dir()]
    base = Path(nvcc).resolve().parent.parent
    if base == Path("/usr"):
        return []
    return [str(p) for p in (base / "lib64", base / "targets" / "x86_64-linux" / "lib") if p.is_dir()]


# ------------------------------------------------------------------------------------------------ 1. the PC
def choose_gpus(a, found) -> list:
    """The GPUs the model runs on: --gpu / --gpus, else asked when several can share it (all of them recommended)."""
    byid = {g["index"]: g for g in found}
    usable = [g for g in found if int(g["arch"]) >= MIN_CC]
    if a.gpus or a.gpu is not None:
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()] if a.gpus else [a.gpu]
        except ValueError:
            fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,1, not {a.gpus!r}")
        if not 1 <= len(want) <= MAX_GPUS or len(set(want)) != len(want):
            fail(f"Maya runs on one GPU or splits the model's layers across up to {MAX_GPUS}: --gpu 0, or "
                 "--gpus 0,1,2,3 (each number once)")
        for i in want:
            if i not in byid or int(byid[i]["arch"]) < MIN_CC:
                fail(f"GPU {i} cannot be used" + ("" if i in byid else " (not found)"),
                     "use one of: " + ", ".join(str(g["index"]) for g in usable))
        return [byid[i] for i in want]
    if not usable:
        fail("none of the GPUs can run Maya", "it needs an NVIDIA GPU of compute capability 7.0 or newer (V100 and newer)")
    best = sorted(usable, key=lambda g: (-round(g["vram_gb"]), g["index"]))
    if len(best) == 1:
        return best
    opts = ([best[:MAX_GPUS]] if len(best) > 2 else []) + [best[:2], best[:1]]
    labels = []
    for gs in opts:
        if len(gs) > 2:
            labels.append(f"all {len(gs)} together: GPUs " + ", ".join(str(g["index"]) for g in gs) +
                          f" ({sum(g['vram_gb'] for g in gs):.0f} GB of VRAM)")
        elif len(gs) == 2:
            labels.append(f"{gpu_label(gs[0])} + {gpu_label(gs[1])} together")
        else:
            labels.append(f"{gpu_label(gs[0])} only")
    pick = choose("Which GPUs?",
                  ["Maya can run on one GPU, or split the model's layers across several: each then caches the experts "
                   "of its", "own layers, so together they hold more of them (V100s: ~50 tok/s benchmark on two, ~24 "
                   "on one)."],
                  [(label, "recommended" if i == 0 else None, None) for i, label in enumerate(labels)], 0,
                  a.yes or a.check,
                  ["(--gpus picks any others, e.g. --gpus 0,2,5; a model with a draft block drafts tokens on two GPUs "
                   "or more)"] if len(best) > 2 and S.UI is None else [])   # (the plain setup's; not on the screen)
    return opts[pick]


def check_pc(a) -> dict:
    step(1, "checking this PC")
    if a.backend == "hip":
        return check_hip_pc(a)
    if not (WIN or sys.platform.startswith("linux")):
        fail("Project Maya runs on Linux and Windows", "the engine needs an NVIDIA GPU and CUDA")
    if WIN:
        warn("Windows support is new (experimental): Maya is developed and measured on Linux - tell us how it runs")
    if S.is_wsl():
        warn("this is WSL2, which Maya does not support: Strata measured that WSL2's driver pins only about 1 GB of "
             "RAM for the GPU, and Maya's RAM tier pins tens of GB. Use a native Linux install.")
    found = S.gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             ("install the NVIDIA driver (https://www.nvidia.com/drivers)" if WIN else
              "install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install)") + f", restart, and run {ME} again")
    say("  NVIDIA GPUs:")
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if int(g["arch"]) >= MIN_CC else
                                       "too old (Maya needs compute capability 7.0 or newer)"))
    chosen = choose_gpus(a, found)
    archs = sorted({int(g["arch"]) for g in chosen})
    vram = sum(g["vram_gb"] for g in chosen)
    ok("using " + " + ".join(gpu_label(g) for g in chosen) +
       (f": the model's layers are split across the {len(chosen)}" if len(chosen) > 1 else ""))
    if vram < 23:
        say(f"  {vram:.0f} GB of VRAM in total: the engine fills it with the most-used experts and serves the rest from "
            "RAM and the SSD - give it a try (more VRAM is faster; tell us your speed)")
    problems = []                                      # (what is wrong, how to fix it): all of them at once

    nvcc, nv = find_nvcc(archs, a.nvcc)
    lo, hi = nvcc_range(archs)
    want = f"CUDA {lo[0]}.{lo[1]} or newer" + (f" but older than {hi[0]}.0 (CUDA 13 dropped Volta)" if hi else "")
    if nvcc is None:
        problems.append((f"the CUDA toolkit (nvcc) is not installed; these GPUs need {want}", toolkit_hint(archs)))
    elif nv is None:
        problems.append((f"{nvcc} does not say its version (nvcc --version)", "pass the toolkit's nvcc with --nvcc"))
    elif nv < lo or (hi is not None and nv >= hi):
        problems.append((f"nvcc {nv[0]}.{nv[1]} ({nvcc}) cannot build for these GPUs; they need {want}",
                         toolkit_hint(archs) + " (toolkits can be installed side by side: the newest that fits is "
                                                "used, or pass one with --nvcc)"))
    else:
        ok(f"CUDA toolkit {nv[0]}.{nv[1]}: {nvcc}")

    drv = min(S.driver_major(g) for g in chosen)
    need_drv = 580 if nv and nv >= (13, 0) else 525
    if drv < need_drv:
        problems.append((f"the NVIDIA driver {chosen[0]['driver']} is too old: {need_drv} or newer is needed",
                         ("" if WIN else "Ubuntu: sudo ubuntu-drivers install, then restart; or ") +
                         "https://www.nvidia.com/drivers"))
    else:
        ok(f"NVIDIA driver {chosen[0]['driver']}")

    if WIN:
        vcvars = S.find_vcvars()
        if vcvars is None:
            problems.append(("the C++ compiler (Visual Studio 2022 Build Tools, 'Desktop development with C++') is "
                             "not installed", "https://visualstudio.microsoft.com/visual-cpp-build-tools/ - in the "
                             "installer tick 'Desktop development with C++' (CUDA 12 needs 2019 or 2022, not 2026)"))
        else:
            ok(f"C++ compiler: Visual Studio Build Tools ({vcvars})")
    elif a.host_compiler:
        if shutil.which(a.host_compiler) is None:
            problems.append((f"--host-compiler {a.host_compiler} is not installed",
                             f"Ubuntu/Debian: sudo apt-get install -y {Path(a.host_compiler).name}"))
        else:
            ok(f"C++ compiler for nvcc: {a.host_compiler}")
    elif shutil.which("g++") is None:
        problems.append(("the C++ compiler (g++) is not installed", "Ubuntu/Debian: sudo apt-get install -y build-essential"))
    else:
        ok("C++ compiler: " + (S.out(["g++", "--version"]).splitlines() or ["g++"])[0])

    sc = pick_cmake() or shutil.which("cmake")
    scv = tool_version(sc) if sc else (0, 0)
    if scv >= (3, 24):
        ok(f"CMake {scv[0]}.{scv[1]}")
    else:
        ok((f"CMake {scv[0]}.{scv[1]} is older than 3.24" if sc else "no CMake on PATH") +
           ": a current one goes into .venv in step 3 (pip)")
    ok(f"Python {sys.version.split()[0]} ({sys.prefix})")

    total, avail = mem_gb()
    msg = f"RAM: {total:.0f} GB, {avail:.0f} GB available now"
    ok(msg + ("" if total >= 60 else " - it runs with 32 GB; more RAM keeps more experts close and is faster"))
    pf = S.page_file_gb()
    need_pf = int(vram + 8.999)   # the card's memory and the pinned RAM tier are both charged to RAM + page file
    if pf is not None and pf < need_pf:
        warn(f"Windows' page file is {pf:.0f} GB; set it to at least {need_pf} GB. Under Windows every allocation on "
             f"the graphics card ({vram:.0f} GB here) and the engine's pinned RAM tier are both charged to RAM + page "
             f"file, so with {pf:.0f} GB the RAM tier stays about {need_pf - pf:.0f} GB short of your free RAM and "
             "those experts are read from the SSD (issue #20: 2.4-4.1 -> 8.4-11 tokens/s on an RTX 5090). \"System "
             f"managed\" can stay small (8 GB on a 128 GB PC): set a custom size of {need_pf} GB or more - System > "
             "About > Advanced system settings > Performance > Advanced > Virtual memory - and restart")
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        warn(f"the CPU ({cpu}) has no AVX2: the engine runs, its CPU expert lane on ggml's own kernels (slower; "
             "on 2x Xeon E5-2660 v2 it still added ~15% to 4-GPU decode). STRATA_GLM_CPU_LANE=0 turns it off")
    else:
        ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'}, {os.cpu_count()} threads)")

    if problems:
        say()
        for what, how in problems:
            say(f"  [X]  {what}")
            say(f"       {how}")
        fail("something Maya needs is missing (above)",
             f"install it and run {ME} again - this script installs nothing system-wide")
    select_build_backend("cuda")
    return {"backend": "cuda", "gpus": chosen, "archs": archs, "nvcc": nvcc,
            "vcvars": str(S.find_vcvars()) if WIN else None}


# ------------------------------------------------------------------------------------------------ 2. choices
def choose_context(a, prev_ctx) -> int:
    if a.context:
        return a.context
    default = prev_ctx if prev_ctx in CONTEXTS else DEFAULT_CONTEXT
    pick = choose("Context?", ["Context length = how much text the model sees at once (the chat, files, tool output). "
                               "A longer one takes", "VRAM from the expert cache, so answers get a little slower:"],
                  [(f"{c // 1024}K tokens", "recommended" if c == DEFAULT_CONTEXT else None, None) for c in CONTEXTS],
                  CONTEXTS.index(default), a.yes)
    return CONTEXTS[pick]


def shard_names(m: dict, pattern: str | None = None) -> list:
    return [(pattern or m["file"]).format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)]


def old_names(m: dict) -> list:
    """The names a model's files had on Hugging Face before ("was", newest first), each as its list of shards."""
    return [shard_names(m, p) for p in m.get("was") or []]


def local_shards(m: dict, d: Path) -> list:
    """The model's files in d: under an earlier name when any of its files is there (a download started or finished
    under it - its run config points at them), else under the name on Hugging Face."""
    for names in old_names(m):
        if any((d / n).exists() for n in names):
            return [d / n for n in names]
    return [d / n for n in shard_names(m)]


def hf_name(m: dict, local: Path) -> str:
    """A local file's name on Hugging Face (a file under an earlier name downloads from, and checks against, it)."""
    for names in old_names(m):
        if local.name in names:
            return shard_names(m)[names.index(local.name)]
    return local.name


def download_dir(models: Path, quant: str) -> Path:
    """A download's folder: <models folder>/<quant> (--models-dir).  Files already there are used where they are:
    in that folder, in <models folder>/glm-5.3-flash-<quant> (an earlier version's downloads), or straight in the
    models folder - under their names on Hugging Face or an earlier one."""
    m = MODELS[quant]
    d = models / quant
    for c in (d, models / f"glm-5.3-flash-{quant}".lower(), models):
        for names in [shard_names(m)] + old_names(m):
            if all((c / n).exists() for n in names):
                return c
    return d


def choose_model(a, models: Path, inst: dict) -> tuple:
    """("download", one of MODELS) or ("local", the model's first .gguf file): --gguf-dir, --model, else asked among
    the downloads - the earlier download is the default."""
    if a.gguf_dir:
        first, why = local_first(Path(a.gguf_dir))
        if first is None:
            fail(why, "--gguf-dir takes the folder that holds the GLM-5.3-Flash .gguf files, or the (first) .gguf file")
        return "local", first
    if a.model:
        return "download", a.model
    opts = list(MODELS)
    # cards of 24 GB or less: Maya-S24 (its 4-bit attention leaves ~1.5 GB more of the card for experts - decode about
    # 14% faster there), else Maya-S
    try:
        cards = [g for g in S.gpus() if g.get("vram_gb")]
    except Exception:  # noqa: BLE001 - no NVIDIA tools (AMD): no card-based recommendation
        cards = []
    rec = "Maya-S24" if cards and all(g["vram_gb"] <= 24.5 for g in cards) and "Maya-S24" in MODELS else opts[0]
    default = inst["quant"] if inst.get("quant") in MODELS else rec
    rows = []
    for q in opts:
        m, d = MODELS[q], download_dir(models, q)
        have = all(p.exists() for p in local_shards(m, d))
        rows.append((f"{q}: " + (f"downloaded, in {d}" if have else f"download {m['download_gb']:.1f} GB from Hugging "
                                                                     "Face"),
                     ("recommended for cards of 24 GB or less" if q == "Maya-S24" else "recommended") if q == rec
                     else None, m["about"]))
    pick = choose("Model?", ["The model to download (GLM-5.3-Flash GGUF files you already have: --gguf-dir <file or "
                             "folder>):"], rows, opts.index(default), a.yes)
    return "download", opts[pick]


# ------------------------------------------------------------------------------------------------ 3. tools
def tools_step(a) -> Path:
    """The Python packages (into .venv) and llama.cpp's source at the pinned commit - listed, and asked, first."""
    step(3, "Python packages and llama.cpp's source")
    stamp = Path(sys.prefix) / ".strata-pip.json"          # S.pip_install's record of what it installed
    have = read_json(stamp) if stamp.exists() else []
    need = [p for p in PY_PACKAGES if p not in (have if isinstance(have, list) else [])]
    llama = Path(a.llama_dir).expanduser().resolve() if a.llama_dir else ROOT / "third_party" / "llama.cpp"
    llama_ok = (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir()
    if a.llama_dir and not llama_ok:
        fail(f"{llama} is not a llama.cpp checkout (its ggml/ and gguf-py/ folders are missing)")
    downloads = []
    if need:
        downloads.append("Python packages from PyPI into .venv: " + ", ".join(need) + " (roughly 50-100 MB)")
    if not llama_ok:
        downloads.append(f"llama.cpp's source at the pinned commit {S.LLAMA_CPP_COMMIT[:10]} from GitHub, "
                         f"{S.LLAMA_CPP_ZIP} (tens of MB): ggml for the engine build, gguf-py for the pack builder")
    if downloads:
        say("  This step downloads:")
        for d in downloads:
            say("    - " + d)
        if ask("  Download them now?", ["y", "n"], "y", a.yes) != "y":
            fail("nothing was downloaded", f"run {ME} again when you are ready (or pass --llama-dir with a "
                                           f"llama.cpp checkout at commit {S.LLAMA_CPP_COMMIT[:10]})")
    S.pip_install(PY_PACKAGES, ", ".join(PY_PACKAGES))
    if not llama_ok:
        llama = S.get_llama_cpp()
    ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:10]}: {llama}")
    return llama


# ------------------------------------------------------------------------------------------------ 4. the engine
def cached_generator():
    """The CMake generator build/ was configured with before (another one cannot be used there), or None."""
    m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (BUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (BUILD / "CMakeCache.txt").exists() else None
    return m.group(1).strip() if m else None


def arch_setting(archs) -> str:
    """CMAKE_CUDA_ARCHITECTURES for these GPUs.  CMakeLists.txt refuses an explicit arch below 75 (Strata's own
    qwen4exp floor), but the GLM path is developed and measured on Volta V100s (sm_70) - so for those the build
    asks CMake for this machine's own GPUs ("native", which the guard leaves to CMake), with CUDA_VISIBLE_DEVICES
    limited to the chosen cards.  RELEASE-CHECKLIST.md has the one-line CMake fix that makes this unnecessary."""
    return ";".join(str(x) for x in archs) if min(archs) >= 75 else "native"


def compile_engine(archs, gpu_ids, nvcc, host_compiler, llama: Path, src: str, soft=False) -> dict | None:
    """cmake configure + build of the `strata` target in build/; the exact commands are printed as they run.
    soft: a failure warns and returns None (the engine already there keeps working) instead of stopping."""
    def stop(what):
        msg = f"the engine build stopped while {what} (the reason is above)"
        hint = ("common causes: 'unsupported Microsoft Visual Studio version' - CUDA 12 needs Visual Studio 2019 or "
                "2022 (its Build Tools are enough);" if WIN else
                "common causes: 'unsupported GNU version' - install an older g++ your CUDA accepts (e.g. g++-13) and run "
                "again (it is found by itself), or pass --host-compiler g++-13;") + (
                "\n       'Unsupported gpu architecture compute_70' - CUDA 13 cannot build for Volta, "
                "install CUDA 12.x;\n       the compiler killed (out of memory) - close programs and run it again "
                "(it continues)")
        if soft:
            warn(msg + "; starting the engine compiled before")
            return None
        fail(msg, hint)

    cmake = pick_cmake()
    if cmake is None:
        return stop(f"looking for CMake 3.24+ (pip installs one into .venv: run {ME} --setup)")
    gen = cached_generator()
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and gen in (None, "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    cuda_archs = arch_setting(archs)
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"]
    if host_compiler:
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(host_compiler) or host_compiler}")
    # "native" is what CMake sees: only the chosen cards, numbered as nvidia-smi numbers them
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(str(i) for i in gpu_ids))
    # nvcc takes 2-4 GB per job on the big kernels: half the threads, and no more jobs than RAM / 4 GB
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    build = [cmake, "--build", str(BUILD), "--target", "strata", "-j", str(jobs)]
    say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) +
        " (10-30 minutes the first time, a few minutes after an update) ...")
    if cuda_archs == "native":
        say("  (CMakeLists.txt refuses an explicit sm_70 - Strata's own floor is sm_75 - so CMake is asked for this")
        say(f"  machine's GPUs instead: CMAKE_CUDA_ARCHITECTURES=native with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    stopped = cmake_steps(conf, build, env, "build-maya.bat")
    if stopped:
        return stop(stopped)
    meta = {"src": src, "archs": archs, "gpu_ids": list(gpu_ids), "nvcc": nvcc, "host_compiler": host_compiler,
            "llama": str(llama), "lib_dirs": cuda_lib_dirs(nvcc), "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def cmdline(cmd) -> str:
    """One command as a line of a Windows .bat."""
    return " ".join(f'"{x}"' if re.search(r'[\s;&|<>^()]', str(x)) else str(x) for x in cmd)


def shell_join(cmd) -> str:
    """One command as it is typed here: a POSIX shell, or Windows' cmd."""
    return cmdline(cmd) if WIN else shlex.join(cmd)


def cmake_steps(conf, build, env, bat_name: str, vcvars=None) -> str | None:
    """CMake's configure, then its build (once more when that stops: it continues where it stopped).  None when both
    worked, else what stopped.  On Windows both run in a .bat that first calls Visual Studio's vcvars64.bat - the
    compiler's environment - the way Strata's setup.py builds there (cmake_build)."""
    if not WIN:
        if run(conf, env=env, check=False).returncode != 0:
            return "configuring"
        if run(build, env=env, check=False).returncode != 0:
            say("  (the build stopped - trying it once more: it continues where it stopped)")
            if run(build, env=env, check=False).returncode != 0:
                return "compiling"
        return None
    vcvars = vcvars or S.find_vcvars()
    if vcvars is None:
        return "looking for Visual Studio's C++ compiler (the Build Tools with 'Desktop development with C++')"
    bat = ROOT / bat_name
    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(conf)} || exit /b 3\r\n{cmdline(build)} && exit /b 0'
                   '\r\necho   (the build stopped - trying it once more: it continues where it stopped)\r\n'
                   f'{cmdline(build)} || exit /b 4\r\n', encoding="utf-8")
    say(f"  > {bat}   (Visual Studio's environment, then:)")
    say("  > " + cmdline(conf))
    say("  > " + cmdline(build))
    rc = run(["cmd", "/c", str(bat)], env=env, check=False, echo=False).returncode
    return None if rc == 0 else "compiling" if rc == 4 else "configuring"


PROBE_CU ="""#include <cmath>
#include <string>
#include <vector>
__global__ void k(float* x) { x[0] = rsqrtf(x[0]) + sinf(x[1]) + expf(x[2]) + __expf(x[3]) + sqrtf(x[4]); }
int main() { std::vector<std::string> v(1); return (int) std::cos(0.0) - 1 + (int) v.size() - 1; }
"""


def toolchain_for(archs, nvcc_given: str | None, hc_given: str | None) -> tuple:
    """(nvcc, host compiler or None for the default) that compile the engine's kind of CUDA here.  Every installed
    toolkit that can build for these GPUs (newest first; --nvcc alone if given) with the default g++, then each
    installed g++-N (newest first; --host-compiler alone if given), on a small file the way the engine compiles:
    C++20 and the math functions the kernels use.  A toolkit refuses a g++ newer than it supports, and a newer
    glibc's own rsqrtf/sinpi declarations clash with older CUDA headers - nothing but trying tells which pair works
    on a given system.  On Windows the compiler is Visual Studio's (in its vcvars64.bat environment): only the
    toolkits are tried."""
    import tempfile

    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if nvcc_given:
        nvccs = [nvcc_given]
    else:
        seen, found = set(), []
        for c in nvcc_candidates():
            if c and Path(c).exists() and os.path.realpath(c) not in seen:
                seen.add(os.path.realpath(c))
                found.append((c, version(c)))
        lo, hi = nvcc_range(archs)
        nvccs = [c for c, v in sorted((f for f in found if f[1]), key=lambda f: f[1], reverse=True)
                 if v >= lo and (hi is None or v < hi)]
    hcs = [None] if WIN else [hc_given] if hc_given else [None] + sorted(
        {q.name for d in ("/usr/bin", "/usr/local/bin") for q in Path(d).glob("g++-[0-9]*")
         if re.fullmatch(r"g\+\+-\d+", q.name)}, key=lambda n: -int(n[4:]))
    vcvars = S.find_vcvars() if WIN else None
    say("  Finding a CUDA toolkit and C++ compiler that build the engine here ...")
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "probe.cu"
        src.write_text(PROBE_CU, encoding="utf-8")
        for nv in nvccs:
            for c in hcs:
                cmd = [nv, "-std=c++20", "-c", str(src), "-o", str(Path(t) / "probe.o")] + \
                      (["-ccbin", shutil.which(c) or c] if c else [])
                if vcvars:                             # cl.exe is on PATH only inside Visual Studio's environment
                    bat = Path(t) / "probe.bat"
                    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(cmd)}\r\n', encoding="utf-8")
                    cmd = ["cmd", "/c", str(bat)]
                if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                    ok(f"{nv} with " + ("Visual Studio's C++ compiler" if WIN else c or "the default g++"))
                    return nv, c
    warn("no installed CUDA toolkit and C++ compiler pair compiled a test file: the build will show why (" +
         toolkit_hint(archs) + ")")
    return (nvccs[0] if nvccs else nvcc_given), hc_given


def compile_engine_hip(pc: dict, llama: Path, src: str, soft=False) -> dict | None:
    root = Path(pc["rocm"])
    cmake = pick_cmake()
    if not cmake:
        fail("CMake 3.24 or newer is needed")
    clang = hip_clang(root) or root / "llvm/bin/clang++"
    bindir = Path(pc.get("rocm_bin") or root / "bin")
    env = dict(os.environ, HIP_PLATFORM="amd", HIP_COMPILER="clang", HIP_RUNTIME="rocclr",
               ROCM_PATH=str(root), HIP_PATH=str(root))
    env["PATH"] = os.pathsep.join([str(bindir), str(clang.parent), env.get("PATH", "")])
    if not WIN:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(root / "lib"), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    # Keep compiler-cache writes inside this project, including when run in a workspace sandbox.
    env.setdefault("CCACHE_DIR", str(BUILD / ".ccache"))
    conf = [cmake]
    if WIN:
        # as tools\hip\build_maya_windows.bat, which built it on Windows (#54): ROCm's clang for the host code too, in
        # Visual Studio's environment (cmake_steps), with Ninja, the paths with forward slashes; AVX2 ggml
        ninja = venv_tool("ninja") or shutil.which("ninja")
        if not ninja:
            fail("Ninja is needed to compile the HIP engine on Windows", f"run {ME} --setup again (pip puts it into .venv)")
        bitcode = clang.parent.parent / "amdgcn" / "bitcode"
        env["HIP_DEVICE_LIB_PATH"] = str(bitcode)
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}",
                 f"-DCMAKE_C_COMPILER={(clang.parent / 'clang.exe').as_posix()}",
                 f"-DCMAKE_CXX_COMPILER={clang.as_posix()}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root.as_posix()}",
                 "-DSTRATA_PORTABLE=ON"]
        if " " not in str(root) + str(bitcode):        # (with a space in them, clang takes HIP_PATH and the above)
            conf.append(f"-DCMAKE_HIP_FLAGS=--rocm-path={root.as_posix()} --rocm-device-lib-path={bitcode.as_posix()}")
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_PREFILL_MMQ=ON",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_HIP_ARCHITECTURES={';'.join(pc['archs'])}",
             f"-DCMAKE_HIP_COMPILER={clang.as_posix()}", f"-DCMAKE_PREFIX_PATH={root.as_posix()}",
             f"-DSTRATA_GGML_DIR={llama.as_posix()}"]
    jobs = max(1, min(4, (os.cpu_count() or 4) // 2))
    say("  Compiling Maya for AMD " + ", ".join(pc["archs"]) + " ...")
    if cmake_steps(conf, [cmake, "--build", str(BUILD), "--target", "strata", "-j", str(jobs)], env,
                   "build-maya-hip.bat", vcvars=S.find_vcvars(cuda=False) if WIN else None):
        if soft:
            warn("HIP rebuild failed; starting the previous engine")
            return None
        fail("the HIP engine build failed", "see the compiler output above")
    if WIN:
        # Windows takes a program's DLLs from its own folder, then System32 - where AMD's driver puts its own HIP
        # runtime - and PATH only after: the runtime the engine was built with goes next to it (as llama.cpp's
        # Windows builds do; with the driver's, cudaMemGetInfo failed there). lib_dirs keeps the libraries
        for pattern in ("amdhip64*.dll", "amd_comgr*.dll", "rocm_kpack*.dll"):
            for dll in bindir.glob(pattern):
                try:
                    shutil.copy2(dll, EXE.parent)
                except OSError as e:                   # (in use: an engine still running keeps the one it has)
                    warn(f"could not copy {dll.name} next to the engine ({e})")
    meta = {"backend": "hip", "src": src, "archs": pc["archs"], "rocm": str(root),
            "gpu_ids": [g["index"] for g in pc["gpus"]], "llama": str(llama),
            "lib_dirs": [str(bindir if WIN else root / "lib")], "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def build_step(a, pc, llama: Path) -> dict:
    step(4, "the engine")
    src = S.source_hash(S.ENGINE_SOURCES)              # src/, include/, third_party/ggml, CMakeLists.txt
    meta = read_json(STAMP)
    if pc.get("backend") == "hip":
        if (EXE.exists() and not a.rebuild and meta.get("backend") == "hip" and meta.get("src") == src
                and meta.get("archs") == pc["archs"] and meta.get("rocm") == pc["rocm"]):
            ok(f"HIP engine already compiled: {EXE}")
            return meta
        return compile_engine_hip(pc, llama, src)
    if (EXE.exists() and not a.rebuild and meta.get("src") == src and set(pc["archs"]) <= set(meta.get("archs", []))
            and (not a.nvcc or a.nvcc == meta.get("nvcc"))
            and (not a.host_compiler or a.host_compiler == meta.get("host_compiler"))):
        ok(f"engine already compiled for " + ", ".join(f"sm_{x}" for x in meta["archs"]) + f": {EXE}")
        return meta
    nvcc, hc = toolchain_for(pc["archs"], a.nvcc, a.host_compiler)
    pc["nvcc"] = nvcc or pc["nvcc"]
    return compile_engine(pc["archs"], [g["index"] for g in pc["gpus"]], pc["nvcc"], hc, llama, src)


def refresh_engine(cfg: dict) -> None:
    """At a start: the engine's source changed since it was compiled (a git pull) - compile what changed."""
    meta = read_json(STAMP)
    if not meta or Path(cfg["exe"]).resolve() != EXE.resolve():
        return
    src = S.source_hash(S.ENGINE_SOURCES)
    if meta.get("src") == src:
        return
    if cfg.get("backend") == "hip":
        root = Path(meta.get("rocm") or "/opt/rocm")
        llama = Path(meta.get("llama") or ROOT / "third_party/llama.cpp")
        if hip_clang(root) is None or not (llama / "ggml/CMakeLists.txt").exists():
            warn("HIP compiler or llama.cpp source is missing; starting the previous engine")
            return
        pc = {"rocm": str(root), "archs": meta["archs"],
              "gpus": [{"index": i} for i in cfg.get("gpu", [0])]}
        if WIN and meta.get("lib_dirs"):
            pc["rocm_bin"] = meta["lib_dirs"][0]
        compile_engine_hip(pc, llama, src, soft=True)
        return
    say("  The engine's source changed since it was compiled (an update): compiling what changed ...")
    nvcc, llama = meta.get("nvcc"), Path(meta.get("llama") or ROOT / "third_party" / "llama.cpp")
    if not nvcc or not Path(nvcc).exists() or not (llama / "ggml" / "CMakeLists.txt").exists():
        warn("the compiler or llama.cpp's source it was built with is gone: starting the engine compiled before "
             f"({ME} --setup --rebuild compiles it again)")
        return
    gpu_ids = meta.get("gpu_ids") or (cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [0])
    compile_engine(meta["archs"], gpu_ids, nvcc, meta.get("host_compiler"), llama, src, soft=True)


# ------------------------------------------------------------------------------------------------ 5. the model
def shard_set(first: Path) -> list:
    """All files of a split GGUF from its first one (<name>-00001-of-0000N.gguf), or just the file."""
    m = SHARD_RE.search(first.name)
    if not m:
        return [first]
    n, stem = int(m.group(2)), first.name[:m.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def gguf_head(path: Path) -> tuple:
    """(general.architecture, tensor count) from the start of a GGUF header: only the key/values before the
    architecture are read (gguf-py writes it first), so a folder of big files is scanned quickly.  ("", 0) when it
    is not a GGUF file or the key does not come early."""
    import struct
    from gguf_reader import GGUF_MAGIC, GGUF_META
    try:
        with open(path, "rb") as f:
            def blob():
                (n,) = struct.unpack("<Q", f.read(8))
                if n > 1 << 20:
                    raise ValueError("not a GGUF string")
                return f.read(n)
            magic, _, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
            if magic != GGUF_MAGIC:
                return "", 0
            for _ in range(min(n_kv, 32)):
                key = blob()
                kind, size = GGUF_META[struct.unpack("<I", f.read(4))[0]]
                if kind == "string":
                    val = blob()
                    if key == b"general.architecture":
                        return val.decode("utf-8", "replace"), n_tensors
                elif kind == "array":
                    _, esize = GGUF_META[struct.unpack("<I", f.read(4))[0]]
                    (count,) = struct.unpack("<Q", f.read(8))
                    if esize is None:                  # an array of strings (the vocabulary) - the key was not early
                        break
                    f.seek(esize * count, 1)
                else:
                    f.seek(size, 1)
    except (OSError, ValueError, KeyError, struct.error):
        pass
    return "", 0


def glm_files(d: Path) -> list:
    """The GLM-5.3-Flash models in the folder `d`, as their first files (-00001-of- for a split one).  A single file
    without tensors (a vocabulary, like the vision folder's) is no model; unsloth's split models keep only metadata
    in their first file."""
    try:
        files = sorted(d.glob("*.gguf"))
    except OSError:
        return []
    found = []
    for f in files:
        m = SHARD_RE.search(f.name)
        if m and int(m.group(1)) != 1:
            continue
        arch, n = gguf_head(f)
        if arch in GLM_ARCHS and (n > 0 or m):
            found.append(f)
    return found


def local_first(p: Path) -> tuple:
    """(the model's first .gguf file, None) from one of its files or the folder that holds it, or (None, why not)."""
    p = p.expanduser().resolve()
    if p.is_file():
        first = shard_set(p)[0]
        arch = gguf_head(first)[0] if first.exists() else ""
        if first.exists() and arch not in GLM_ARCHS:
            return None, f"{first.name} is " + (f"a {arch!r} model, not GLM-5.3-Flash ({GLM_ARCHS[0]})" if arch else
                                                "not a GGUF file")
        return first, None
    found = glm_files(p) if p.is_dir() else []
    if not found:
        return None, (f"no GLM-5.3-Flash GGUF file in {p}" if p.is_dir() else f"{p} does not exist")
    if len(found) > 1:
        return None, f"{p} holds {len(found)} GLM-5.3-Flash models: name the file (" + \
                     ", ".join(f.name for f in found) + ")"
    return found[0], None


def incomplete(path: Path):
    """Why this GGUF file is not whole, or None when it is: as long as its own tensor directory says (the same test
    as Strata's check_shards, without stopping).  Reads the header only."""
    if not path.exists():
        return "missing"
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(path)
    except Exception as e:                             # noqa: BLE001 - a partial header raises anything
        return f"not a whole GGUF file ({e})"
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    have = path.stat().st_size
    return None if have >= need else f"short: {have:,} of {need:,} bytes"


def quant_of(first: Path) -> str:
    for q, mm in MODELS.items():                       # a Maya download, under its name on Hugging Face or its old one
        if first.name in [names[0] for names in [shard_names(mm)] + old_names(mm)]:
            return q
    m = re.match(r"GLM-5\.3-Flash-(.+?)(-\d{5}-of-\d{5})?\.gguf$", first.name, re.I)
    return m.group(1) if m else first.parent.name


def sha256_ok(path: Path, want) -> bool:
    """`want`: the published sha256, or a list of them - a file published again with only its header changed (v1.0.28:
    the architecture spelled as llama.cpp spells it) keeps the earlier download valid."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest() in [w.lower() for w in ([want] if isinstance(want, str) else want)]


def offer_download(a, quant: str, d: Path, shards: list) -> bool:
    """Shows what would be downloaded - the source, the size, the exact commands - and downloads only after a yes
    (or --download-model).  False: nothing was downloaded."""
    m = MODELS[quant]
    missing = [s for s in shards if incomplete(s)]
    urls = {s: hf_url(m["repo"], m["revision"], m["folder"], hf_name(m, s)) for s in shards}
    on_disk = sum(s.stat().st_size for s in shards if s.exists()) / 1e9
    remaining = max(0.0, m["download_gb"] - on_disk)
    free = shutil.disk_usage(existing(d)).free / 1e9
    curl = shutil.which("curl")
    cmds = [["mkdir"] + ([] if WIN else ["-p"]) + [str(d)]] + \
        [["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]] for s in missing]
    say(f"  {quant}: {m['about']}.")
    say(f"  Source: https://huggingface.co/{m['repo']}" + (f" (folder {m['folder']}/)" if m["folder"] else "") +
        "; the files' own license applies.")
    say(f"  The model is {m['download_gb']:.1f} GB in {len(shards)} file{'s' if len(shards) != 1 else ''}; "
        f"{len(missing)} still to download, about "
        f"{remaining:.0f} GB, into")
    say(f"    {d}   ({free:.0f} GB free there)")
    if free < remaining + 3:
        fail(f"not enough free space in {d}: about {remaining + 3:.0f} GB are needed (the model + its pack)",
             f"free some space, or put the model on another drive: {ME} --setup --models-dir " +
             (r"D:\Maya-models" if WIN else "/path/on/nvme"))
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads experts from these files while it answers - use an "
             "NVMe SSD (--models-dir)")
    say("  The exact commands (resumable: running them again continues an interrupted download):")
    for c in cmds:
        say("    " + shell_join(c))
    say("  You can also run them yourself (or download the files any other way into that folder, or pass")
    say(f"  --gguf-dir <folder with the files>), then run {ME} again.")
    if not curl:
        warn("curl is not installed" + ("" if WIN else " (sudo apt-get install -y curl)") + ": if you say yes, the "
             "same URLs are downloaded with Python instead (also resumable)")
    if not a.download_model and ask(f"  Download about {remaining:.0f} GB now with these commands? (y/n)",
                                    ["y", "n"], "n", False) != "y":
        say()
        say(f"Nothing was downloaded. When the files are in place, run {ME} again "
            f"({ME} --download-model downloads them without asking).")
        return False
    d.mkdir(parents=True, exist_ok=True)
    for s in missing:
        if curl:
            cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]]
            if run(cmd, check=False).returncode != 0:
                fail(f"the download of {s.name} stopped (the reason is above)",
                     f"run {ME} --download-model again: it continues where it stopped")
        else:
            S.download(urls[s], s)
        why = incomplete(s)
        if why:
            fail(f"{s.name} after the download: {why}", f"delete it and run {ME} --download-model again")
        want = m["sha256"].get(hf_name(m, s))
        if want:
            say(f"  checking {s.name}'s sha256 ...")
            if not sha256_ok(s, want):
                fail(f"{s.name}: the sha256 does not match the published one", "delete it and download it again")
        ok(f"{s.name} downloaded")
    return True


def model_step(a, models: Path, choice: tuple):
    """(model folder, shards, quant name), or None when the files are not there and were not downloaded."""
    step(5, "the model (GLM-5.3-Flash, GGUF)")
    kind, what = choice
    if kind == "local":
        shards, quant, d = shard_set(what), quant_of(what), what.parent
        for s in shards:
            why = incomplete(s)
            if why:
                fail(f"{s.name}: {why}", f"finish copying or downloading it, then run {ME} again")
        if quant not in MODELS:
            warn(f"{quant}: experimental - only {', '.join(MODELS)} has been measured with Maya (README.md)")
    else:
        quant = what
        m = MODELS[quant]
        d = download_dir(models, quant)
        shards = local_shards(m, d)
        if any(incomplete(s) for s in shards) and not offer_download(a, quant, d, shards):
            return None
    from gguf_reader import GGUFFile
    arch = str(GGUFFile(shards[0]).metadata.get("general.architecture", ""))
    if arch not in GLM_ARCHS:
        fail(f"{shards[0].name} is a {arch!r} model, not GLM-5.3-Flash (glm5-next)")
    gb = sum(s.stat().st_size for s in shards) / 1e9
    ok(f"{quant}: {len(shards)} file(s), {gb:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: expect slow answers (the engine reads experts from it) - an NVMe SSD "
             "is strongly recommended")
    return d, shards, quant


# ------------------------------------------------------------------------------------------------ 6. the pack
def pack_source(pack: Path):
    """The first .gguf file a pack indexes (native_experts.txt's header names it), or None (no pack there)."""
    try:
        with open(pack / "native_experts.txt", encoding="utf-8") as f:
            m = re.search(r"absolute offsets in ([^,)]+)", f.readline())
    except OSError:
        return None
    return m.group(1) if m else None


def pack_step(a, d: Path, shards: list, llama: Path) -> Path:
    step(6, "the pack (the engine's index of the model files)")
    # the engine finds the GGUF files at <pack>/.. (src/core/glm_model.cu, load_pack): the pack lives in their folder,
    # in pack/ - or pack-<model>/ when pack/ indexes another model of the same folder
    pack = d / "pack"
    if pack_source(pack) not in (None, shards[0].name):
        pack = d / f"pack-{quant_of(shards[0])}".lower()
    if not a.repack and pack_source(pack) == shards[0].name and all((pack / f).exists() for f in PACK_FILES):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written next to the GGUF files (the engine finds them at <pack>/..)",
             "link the .gguf files into a writable folder (" + ("mklink /H DIR\\<file> <file>, for each file" if WIN
                                                                else "mkdir -p DIR && ln -s /path/to/*.gguf DIR/") +
             ") and pass "
             "--gguf-dir DIR")
    say("  Writing the index of every tensor, the small float weights (about 1 GB) and the tokenizer; the experts")
    say("  stay in the GGUF files (a few minutes) ...")
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    # --compat-bf16: the GLM pack stores the projections its kernels read as BF16 (router, indexer, absorbed MLA,
    # KDA gates) in BF16 even where the GGUF quantized them or keeps them F32 (tools/iq_pack.py GLM5NEXT_BF16)
    run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
         "--compat-bf16"], env=env)
    missing = [f for f in PACK_FILES if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    ok(f"pack: {pack}")
    return pack


# ------------------------------------------------------------------------------------------------ 7. images
def fetch(url: str, dst: Path, want: str | None) -> None:
    """One file, resumable (curl -C -, else Python), then its sha256 when one is published."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        if run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dst), url], check=False).returncode != 0:
            fail(f"the download of {dst.name} stopped (the reason is above)", f"run {ME} --setup again: it "
                                                                              "continues where it stopped")
    else:
        S.download(url, dst)
    if want and not sha256_ok(dst, want):
        dst.unlink(missing_ok=True)
        fail(f"{dst.name}: the sha256 does not match the published one (the file was removed)",
             f"run {ME} --setup again to download it again")


def compile_vision(pc, meta, llama: Path, src: str) -> bool:
    """strata-vision: llama.cpp's mtmd at the pinned commit with GLM-5.3-Flash's vision tower added
    (tools/vision/glm5next_patch.py, idempotent), for the same GPUs and toolkit as the engine."""
    cmake = pick_cmake()
    if cmake is None:
        return False
    run([sys.executable, str(ROOT / "tools" / "vision" / "glm5next_patch.py"), str(llama)])
    gen = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (VBUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (VBUILD / "CMakeCache.txt").exists() else None
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and (gen is None or gen.group(1).strip() == "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    archs = meta.get("archs") or pc["archs"]
    conf += ["-S", str(ROOT / "tools" / "vision"), "-B", str(VBUILD), "-DCMAKE_BUILD_TYPE=Release",
             f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
             "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(str(x) for x in archs),
             f"-DCMAKE_CUDA_COMPILER={meta.get('nvcc') or pc['nvcc']}"]
    if meta.get("host_compiler"):
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(meta['host_compiler']) or meta['host_compiler']}")
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    say("  Compiling the vision encoder (llama.cpp's image library with ggml's CUDA kernels: 10-20 minutes, once) ...")
    if cmake_steps(conf, [cmake, "--build", str(VBUILD), "--target", "strata-vision", "-j", str(jobs)], None,
                   "build-maya-vision.bat"):
        return False
    VSTAMP.write_text(json.dumps({"src": src, "archs": archs, "llama": str(llama),
                                  "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1), encoding="utf-8")
    return VEXE.exists()


def vision_step(a, pc, meta, llama: Path, d: Path, quant: str) -> dict | None:
    """The config's "vision" entry - the image encoder and its files - or None (images off).  The server starts the
    encoder only while a request's new pictures are encoded, in GPU memory the model lends it, and measures on the
    first start how much that is on this GPU (serve/server.py, vision_footprint)."""
    step(7, "images (the vision encoder)")
    if pc.get("backend") == "hip":
        ok("HIP port: text only; vision is not enabled")
        return None
    # a GGUF of your own reads pictures with the same files: the vision tower and the tokenizer are the same in
    # every GLM-5.3-Flash quant
    spec = MODELS.get(quant) or MODELS[next(iter(MODELS))]
    m = spec.get("vision")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return None
    if m is None:
        warn(f"no vision files are published for {quant}: the model reads text only")
        return None
    vd = d / m["folder"]
    files = {k: vd / m[k] for k in ("mmproj", "vocab")}
    missing = [p for p in files.values() if not p.exists() or (m["sha256"].get(p.name) and p.stat().st_size == 0)]
    if missing:
        repo, rev = m.get("repo", spec["repo"]), m.get("revision", spec["revision"])
        src_url = {p: hf_url(repo, rev, m["folder"], p.name) for p in missing}
        say(f"  The vision encoder's files ({m['download_gb']:.2f} GB) from https://huggingface.co/"
            f"{repo} (folder {m['folder']}/), into {vd}:")
        for p in missing:
            say("    " + shell_join(["curl", "-L", "--fail", "-C", "-", "-o", str(p), src_url[p]]))
        if not (a.download_model or a.yes) and ask("  Download them now? (y/n)", ["y", "n"], "y", False) != "y":
            warn(f"not downloaded: the model reads text only ({ME} --setup asks again)")
            return None
        for p in missing:
            fetch(src_url[p], p, m["sha256"].get(p.name))
            ok(f"{p.name} downloaded")
    src = S.source_hash(("tools/vision",))
    vmeta = read_json(VSTAMP)
    if not (VEXE.exists() and not a.rebuild and vmeta.get("src") == src
            and set(meta.get("archs") or pc["archs"]) <= set(vmeta.get("archs", []))):
        if not compile_vision(pc, meta, llama, src):
            warn("the vision encoder did not compile (the reason is above): the model reads text only - "
                 f"{ME} --setup --rebuild tries again")
            return None
    ok(f"vision encoder: {VEXE}")
    return {"exe": str(VEXE), "mmproj": str(files["mmproj"]), "model": str(files["vocab"]), "gpu": True}


# ------------------------------------------------------------------------------------------------ 8. config + start
def parse_env(items) -> dict:
    env = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            fail(f"--env takes KEY=VALUE, e.g. --env STRATA_GLM_RAM_HEADROOM_GB=4, not {it!r}")
        env[k] = v
    return env


def write_run_script(cfg_path: Path, port: int) -> Path:
    """run-<config>.sh (.bat): this config's start through maya.py - on its screen in a terminal (its loading and
    dashboard), else (--plain, or no terminal: a service) the server's text as it prints."""
    cmd = [sys.executable, str(HERE / "maya.py"), "--config", str(cfg_path), "--port", str(port)]
    if WIN:
        script = ROOT / f"run-{cfg_path.stem}.bat"
        script.write_text(f'@echo off\r\nrem starts the Project Maya dashboard and API (written by maya.py; {ME} does '
                          f'the same)\r\ncd /d "{ROOT}" || exit /b 1\r\n{cmdline(cmd)} %*\r\n', encoding="utf-8")
        return script
    script = ROOT / f"run-{cfg_path.stem}.sh"
    script.write_text("#!/bin/sh\n# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the "
                      "same)\ncd " + shlex.quote(str(ROOT)) + " || exit 1\nexec " + shlex.join(cmd) + ' "$@"\n',
                      encoding="utf-8")
    script.chmod(0o755)
    return script


def write_config(a, pc, meta, pack: Path, quant: str, ctx: int, models: Path, vision: dict | None,
                 choice: tuple | None = None) -> Path:
    step(8, "the configuration and the start script")
    port = a.port or 8080
    # --prefill auto (as Strata's setup writes it): the engine picks its prompt chunk - the largest its expert pool
    # can lend, up to 8192; a number (`32768`) sets the chunk itself
    cfg = {"exe": str(EXE), "args": ["--glm-pack", str(pack), "--max-context", str(ctx), "--prefill", "auto"],
           "cwd": str(ROOT),
           "tokenizer": str(pack / "tokenizer"), "model_name": MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": dict(SAMPLING), "reasoning_effort": EFFORT, "lib_dirs": meta.get("lib_dirs") or [],
           "port": port}
    env = parse_env(a.env)
    if pc.get("backend") == "hip":
        cfg["backend"] = "hip"
        # Leave space for the Linux desktop. The engine sizes its prompt chunk from
        # the available prompt-memory budget; forcing 256 here severely slows HIP.
        hip_env = {"STRATA_GLM_RESERVE_MB": "3072", "STRATA_GLM_RAM_HEADROOM_GB": "16"}
        gpus = pc["gpus"]
        if len(gpus) == 1:
            hip_env["STRATA_GLM_SPLIT"] = "0"   # two cards: the engine picks the split (and drafts with MTP)
        # Larger prompt sub-batches feed the matrix cores much better (7900 XT: ~250 -> ~410 tok/s); they need
        # a bigger prompt budget, borrowed from the expert pool only while a prompt runs.
        if any(hip_unified_memory(g) for g in gpus):   # an APU is always the only GPU (check_hip_pc)
            total_ram, _ = mem_gb()
            if WIN:     # Windows counts the GPU's carve-out apart from its RAM, and its desktop runs on that GPU
                total_ram += sum(g.get("vram_gb", 0) for g in gpus)
            hip_env.update({"STRATA_GLM_RESERVE_MB": "3072" if WIN else "1024", "STRATA_GLM_PREFILL_SUB": "1024",
                            "STRATA_GLM_PREFILL_MB": "6144" if total_ram >= 96 else "4096"})
        elif min(g.get("vram_gb", 0) for g in gpus) >= 20:
            hip_env.update({"STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "4096"})
        # The prompt projections' hipBLASLt solutions measured per architecture (tools/hip), one table per card's
        # architecture (':'-separated: each card takes its own). The engine refuses a table made for another
        # hipBLASLt version and keeps plain hipBLAS.
        tables = []
        for arch in dict.fromkeys(g.get("arch", "") for g in gpus):
            found = sorted((ROOT / "tools" / "hip").glob(f"{arch}-glm-hipblaslt-*.txt"))
            if found:
                tables.append(str(found[-1]))
        if tables:
            hip_env["STRATA_HIPBLASLT_TUNING"] = ":".join(tables)
        env = {**hip_env, **env}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if vision:
        cfg["vision"] = vision
    suffix = "-hip" if pc.get("backend") == "hip" else ""
    cfg_path = ROOT / f"maya-{quant.lower()}{suffix}.json"
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    local = choice[1] if choice and choice[0] == "local" else None   # (choose_model's answer)
    if cfg_path.exists():
        kept = keep_args(read_json(cfg_path).get("args") or [], cfg["args"], ("--prefill",))
        if kept:
            ok("kept from the config before: " + ", ".join(kept))
    for line in prefill_tips(cfg["args"], mem_gb()[0], len(pc["gpus"])):
        ok(line)
    gguf_dir = local.parent if local else Path(a.gguf_dir).expanduser().resolve() if a.gguf_dir else None
    cfg["installer"] = {"models_dir": str(models), "quant": quant, "gguf": str(local) if local else None,
                        "gguf_dir": str(gguf_dir) if gguf_dir else None, "written": time.strftime("%Y-%m-%d %H:%M")}
    cal = saved_calibration(cfg, any_context=True)     # tuned on this PC for this model (and context) before
    if cal is not None:
        tuned = CAL.apply(cfg.get("env") or {}, cal.get("settings") or {})
        tuned.update(env)                              # (an --env given now wins)
        if tuned:
            cfg["env"] = tuned
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else "") +
           (": " + ", ".join(f"{k}={v}" for k, v in cal["settings"].items()) if cal.get("settings") else ""))
        if cal.get("context"):
            ok(f"(tuned at context {cal['context']}: {ME} --calibrate tunes this context too)")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, port)}")
    return cfg_path


PREFILL_BIG_RAM_GB = 96        # Strata's benchmarks: 32768-token chunks +21-35% at 96 GB, ~3x slower at 32 GB
PREFILL_RISK_RAM_GB = 64       # below this a chunk set above 8192 is warned about


def prefill_tips(args: list, ram: float, gpus: int = 2) -> list:
    """Strata's --prefill recommendations for a split (one GPU's auto already takes up to 32768): text only, nothing in
    the config changes."""
    prefill = args[args.index("--prefill") + 1] if "--prefill" in args[:-1] else None
    if gpus < 2:
        return []
    if prefill is not None and prefill.isdigit() and int(prefill) > 8192 and ram < PREFILL_RISK_RAM_GB:
        return [f"warning: --prefill {prefill} on {ram:.0f} GB of RAM: on a split, 32768-token chunks read prompts "
                "slower than --prefill auto with little RAM (two V100s, 30 GB: 551 against 709 tok/s); they paid off "
                "(+21-35%) with 96 GB in Strata's community benchmarks"]
    if prefill == "auto" and ram >= PREFILL_BIG_RAM_GB:
        return [f"tip: with {ram:.0f} GB of RAM, --prefill 32768 in the config's args read prompts 21-35% faster in "
                "Strata's community benchmarks; not set, nothing changes"]
    return []


def keep_args(old: list, new: list, flags: tuple) -> list:
    """The engine arguments `flags` as the old config had them (edited by hand, or written by an earlier setup):
    their values replace the new ones, or are added.  `new` is changed in place; what was kept is returned."""
    kept = []
    for flag in flags:
        if flag not in old[:-1]:
            continue
        v = old[old.index(flag) + 1]
        if flag in new[:-1]:
            if new[new.index(flag) + 1] == v:
                continue
            new[new.index(flag) + 1] = v
        else:
            new += [flag, v]
        kept.append(f"{flag} {v}")
    return kept


# ------------------------------------------------------------------------------------------------ calibration
CAL_STORE = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "project-maya" / "calibration.json"


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: these GPUs, this CPU and RAM, the model (its quant), the context, text or
    images (the context's KV cache and the image encoder take VRAM from the expert pool)."""
    found = {g["index"]: g for g in S.gpus() or []}
    sel = [found.get(i) or {} for i in cfg.get("gpu") or []]
    a = cfg.get("args") or []
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a[:-1] else "?"
    quant = (cfg.get("installer") or {}).get("quant") or cfg.get("model_name", "?")
    return "|".join([" + ".join(g.get("name", "?") for g in sel) or "?",
                     f"{sum(g.get('vram_gb', 0) for g in sel):.0f}GB", S.cpu_info()[0], f"{S.ram_gb():.0f}GB", quant,
                     str(ctx), "images" if cfg.get("vision") else "text"])


def load_calibrations() -> dict:
    try:
        return json.loads(CAL_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def saved_calibration(cfg: dict, any_context=False) -> dict | None:
    """The settings an earlier tuning found for this PC, model and context, if any (CUDA only, as the tuning).
    any_context: else the ones tuned at the nearest other context, with that context in "context" - the CPU lane's
    split depends on the PC far more than on the context, and a setup for another context used to drop them (#57: the
    engine's own settings, half the speed on a two-socket DDR3 PC)."""
    if cfg.get("backend") == "hip":
        return None
    store, key = load_calibrations(), hardware_key(cfg)
    if key in store or not any_context:
        return store.get(key)
    want = key.split("|")
    best, best_d = None, None
    for k, v in store.items():
        have = k.split("|")
        if len(have) != len(want) or have[:5] != want[:5] or have[6:] != want[6:]:
            continue
        try:
            d = abs(math.log(int(have[5]) / int(want[5])))
        except (ValueError, ZeroDivisionError):
            continue
        if isinstance(v, dict) and (best_d is None or d < best_d):
            best, best_d = dict(v, context=int(have[5])), d
    return best


def busy_gpus(cfg: dict) -> list:
    """The config's GPUs with more than 1 GiB of VRAM in use (another engine on them): [(index, MiB)]."""
    try:
        txt = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    used = {}
    for line in txt.splitlines():
        i, _, m = line.partition(",")
        try:
            used[int(i)] = int(float(m))
        except ValueError:
            pass
    return [(i, used[i]) for i in cfg.get("gpu") or [] if used.get(i, 0) > 1024]


def calibrate_config(cfg_path: Path) -> bool:
    """Tune the CPU lane on this PC (tools/calibrate_glm.py): the result goes into the config's env and into
    CAL_STORE for this PC, model and context (a setup again applies it)."""
    cfg = read_json(cfg_path)
    if cfg.get("backend") == "hip":
        warn("the tuning measures the CUDA engine's CPU lane (NVIDIA GPUs): the HIP port keeps its own settings")
        return False
    busy = busy_gpus(cfg)
    if busy:
        warn("the tuning needs the GPU(s) to itself: " + ", ".join(f"GPU {i} has {m} MiB in use" for i, m in busy) +
             " (a model server still running?) - stop it, then run ./maya.sh --calibrate")
        return False
    say()
    say("  Tuning Maya for this PC: the output speed is measured with a few splits of the work between the CPU and")
    say("  the PCIe link, and with fewer CPU threads. It takes about 10-15 minutes (the model loads first); the PC is")
    say("  busy meanwhile.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops a setup: the engine's own choices stay
        warn(f"the tuning did not finish ({e}): the engine's own settings stay")
        why = CAL.engine_error(log, since)
        if why:
            say(f"       the engine said: {why}")
        return False
    env = CAL.apply(cfg.get("env") or {}, res["settings"])
    if env:
        cfg["env"] = env
    else:
        cfg.pop("env", None)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    store = load_calibrations()
    store[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                "date": time.strftime("%Y-%m-%d")}
    try:
        CAL_STORE.parent.mkdir(parents=True, exist_ok=True)
        CAL_STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {CAL_STORE} ({e}): {cfg_path.name} has the settings, a setup again does not")
    speed = f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k}={v}" for k, v in res["settings"].items()) + speed)
    else:
        ok("tuned for this PC: the engine's own settings are already the fastest here" + speed)
    return True


def server_command(cfg_path: Path, a) -> tuple:
    """The dashboard's server for an installed config: (its command, its environment, what the setup's screen shows of
    it).  The engine is compiled again first when its source changed (an update)."""
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    for p, what in ((Path(cfg.get("exe", "")), "the engine"), (pack, "the pack"),
                    (Path(cfg.get("tokenizer", "")) / "vocab.json", "the tokenizer")):
        if p is None or not p.exists():
            fail(f"{cfg_path.name}: {what} is missing ({p})", f"run {ME} --setup to repair it")
    cfg_path.touch()                                   # the most recently used model
    refresh_engine(cfg)
    port = a.port or cfg.get("port") or 8080
    host = a.host or cfg.get("host") or "127.0.0.1"
    key = a.api_key or cfg.get("api_key")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)]
    if a.host:
        cmd += ["--host", a.host]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.gpus or a.gpu is not None:                    # this start only, on these cards
        cmd += ["--gpu", str(a.gpus or a.gpu)]
    if WIN or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        cmd.append("--open")                           # a desktop: the browser opens when the model is ready
    here = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else None
    gpus = str(a.gpus or a.gpu if a.gpus or a.gpu is not None else cfg.get("gpu", 0)).strip("[]").split(",")
    info = {"model": cfg.get("model_name", MODEL_NAME), "quant": (cfg.get("installer") or {}).get("quant"),
            "context": ctx, "dashboard": f"http://{here}:{port}/", "api": f"http://{here}:{port}/v1",
            "log": cfg.get("log"), "key": (key or "").split(",")[0],      # (its dashboard reads /metrics with it)
            "gpus": [int(g) for g in gpus if g.strip().isdigit()]}         # (its load: one part each, in this order)
    env = dict(os.environ, MAYA_RESTART_ON_UPDATE="1")  # (the dashboard may update Maya)
    if S.UI is not None:                               # (the setup's screen shows the same in its card, and reads
        env["PYTHONUNBUFFERED"] = "1"                  # the server's output from a pipe)
        found = S.amd_gpus() if cfg.get("backend") == "hip" else S.gpus()
        info["vram"] = {g["index"]: g["vram_gb"] for g in found}   # (its load: a GPU not started yet, by its VRAM)
        return cmd, env, info
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', MODEL_NAME)} ({cfg_path.name}).")
    say(f"  Dashboard:        http://{here}:{port}/")
    say(f"  API (OpenAI):     http://{here}:{port}/v1        (any model name; the API key: "
        + ("the one you set)" if key else "any)"))
    say(f"  API (Anthropic):  http://{here}:{port}/v1/messages")
    if host not in ("127.0.0.1", "localhost", "::1"):
        say("  Other devices:    the server prints this machine's addresses when it is ready" +
            ("" if key else " - WARNING: no API key, anyone on your network can use it (--api-key KEY)"))
    say("  Loading takes a few minutes: the engine pins up to all but about 6 GB of the free RAM for its expert tier")
    say("  (STRATA_GLM_RAM_HEADROOM_GB changes the 6) and warms its caches - the first answers are the slowest.")
    say("  Ctrl+C (or closing this terminal) stops it. Engine log: " + str(cfg.get("log", "")))
    say("  " + "-" * 100)
    return cmd, env, info


def start(cfg_path: Path, a) -> int:
    """Maya's server for an installed config: on a screen of its own in a terminal (tools/setup_tui.py: its
    dashboard - the Monitor's numbers from the engine - and its log), else in the terminal as it prints."""
    if setup_screen(a, asks=False):
        import setup_tui
        rc = setup_tui.run(lambda: serve_on_screen(cfg_path, a), maya_version(), None, ME, steps=False)
        return after_server(rc, a)
    cmd, env, _ = server_command(cfg_path, a)
    return after_server(subprocess.call(cmd, env=env), a)


def serve_on_screen(cfg_path: Path, a) -> int:
    """Maya's server on the setup's screen (setup.UI): its exit code; one it did not choose stops there, with the
    last line its engine logged in this start (mostly the reason: "strata generate: pack: out of memory")."""
    cmd, env, info = server_command(cfg_path, a)
    log = Path(info["log"]) if info.get("log") else None
    seen = log.stat().st_size if log is not None and log.is_file() else 0
    rc = S.UI.serve(cmd, env, info)
    if rc not in (0, UPDATE_EXIT):
        last = last_line(log, seen)
        fail(f"Maya stopped: its server ended with exit code {rc}",
             (f"its engine's last line: {last} (the whole log: {log})" if last else "the reason is in its output")
             + f"; {ME} starts it again")
    return rc


def last_line(path: Path | None, since: int) -> str:
    """The last line written to `path` after byte `since` ("": none, or no such file)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(since, f.seek(0, os.SEEK_END) - 8192))
            lines = [s.strip() for s in f.read().decode(errors="replace").splitlines() if s.strip()]
    except (OSError, TypeError):
        return ""
    return lines[-1][:200] if lines else ""


def after_server(rc: int, a) -> int:
    """The server ended: its exit code - or About > Updates moved this folder to a new release, and the new version
    starts: its maya.py compiles what changed in the engine and loads the same model (the most recently used, so no
    question)."""
    if rc != UPDATE_EXIT:
        return rc
    again = [sys.executable, str(HERE / "maya.py"), "--yes"]
    for flag, v in (("--backend", a.backend), ("--port", a.port), ("--host", a.host), ("--api-key", a.api_key),
                    ("--gpu", a.gpu), ("--gpus", a.gpus), ("--config", getattr(a, "config", None))):
        if v is not None:
            again += [flag, str(v)]
    say()
    say("  Updated from the dashboard: starting the new version ...")
    sys.stdout.flush()
    if WIN:
        return subprocess.call(again)
    os.execv(sys.executable, again)
    return 0


# ------------------------------------------------------------------------------------------------ the report
# the engine log's lines that tell where the time goes: how the model was split across VRAM / RAM / the SSD, the
# prompt path's chunks, and the per-token breakdown ("glm stat": VRAM hits, RAM fetches, disk reads, CPU lane)
REPORT_LINES = re.compile(r"glm fast:|glm prefill: (?:CUDA|HIP)|glm split|glm stat|glm slots|glm prefill: \d|ERR|error|failed|"
                          r"out of memory", re.I)
STAT_DECODE = re.compile(r"glm stat: decode ([\d.]+) ms/tok")


def speed_lines(text) -> list:
    """The log lines REPORT_LINES picks.  A request that only read a prompt (one token out) has no decode to report:
    its "glm stat" line keeps only the prompt part, not a meaningless decode speed (96,000 tokens/s, issue #8)."""
    picked = []
    for x in text:
        if not REPORT_LINES.search(x) or "warming the expert tiers" in x:
            continue
        m = STAT_DECODE.search(x)   # no decode: under 1 ms a token, or no expert touched (one prompt token out)
        if m and (float(m.group(1)) < 1.0 or "vram hit 0.00%" in x) and "| prompt " in x:
            x = "glm stat (prompt only): prompt " + x.split("| prompt ", 1)[1]
        picked.append(x.strip())
    return picked


def hip_gpu_report(rocm: Path, bindir=None) -> str:
    """Static AMD details, with KFD's HIP indices/architectures even when no SMI tool is installed.

    Keep the SMI JSON in its own device order: its DRM indices need not match KFD's HIP indices.
    This also retains driver/VRAM fields across the different versions of the two tools.
    On Windows: ROCm's hipInfo (HIP's indices), else Windows' display adapters.
    """
    import platform
    if WIN:
        exe = hip_info(bindir or rocm / "bin")
        gpus = S.amd_gpus(exe)
        lines = [("hipInfo" if exe else "Windows' display adapters (no hipInfo)") + " (HIP GPU indices):"]
        lines += [f"{gpu_label(g)}; driver {g.get('driver', '?')}" for g in gpus] or ["no AMD GPUs found"]
        lines.append(f"Windows {platform.version()}")
        return "\n".join(lines)
    gpus = S.amd_gpus()
    lines = ["KFD topology (HIP GPU indices):"]
    lines += [f"{gpu_label(g)}; driver {g.get('driver', 'amdgpu')}" for g in gpus]
    if not gpus:
        lines.append("no AMD GPUs found in KFD topology")
    try:
        driver = Path("/sys/module/amdgpu/version").read_text().strip()
    except OSError:
        driver = ""
    lines.append(f"amdgpu driver: {driver or 'in-tree'}, kernel {platform.release()}")
    for name, args in (("rocm-smi", ["--showproductname", "--showmeminfo", "vram", "--showdriverversion", "--json"]),
                       ("amd-smi", ["static", "--json"])):
        exe = shutil.which(name)
        if not exe and (rocm / "bin" / name).is_file():
            exe = str(rocm / "bin" / name)
        if not exe:
            continue
        try:
            data = json.loads(S.out([exe, *args]))
        except ValueError:
            continue
        if isinstance(data, (dict, list)) and data:
            lines += [f"\n{name} (tool GPU indices):", json.dumps(data, indent=2)]
            break
    else:
        lines.append("rocm-smi / amd-smi unavailable or did not answer; using KFD topology")
    return "\n".join(lines)


def rocm_report(rocm: Path) -> str:
    """ROCm's install version file first, then hipcc; reporting never requires the compiler."""
    version = ""
    for name in ("version", "version-dev"):
        try:
            version = (rocm / ".info" / name).read_text().strip()
        except OSError:
            continue
        if version:
            break
    if not version:
        hipcc = next((p for p in (rocm / "bin/hipcc", rocm / "bin/hipcc.exe") if p.is_file()), None)
        if hipcc is not None:
            version = S.out([str(hipcc), "--version"]).strip()
    return f"path {rocm}\nversion {version or 'unknown (ROCm version file / hipcc unavailable)'}"


def report(version: str, backend: str | None = None, cfg_path: Path | None = None) -> int:
    """--report: maya-report.txt in the Maya folder - this PC, the installed setup and the engine log's speed lines,
    for a bug or speed report.  Nothing is sent anywhere; the home folder is written as ~ and API keys are left out."""
    home = str(Path.home())
    lines = []
    have = [cfg_path] if cfg_path is not None else configs()
    cfg = {}
    for p in have:
        candidate = read_json(p)
        if backend is None or candidate.get("backend", "cuda") == backend:
            cfg = candidate
            break
    backend = backend or cfg.get("backend", "cuda")
    select_build_backend(backend)
    stamp = read_json(STAMP)

    def add(title, text=""):
        lines.append(f"## {title}")
        lines.extend(str(text).rstrip().replace(home, "~").splitlines() or ["(nothing)"])
        lines.append("")

    git = S.out(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"]).strip()
    add("Maya", f"version {version}, commit {git or '?'}, Python {sys.version.split()[0]}, "
                f"{sys.platform}{' (WSL)' if S.is_wsl() else ''}")
    import platform
    add("System", f"{platform.platform()}")
    if backend == "hip":
        rocm = Path((cfg.get("env") or {}).get("ROCM_PATH") or os.environ.get("ROCM_PATH")
                    or stamp.get("rocm") or "/opt/rocm").expanduser().resolve()
        add("GPUs", hip_gpu_report(rocm, (stamp.get("lib_dirs") or [None])[0] if WIN else None))
        add("ROCm", rocm_report(rocm))
    else:
        add("GPUs", S.out(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,"
                                         "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,"
                                         "temperature.gpu", "--format=csv"]) or "nvidia-smi did not answer")
    cpu, avx2, avx512 = S.cpu_info()
    total, avail = mem_gb()
    pf = S.page_file_gb()
    add("CPU and RAM", f"{cpu}, {os.cpu_count()} threads, {'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'}\n"
                       f"RAM {total:.1f} GB, {avail:.1f} GB available now" +
                       (f", page file {pf:.0f} GB" if pf is not None else ""))
    if WIN:
        disks = S.out(["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Format-Table -AutoSize "
                                                              "FriendlyName,MediaType,BusType,Size | Out-String -Width 200"])
    else:
        disks = S.out(["lsblk", "-d", "-o", "NAME,MODEL,ROTA,TRAN,SIZE"])
    add("Disks", disks)
    keys = ("backend", "archs", "rocm", "date") if backend == "hip" else ("archs", "nvcc", "host_compiler", "date")
    add("Engine build", json.dumps({k: stamp.get(k) for k in keys})
        if stamp else "not compiled yet")
    if not have:
        add("Setup", f"no maya-*.json yet: {ME} has not finished a setup here")
    for c in have:
        cfg = read_json(c)
        args = cfg.get("args") or []
        pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
        ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else "?"
        d = pack.parent if pack else None
        where = ""
        if d is not None and d.exists():
            gb = sum(p.stat().st_size for p in d.glob("*.gguf")) / 1e9
            where = (f"\nmodel folder {d}: {gb:.1f} GB of GGUF, {shutil.disk_usage(d).free / 1e9:.0f} GB free, "
                     f"{'a spinning hard disk' if rotational(d) else 'not a hard disk'}")
        add(f"Setup {c.name}", f"model {(cfg.get('installer') or {}).get('quant', '?')}, context {ctx}, "
                               f"GPUs {cfg.get('gpu')}, images {'on' if cfg.get('vision') else 'off'}, "
                               f"settings {json.dumps(cfg.get('env') or {})}" + where)
        log = Path(cfg.get("log") or c.with_suffix(".log"))
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace").splitlines()
            picked = speed_lines(text)
            add(f"Engine log {log.name}: the speed and memory lines (last 80 of {len(picked)})", "\n".join(picked[-80:]))
            add(f"Engine log {log.name}: the last 25 lines", "\n".join(text[-25:]))
        else:
            add(f"Engine log {log.name}", "not written yet (start Maya once and ask it something)")
    bench_txt = ROOT / "maya-bench.txt"
    if bench_txt.exists():
        add("Benchmark (maya-bench.txt, from --bench)", bench_txt.read_text(encoding="utf-8", errors="replace"))
    out = ROOT / "maya-report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    ok(f"written: {out}")
    say("  Attach this file to your report (GitHub issue, X, or wherever you are asking). It has this PC's hardware,")
    say("  your Maya setup and the engine's speed lines - no API key, and your home folder is shown as ~.")
    say("  Best: run it right after a slow answer, so the log has that answer's numbers.")
    return 0


# ------------------------------------------------------------------------------------------------ the benchmark
# the same three questions on every machine (greedy, thinking off), after one warm-up answer, so speeds compare
BENCH_TOPICS = [
    "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.",
    "Explain how photosynthesis works, step by step, for a high-school student.",
    "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.",
]


def bench(cfg_path: Path, version: str) -> int:
    """--bench: a standard speed test of the installed model on this PC - decode (writing an answer) on three
    questions and prefill (reading a prompt) at 2k and 8k tokens of this folder's docs - through the engine alone, the
    way the dashboard starts it.  Writes maya-bench.txt (--report includes it).  A few minutes."""
    import urllib.request
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    if cfg.get("backend") == "hip":
        cfg["exe"] = str(EXE)
    port = cfg.get("port") or 8080
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        fail(f"Maya is running (port {port}): the benchmark needs the GPU memory it holds",
             "stop it (Ctrl+C in its window), then run --bench again")
    except OSError:
        pass
    sys.path.insert(0, str(ROOT))
    import serve.server as SV                      # the server's own engine command and environment
    import strata_tokenizer as ST
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    tp = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
    if not (tp / "vocab.json").exists():
        fail(f"{cfg_path.name}: the tokenizer is missing ({tp})", f"run {ME} --setup to repair it")
    tok = ST.Tokenizer.from_dir(tp)
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) +
                       sorted((ROOT / "bench" / "results").glob("*.md")) if p.exists())
    doc = tok.encode(text)
    lens = [n for n in (2048, 8192) if n + 64 <= ctx and n <= len(doc)]
    log_path = ROOT / "maya-bench.log"
    step(1, f"benchmark: {cfg_path.name} (loading the model first - a minute or a few)")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=SV.child_env(cfg),
                             cwd=cfg.get("cwd") or str(ROOT), bufsize=1)

        def gen(ids, n_new):
            p.stdin.write(f"GEN {n_new} temperature=0 " + ",".join(map(str, ids)) + "\n")
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if not line:
                    fail("the engine stopped during the benchmark", f"its log: {log_path}")
                if line.startswith("DONE") or line.startswith("ERR"):
                    return line.split()

        while True:
            line = p.stdout.readline()
            if not line:
                fail("the engine stopped while loading", f"its log: {log_path}")
            if line.startswith("READY"):
                break
        ok(f"loaded in {time.time() - t0:.0f} s")
        chat = lambda q: tok.encode("[gMASK]<sop><|user|>\n" + q + "<|assistant|>\n</think>", parse_special=True)
        gen(chat("Say hello in five languages."), 64)          # warm-up: the first answer after a start is slower
        results = []
        for q in BENCH_TOPICS:
            f = gen(chat(q), 256)
            if f[0] == "DONE" and float(f[4]) > 0:
                results.append(("decode", q, int(f[1]) / float(f[4]) * 1000.0))
                say(f"  decode  {results[-1][2]:6.1f} tokens/s   {q[:60]}")
        if lens:                                            # warm-up: the first prompt after a start is slower
            gen(doc[-lens[0]:], 1)                          # (cold caches, issue #8); its opening is not reused below
        for i, n in enumerate(lens):
            f = gen(doc[i * 997:i * 997 + n], 1)            # different openings: nothing reused between them
            if f[0] == "DONE" and float(f[3]) > 0:
                results.append(("prefill", f"{n} tokens", n / float(f[3]) * 1000.0))
                say(f"  prefill {results[-1][2]:6.0f} tokens/s   a {n}-token prompt")
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    stats = speed_lines(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
    dec = [r[2] for r in results if r[0] == "decode"]
    found = S.amd_gpus() if cfg.get("backend") == "hip" else S.gpus()
    if cfg.get("backend") == "hip" and SV.gpu_list(cfg):
        by_index = {g["index"]: g for g in found}
        found = [by_index[i] for i in SV.gpu_list(cfg) if i in by_index]
    gpus = ", ".join(f"{g['name']} {g['vram_gb']:.0f} GB" for g in found) or "?"
    total, _ = mem_gb()
    lines = [f"Project Maya v{version} benchmark, {time.strftime('%Y-%m-%d %H:%M')}",
             f"GPUs: {gpus}; RAM {total:.0f} GB; CPU {S.cpu_info()[0]}",
             f"setup: {cfg_path.name}, context {ctx}, GPUs {cfg.get('gpu')}",
             f"decode (writing the answer): mean {sum(dec) / max(1, len(dec)):.1f} tokens/s over {len(dec)} answers of "
             f"256 tokens (greedy, thinking off)"]
    lines += [f"  {r[2]:6.1f} tokens/s  {r[1]}" for r in results if r[0] == "decode"]
    lines += [f"prefill (reading the prompt): {r[2]:.0f} tokens/s, {r[1]}" for r in results if r[0] == "prefill"]
    lines += ["", "engine lines:"] + [x.replace(str(Path.home()), "~") for x in stats[-40:]]
    out = ROOT / "maya-bench.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    for x in lines[:4 + len(results)]:
        say("  " + x)
    ok(f"written: {out} - attach it (with {ME} --report) to a speed report")
    return 0


# ------------------------------------------------------------------------------------------------ the setup
TUI_WHEELS = ROOT / "third_party" / "wheels"       # the setup's screen: Textual (MIT) and what it needs, pure Python


def setup_screen(a, asks: bool = True) -> bool:
    """Whether the setup - or Maya, `asks` False: nothing to answer - runs on a screen of its own (tools/setup_tui.py):
    in a terminal, not with --plain (nor a setup with --yes: nobody there to answer).  Its library comes with Maya
    (TUI_WHEELS) and goes into .venv from there the first time: nothing is downloaded.  False: plain text, as before."""
    if getattr(a, "plain", False) or (asks and getattr(a, "yes", False)) or os.environ.get("TERM") == "dumb" or \
            not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    wheel = max(TUI_WHEELS.glob("textual-*.whl"), default=None)
    if wheel is None:
        return False
    want = wheel.name.split("-")[1]
    from importlib import invalidate_caches, metadata
    try:
        if metadata.version("textual") == want:
            return True
    except metadata.PackageNotFoundError:
        pass
    if sys.prefix == sys.base_prefix:                  # not Maya's .venv: nothing goes into a system Python
        return False
    say("  Preparing the setup's screen (Textual, from third_party/wheels: nothing is downloaded) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-index",
                        "--find-links", str(TUI_WHEELS), f"textual=={want}"], capture_output=True, text=True)
    if r.returncode != 0:
        why = (r.stderr or r.stdout).strip().splitlines()
        warn("the setup's screen could not be installed, so the setup runs in plain text" +
             (f" ({why[-1]})" if why else ""))
        return False
    invalidate_caches()
    return True


def set_up(a, prev: dict) -> Path | None:
    """Steps 1-8, then the tuning: the config written, or None when the model was not downloaded.  `prev`: the
    config of the setup before (its answers are the defaults)."""
    inst = prev.get("installer") or {}
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    pc = check_pc(a)                                   # 1
    step(2, "your choices")                            # 2
    ctx = choose_context(a, prev_ctx)
    ok(f"context: {ctx} tokens")
    # the models folder: --models-dir, else the one set up before (a config from before the option has its data
    # folder, whose models/ held the downloads), else Maya-data/models next to this folder
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else
              Path(inst["data_dir"]) / "models" if inst.get("data_dir") else ROOT.parent / "Maya-data" / "models")
    choice = choose_model(a, models, inst)
    if choice[0] == "download":
        ok(f"model: {choice[1]}, in {download_dir(models, choice[1])}")
    else:
        ok(f"model: {choice[1]} (yours)")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama)                    # 4
    got = model_step(a, models, choice)                # 5
    if got is None:
        return None
    d, shards, quant = got
    pack = pack_step(a, d, shards, llama)              # 6
    vision = vision_step(a, pc, meta, llama, d, quant)  # 7
    cfg_path = write_config(a, pc, meta, pack, quant, ctx, models, vision, choice)   # 8
    # the tuning: asked for (--calibrate), or offered when nothing was tuned for this PC and model yet and someone
    # answers (--yes and --no-start setups are not held up by it)
    tuned = None
    cfg = read_json(cfg_path)
    offer = (cfg.get("backend") != "hip" and saved_calibration(cfg, any_context=True) is None and not a.yes and
             not a.no_start)
    if a.calibrate or (offer and ask(
            "Tune Maya for this PC now? It measures the split of the work between the CPU and the PCIe link and the "
            "CPU threads (about 10-15 minutes; later: ./maya.sh --calibrate)", ["y", "n"], "y", a.yes) == "y"):
        if S.UI is not None:                           # (the screen lists the tuning as a step of its own)
            S.UI.step(9, "tuning Maya for this PC")
        tuned = calibrate_config(cfg_path)
    if tuned is False:
        say()
        warn("this PC is NOT tuned (the reason is above): ./maya.sh --calibrate tries again")
    return cfg_path


def set_up_and_serve(a, prev: dict) -> tuple | None:
    """On the setup's screen: the setup, then Maya running on the same screen - its server's output there, Ctrl+C
    stops it.  (The config, the server's exit code or None when it was not started), or None when the model was not
    downloaded."""
    cfg_path = set_up(a, prev)
    if cfg_path is None or a.no_start:
        return cfg_path and (cfg_path, None)
    return cfg_path, serve_on_screen(cfg_path, a)


# ------------------------------------------------------------------------------------------------ main
def maya_version() -> str:
    return (HERE / "VERSION").read_text(encoding="utf-8").strip() if (HERE / "VERSION").exists() else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cuda", "hip"], help="GPU backend (HIP: experimental gfx1100/gfx1201/gfx1151 on Linux and Windows)")
    ap.add_argument("--setup", action="store_true", help="set up again instead of starting the installed model")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--no-start", action="store_true", help="set up, but do not start the dashboard")
    ap.add_argument("--yes", action="store_true",
                    help="take the recommended answers (the model download still needs --download-model)")
    ap.add_argument("--plain", action="store_true",
                    help="the setup in plain text, the answers typed (not on the setup's own screen)")
    ap.add_argument("--download-model", action="store_true",
                    help="download the model without asking (the commands and the size are still printed)")
    ap.add_argument("--model", choices=list(MODELS), help=f"which download (default: asked, {next(iter(MODELS))} "
                                                          "recommended)")
    ap.add_argument("--no-vision", action="store_true", help="text only: no image encoder (saves its build and "
                                                             "its 1.1 GB of files)")
    ap.add_argument("--gguf-dir", help="use GLM-5.3-Flash GGUF files you already have: the folder with all of them, "
                                       "or the .gguf file (the first one of a split model) when the folder holds "
                                       "several models; it must be writable - the pack is written inside it "
                                       "(without this option the setup lists the ones it finds in ~/models*)")
    ap.add_argument("--models-dir", help="where downloaded models go, each in a folder named after it, e.g. "
                                         "<DIR>/Maya-L/ (default: Maya-data/models next to this folder); use "
                                         "a fast NVMe SSD with ~100 GB free")
    ap.add_argument("--gpu", type=int, help="run on this one GPU (CUDA: nvidia-smi; HIP: KFD topology order, on Windows hipInfo's)")
    ap.add_argument("--gpus", help=f"split the model's layers across these GPUs (up to {MAX_GPUS}), e.g. 0,1 or "
                                   "0,1,2,3")
    ap.add_argument("--context", type=int, help=f"context length in tokens (default {DEFAULT_CONTEXT})")
    ap.add_argument("--port", type=int, help="the dashboard's and the API's port (default 8080)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this machine only (default), 0.0.0.0 = also "
                                   "other devices on your network (set --api-key too)")
    ap.add_argument("--api-key", help="require this key from API clients and the dashboard")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an engine setting kept in the config, e.g. STRATA_GLM_RAM_HEADROOM_GB=4 (README-MAYA.md)")
    ap.add_argument("--nvcc", help="the CUDA toolkit's nvcc to build with (default: the newest that fits your GPUs)")
    ap.add_argument("--host-compiler", help="the C++ compiler nvcc uses, e.g. g++-12 (when the default g++ is newer "
                                            "than your CUDA accepts)")
    ap.add_argument("--llama-dir", help="a llama.cpp checkout at the pinned commit, instead of downloading its source")
    ap.add_argument("--rebuild", action="store_true", help="compile the engine again")
    ap.add_argument("--repack", action="store_true", help="build the pack again")
    ap.add_argument("--report", action="store_true", help="write maya-report.txt - this PC, the setup and the engine's "
                                                          "speed lines - to attach when you report a problem or a "
                                                          "speed (nothing is sent anywhere)")
    ap.add_argument("--bench", action="store_true", help="a standard speed test of the installed model (a few "
                                                         "minutes, with Maya stopped): decode on three questions and "
                                                         "prefill at 2k / 8k tokens; writes maya-bench.txt")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's CPU lane for this PC (the PCIe share and the CPU threads, ~10-15 minutes), "
                         "then start the model (with --no-start: only tune)")
    ap.add_argument("--config", type=Path, help="the installed JSON config to start (its run-maya-<model>.sh does), or "
                                              "for --bench or --report (default: most recently used)")
    a = ap.parse_args()
    if a.config is not None:
        if not a.config.is_file():
            ap.error(f"config not found: {a.config}")
        if a.backend and read_json(a.config).get("backend", "cuda") != a.backend:
            ap.error(f"{a.config}: config backend does not match --backend {a.backend}")
    if not WIN and sys.prefix == sys.base_prefix and not os.environ.get("MAYA_SH"):
        # started as `python3 maya.py`: a system Python takes no pip installs (PEP 668, "externally-managed-
        # environment") - maya.sh makes the private .venv and runs this file with its Python
        say("Starting through ./maya.sh, which runs Maya with its own Python environment (.venv) ...")
        sys.stdout.flush()
        os.execve("/bin/sh", ["/bin/sh", str(HERE / "maya.sh"), *sys.argv[1:]], dict(os.environ, MAYA_SH="1"))
    version = maya_version()
    say(f"Project Maya v{version} - GLM-5.3-Flash on your own GPU(s). Built on Strata (MIT) and ggml/llama.cpp "
        "(MIT).")
    if a.report:
        return report(version, a.backend, a.config)
    if a.bench:
        have = [a.config] if a.config is not None else configs()
        if a.backend:
            have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
        if not have:
            fail("Maya is not set up here yet", f"run {ME} first")
        return bench(have[0], version)

    have = configs() if a.config is None else [a.config.resolve()]   # (--config: that one, no question)
    if a.backend:
        have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
    else:
        a.backend = "hip" if have and read_json(have[0]).get("backend") == "hip" else "cuda"
    setting_up = a.setup or a.check or a.gguf_dir or a.model or a.rebuild or a.repack or a.download_model
    if have and not setting_up and (a.calibrate or not a.no_start):
        pick = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {c.name}")
            pick = have[int(ask("Which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        if a.calibrate:
            refresh_engine(read_json(pick))            # (the engine the tuning measures is the current one)
            if not calibrate_config(pick):
                say()
                warn("this PC is NOT tuned (the reason is above): the model " +
                     ("keeps" if a.no_start else "starts with") + " the engine's own settings")
            if a.no_start:
                return 0
        return start(pick, a)
    if not have and a.calibrate:
        say("Nothing is set up yet: the setup runs first, then the tuning.")
    prev = read_json(have[0]) if have else {}
    if a.check:
        pc = check_pc(a)
        say()
        if a.backend == "hip":
            pick = (f"--gpus {','.join(str(g['index']) for g in pc['gpus'])}" if len(pc["gpus"]) == 2
                    else f"--gpu {pc['gpus'][0]['index']}")
            say(f"HIP prerequisites found. Run {ME} --backend hip {pick} to try the experimental port.")
        else:
            say(f"This PC can run Maya. Run {ME} without --check to set it up.")
        return 0
    rc = None                                          # the server's exit code, when it ran on the setup's screen
    if setup_screen(a):
        import setup_tui
        cfg_path, rc = setup_tui.run(lambda: set_up_and_serve(a, prev), version, ROOT / "maya-setup.log",
                                     ME) or (None, None)
    else:
        cfg_path = set_up(a, prev)
    if cfg_path is None:                               # the model was not downloaded (what to do next is above)
        return 0
    if rc is not None:
        return after_server(rc, a)
    if a.no_start:
        say()
        say(f"All set. Start it with {ME} (or " + (f"run-{cfg_path.stem}.bat" if WIN else f"./run-{cfg_path.stem}.sh")
            + ").")
        return 0
    return start(cfg_path, a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
