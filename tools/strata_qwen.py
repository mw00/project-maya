#!/usr/bin/env python3
"""tools/strata_qwen.py - run Qwen3.8-Flash-Next on Strata's current engine from Maya's server (opt-in).

Maya's `maya` engine forked Strata's at engine 0.1.27 (2026-09-29).  Strata has since gone on to 0.1.67 (sm_86 exact
speed switches, the profile-ranked cache fill, the device-side hit graph, 32K prompt chunks, DFlash2 drafting, ...).  The
engine still speaks the same serve protocol (READY / GEN / T / PP / DONE / STOP / QUIT), so Maya's server drives it as it
is.  This tool:

  1. fetches Strata's engine source at a pinned commit (engine/ only) into third_party/strata,
  2. applies tools/strata_qwen/maya-metrics.patch - the lines Maya's Monitor reads that Strata's engine does not write:
       INFO engine_kind=qwen ... mtp= dflash=   (the "Where the experts live" card, the Speculation line)
       VRAM dev=0 ... weights= kv= drafter= window= pool= ...   (the "What fills the GPU" map, measured at the
                                                                  engine's own load steps)
       STAT tok_s= vram_hit= ram_fetch= promo= ...   (the live tier counters, every 8 tokens, as maya's Qwen path)
  3. builds it for this PC's GPU(s) (the archs of build/MAYA-BUILD.json, or --arch) into build/strata-qwen/,
  4. writes a Maya config from an existing Qwen config (--from): the same model files and settings, with this engine.

    python tools/strata_qwen.py --from maya-qwen-ud-iq2_xs.json --out maya-qwen-strata.json
    python tools/strata_qwen.py --from maya-qwen-ud-iq2_xs.json --out maya-qwen-strata.json \\
        --dflash2 /models/qwen3.8-flash-next-dflash2.q4_0-selq4.pack
    ./maya.sh --config maya-qwen-strata.json

Nothing else changes: the old config and engine stay as they were.  Measured on one PC (RTX 3090 24 GB, Ryzen 9 5900X,
94 GB DDR4, Linux, Qwen3.8-Flash-Next GSQ-RCO IQ3_S, 262K context, int8 KV, `strata bench run --suite standard-v1`):
see docs/QWEN_STRATA_ENGINE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Niko1221/Strata.git"
COMMIT = "d46f5f268f8ed797e83a5c63a69bffbd7a581cf6"   # engine 0.1.67 (Strata 0.2.21, 2026-11-24)
ENGINE_VERSION = "0.1.67"
PATCH = ROOT / "tools" / "strata_qwen" / "maya-metrics.patch"
SRC = ROOT / "third_party" / "strata"
BUILD = ROOT / "build" / "strata-qwen"


def run(cmd, cwd=None, env=None):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def fetch(commit: str) -> None:
    """Strata's engine/ at `commit` in third_party/strata, with the patch applied once."""
    if not (SRC / ".git").is_dir():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO, SRC])
        run(["git", "sparse-checkout", "set", "engine"], cwd=SRC)
    def git(*a):
        return subprocess.run(["git", *a], cwd=SRC, capture_output=True, text=True).stdout.strip()
    # done already: the patch's commit on top of `commit`
    if git("log", "-1", "--format=%s").startswith("Maya integration") and git("rev-parse", "HEAD~1").startswith(commit):
        return
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        sys.exit("strata_qwen: --commit takes a full 40-character commit hash (a branch or tag could move)")
    run(["git", "fetch", "--filter=blob:none", "origin", commit], cwd=SRC)
    run(["git", "checkout", "-q", "--detach", commit], cwd=SRC)
    if git("rev-parse", "HEAD") != commit:
        sys.exit(f"strata_qwen: third_party/strata is not at {commit} after the checkout")
    run(["git", "-c", "user.name=maya", "-c", "user.email=maya@localhost", "am", "-q", PATCH], cwd=SRC)


def tool(name: str) -> str:
    """cmake / ninja: on PATH, else the ones setup put in Maya's .venv."""
    found = shutil.which(name)
    if found:
        return found
    for d in (ROOT / ".venv" / "bin", Path(sys.executable).parent):   # (the venv this tool runs in, too)
        if (d / name).exists():
            return str(d / name)
    sys.exit(f"strata_qwen: {name} not found (./maya.sh --setup installs it into .venv)")


def build(archs: list[int], nvcc: str | None, commit: str) -> Path:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(tool("ninja")).parent), env.get("PATH", "")])
    cmake = tool("cmake")
    arch = ";".join(f"{a}-real" for a in archs)
    nvcc = nvcc or shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)
    if not nvcc or not Path(nvcc).exists():
        sys.exit("strata_qwen: no CUDA compiler (nvcc) found - this engine is built for NVIDIA GPUs only; pass --nvcc")
    cfg = [cmake, "-G", "Ninja", "-S", SRC / "engine", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release",
           "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_EXPERIMENTAL_SM60=OFF",
           "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_PORTABLE=ON", "-DSTRATA_Q6K_EXPERTS=ON",
           f"-DCMAKE_CUDA_ARCHITECTURES={arch}"]   # (Strata's own release flags: tools/packaging/build_cuda_linux.sh)
    cfg.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    run(cfg, env=env)
    run([cmake, "--build", BUILD, "--target", "strata", "-j", str(os.cpu_count() or 4)], env=env)
    exe = BUILD / "strata"
    # the server reads the version beside the exe (StrataEngine: BUILD.json)
    (BUILD / "BUILD.json").write_text(json.dumps({"version": ENGINE_VERSION + "-maya", "strata_commit": commit,
                                                  "archs": ";".join(map(str, archs))}) + "\n")
    return exe


def put(args: list, key: str, value: str | None = None) -> None:
    if key in args:
        if value is not None:
            args[args.index(key) + 1] = value
    else:
        args.extend([key] if value is None else [key, value])


def write_config(src: Path, out: Path, exe: Path, dflash2: Path | None) -> None:
    c = json.loads(src.read_text())
    a = list(c.get("args") or [])
    c["exe"] = str(exe)
    # keep the conversations' checkpoints spread back from the end: an agent that rewrites its history N tokens back
    # reads about N tokens again, not the whole conversation (Strata's own launcher sets it)
    put(a, "--prompt-cache-spread")
    if dflash2 is not None:
        put(a, "--draft", "dflash2")
        put(a, "--draft-model", str(dflash2))
    # (the config's env stays as it is: Strata's engine reads STRATA_* and the server exports each setting under both
    #  its MAYA_ and STRATA_ names - tools/maya_names.py)
    c["args"] = a
    c["log"] = str(out.with_suffix(".log").resolve())
    out.write_text(json.dumps(c, indent=1) + "\n")
    print(f"strata_qwen: wrote {out} - start it with ./maya.sh --config {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", type=Path, help="an existing Qwen3.8-Flash-Next config to copy")
    ap.add_argument("--out", type=Path, default=Path("maya-qwen-strata.json"), help="the config to write")
    ap.add_argument("--commit", default=COMMIT, help="the Strata commit to build (default: engine %s)" % ENGINE_VERSION)
    ap.add_argument("--arch", default=None, help="CUDA archs, e.g. 86 or 86,89 (default: build/MAYA-BUILD.json's)")
    ap.add_argument("--nvcc", default=None, help="the nvcc to use (default: build/MAYA-BUILD.json's, else PATH)")
    ap.add_argument("--dflash2", type=Path, default=None, help="draft with this DFlash2 pack instead of the MTP layer")
    ap.add_argument("--no-build", action="store_true", help="only write the config (the engine is built already)")
    o = ap.parse_args()
    stamp = {}
    try:
        stamp = json.loads((ROOT / "build" / "MAYA-BUILD.json").read_text())
    except (OSError, ValueError):
        pass
    raw = o.arch.replace(";", ",").split(",") if o.arch else [str(x) for x in stamp.get("archs") or []]
    if not raw or not all(x.strip().isdigit() for x in raw):
        sys.exit("strata_qwen: no NVIDIA CUDA arch known (this engine is built for NVIDIA GPUs only) - pass --arch "
                 "(75 for RTX 20, 86 for RTX 30, 89 for RTX 40, 120 for RTX 50)")
    archs = [int(x) for x in raw]
    if o.src:   # (the config's checks first: before a build of several minutes)
        cfg = json.loads(o.src.read_text())
        args = cfg.get("args") or []
        if "--ple-gguf" not in args or "--native" not in args:
            sys.exit(f"strata_qwen: {o.src} is not a Qwen3.8-Flash-Next config (no --native / --ple-gguf)")
        if o.dflash2 is not None and not o.dflash2.is_file():
            sys.exit(f"strata_qwen: {o.dflash2} not found")
        if o.dflash2 is not None and "--mtp" not in args:
            sys.exit("strata_qwen: --dflash2 reads the draft vocabulary from the config's --mtp folder, which it has not")
    exe = BUILD / "strata"
    if not o.no_build:
        fetch(o.commit)
        exe = build(archs, o.nvcc or stamp.get("nvcc"), o.commit)
    elif not exe.is_file():
        sys.exit(f"strata_qwen: --no-build, but {exe} is not built yet")
    if o.src:
        write_config(o.src, o.out, exe, o.dflash2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
