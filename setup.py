#!/usr/bin/env python3
"""The helpers Project Maya's installer (maya.py) uses, from Strata's setup (MIT): the PC checks (GPUs, CPU,
RAM, the page file), the Python packages, llama.cpp's source at the pinned commit, resumable downloads, the
engine's source hash, the shards check and the console messages.  Importing it runs nothing; ./setup.sh and
START-MAYA.bat start maya.py."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]


# ------------------------------------------------------------------------------------------------ output
# A screen of the setup's own while it runs (Project Maya's tools/setup_tui.py): these functions hand it what they
# would print, ask and run.  Each of its methods returns something false when it did not take it (the screen has
# closed): then it is printed as without it.
UI = None


def say(msg=""):
    if UI is not None and UI.say(msg):
        return
    print(msg, flush=True)


def step(n, title):
    if UI is not None and UI.step(n, title):
        return
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    if UI is not None and UI.ok(msg):
        return
    say(f"  [ok] {msg}")


def warn(msg):
    if UI is not None and UI.warn(msg):
        return
    say(f"  [!]  {msg}")


STOPPED = "Setup stopped. Fix the item above and run it again - everything already done is kept and skipped."


def fail(msg, hint=None, end=STOPPED):
    if UI is not None:
        UI.fail(msg, hint)             # shown until it is read; the setup ends there (the lines below follow it)
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    if end:
        say("\n" + end)
    sys.exit(1)


def progress(msg=None, frac=None):
    """A line that rewrites itself (a download's progress; frac: the share done, when known); None ends it."""
    if UI is not None and UI.progress(msg, frac):
        return
    print("\n" if msg is None else f"\r  {msg}   ", end="", flush=True)


def ask(question, choices, default, yes):
    if yes:
        return default
    if UI is not None:
        answer = UI.ask(question, choices, default)
        if answer is not None:
            return answer
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False, echo=True):
    if echo:
        say("  > " + " ".join(str(c) for c in cmd))
    # the setup's screen shows the output as it comes (all of it: a quiet command's too) and returns no stdout
    r = UI.run([str(c) for c in cmd], cwd=cwd, env=env) if UI is not None else None
    if r is None:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                           stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                           text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the PC
def _memory_status():
    """Windows' GlobalMemoryStatusEx: RAM, and the commit limit (ullTotalPageFile = RAM + page file)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def ram_gb():
    if WIN:
        return _memory_status().ullTotalPhys / 2**30
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def page_file_gb():
    """The page file's current size (GB) on Windows, None elsewhere.  The graphics card's memory needs room there
    too: under Windows' driver model every allocation on the card is also charged to the commit (RAM + page file),
    so with the page file off or tiny the engine cannot use the free VRAM (issue #60)."""
    if not WIN:
        return None
    m = _memory_status()
    return max(0.0, (m.ullTotalPageFile - m.ullTotalPhys) / 2**30)


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:
        pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
        avx2 = bool(pf(40)) or _cpuid_avx2()      # PF_AVX2_INSTRUCTIONS_AVAILABLE, else the CPU itself (#159)
        n = out(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        name = n or name
        avx512 = bool(pf(41)) and _cpuid_avx512_full()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
            avx2 = "avx2" in flags
            avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
            m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
            name = m.group(1) if m else name
        except OSError:
            pass
    return name, avx2, avx512


def _cpuid_avx512_full() -> bool:
    """Windows has no feature bit for VNNI / VBMI: ask the CPU (CPUID leaf 7) through a tiny machine-code stub."""
    try:
        code = bytes([0x53, 0x49, 0x89, 0xC8, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x31, 0xC9, 0x0F, 0xA2,   # push rbx; r8=rcx; cpuid(7,0)
                      0x41, 0x89, 0x18, 0x41, 0x89, 0x48, 0x04, 0x5B, 0xC3])                   # [r8]=ebx,[r8+4]=ecx; pop rbx
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
        if not buf:
            return False
        ctypes.memmove(buf, code, len(code))
        regs = (ctypes.c_uint32 * 2)()
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(buf)(ctypes.addressof(regs))
        ebx, ecx = regs[0], regs[1]
        need_ebx = (1 << 16) | (1 << 30) | (1 << 31)                   # F, BW, VL
        need_ecx = (1 << 1) | (1 << 11)                                # VBMI, VNNI
        return (ebx & need_ebx) == need_ebx and (ecx & need_ecx) == need_ecx
    except Exception:
        return False


def _run_stub(code: bytes, *args) -> None:
    """Runs a few bytes of x64 machine code (Windows calling convention: the arguments in rcx, rdx)."""
    k32 = ctypes.windll.kernel32
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
    if not buf:
        raise OSError("VirtualAlloc failed")
    try:
        ctypes.memmove(buf, code, len(code))
        ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * len(args))(buf)(*args)
    finally:
        k32.VirtualFree(buf, 0, 0x8000)


def _cpuid_avx2() -> bool:
    """AVX2 asked from the CPU (CPUID leaf 7 EBX bit 5), with the OS saving the YMM registers (OSXSAVE + XCR0):
    Windows' IsProcessorFeaturePresent(PF_AVX2) says no on some PCs whose CPU has it (a Ryzen 9 3950X, #159)."""
    try:
        def cpuid(leaf):
            regs = (ctypes.c_uint32 * 4)()
            _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                             0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                      ctypes.addressof(regs), leaf)
            return list(regs)
        if cpuid(0)[0] < 7:
            return False
        ecx1 = cpuid(1)[2]
        if not (ecx1 >> 27) & 1 or not (ecx1 >> 28) & 1:             # OSXSAVE, AVX
            return False
        xcr0 = (ctypes.c_uint32 * 2)()
        _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                        # r8=rcx; ecx=0; xgetbv
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
        if xcr0[0] & 6 != 6:                                           # the OS saves XMM and YMM
            return False
        return bool((cpuid(7)[1] >> 5) & 1)
    except Exception:
        return False


def gpus():
    """Every NVIDIA GPU, numbered as nvidia-smi numbers them (by PCI bus, the order the engine is told to use)."""
    s = out(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc, drv = [x.strip() for x in line.split(",")]
            found.append({"index": int(idx), "name": name, "vram_gb": float(mem) / 1024.0, "arch": cc.replace(".", ""),
                          "driver": drv})
        except ValueError:
            continue
    return found


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
                                                        # prompt buffers too (docs/MULTI_GPU.md)


def gpu_info(pick=None):
    """The GPU Strata runs on: `pick` (nvidia-smi's number) if given, else the one with the most VRAM (ties: the
    lower number).  None when there is no NVIDIA GPU.  The dict also says how many there are ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def find_vcvars(cuda=True):
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    # CUDA 13 accepts Visual Studio 2019 and 2022 only: a newer one (2026 = version 18) installed next to them
    # must not be picked ("unsupported Microsoft Visual Studio version"); with only a newer one there is none.
    # ROCm's clang (the HIP build: cuda=False) takes a newer one too, 2019/2022 first
    for versions in ([["-version", "[16.0,18.0)"]] if cuda else [["-version", "[16.0,18.0)"], []]):
        p = out([str(vswhere), "-latest", "-products", "*", *versions, "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]).strip()
        v = Path(p) / "VC/Auxiliary/Build/vcvars64.bat" if p else None
        if v and v.exists():
            return v
    return None


# ------------------------------------------------------------------------------------------------ downloads
def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check your internet connection and run it again")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        progress(f"{what or dst.name}: {size}", have / total if total else None)
            progress()
            if not total or have >= total:
                break
        except OSError as e:
            progress()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    mark(dst)
    ok(f"{what or dst.name} downloaded")


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git."""
    llama = ROOT / "third_party" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = ROOT / "third_party" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    # PR #63: on Windows a rename can fail with PermissionError while an antivirus scanner still holds a file of the
    # fresh unpack; shutil.move falls back to copy-and-delete, and a few retries let the scanner finish.  The target
    # is `llama` itself - moving into its parent would keep the zip's `llama.cpp-<sha>` folder name.
    for attempt in range(5):
        try:
            shutil.move(str(top), str(llama))
            break
        except PermissionError:
            if attempt == 4:
                raise
            shutil.rmtree(llama, ignore_errors=True)   # a partial copy from the failed attempt
            time.sleep(2)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text()) if stamp.exists() else []
    need = [p for p in packages if p not in have]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


AMD_NAMES = {"gfx1100": "AMD Radeon RX 7900 series (gfx1100)",
             "gfx1201": "AMD Radeon RX 9070 / AI PRO R9700 (gfx1201)"}   # when sysfs has no product name
# Windows without ROCm's hipInfo yet: the architecture from the adapter's PCI device ID (Navi 31, Navi 48, the
# Strix Halo / Gorgon Halo APU), else from its name ("AMD Radeon(TM) 8060S Graphics", 8050S, 8065S, ...)
AMD_DEVICE_ARCHS = {"744c": "gfx1100", "7550": "gfx1201", "7551": "gfx1201", "1586": "gfx1151"}
AMD_NAME_ARCHS = ((r"Radeon(?:\s*\(TM\))?\s+80[4-6]\dS", "gfx1151"), (r"RX\s*7900", "gfx1100"),
                  (r"RX\s*9070|R9700", "gfx1201"))


def parse_hipinfo(text: str) -> list:
    """ROCm's hipInfo output as amd_gpus() lists GPUs, in HIP's order (its "device#" lines)."""
    found = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("device#"):
            num = re.search(r"\d+", s)
            found.append({"index": int(num.group()) if num else len(found), "name": "", "vram_gb": 0.0, "arch": "",
                          "driver": "windows", "vendor": "amd"})
            continue
        key, _, val = s.partition(":")
        key, val = key.strip(), val.strip()
        if not found or not val:
            continue
        g = found[-1]
        if key == "Name":
            g["name"] = val
        elif key == "gcnArchName":
            g["arch"] = val.split(":")[0]               # gfx1151, or gfx90a:sramecc+:xnack- -> gfx90a
        elif key == "totalGlobalMem":
            m = re.match(r"([\d.]+)\s*([KMGT]?B)?", val, re.I)
            if m:
                scale = {"KB": 2**-20, "MB": 2**-10, "GB": 1, "TB": 2**10}.get((m.group(2) or "").upper(), 2**-30)
                g["vram_gb"] = float(m.group(1)) * scale
        elif key in ("isIntegrated", "integrated"):
            g["integrated"] = val not in ("0", "false", "False")
    for g in found:
        g["name"] = g["name"] or f"AMD Radeon ({g['arch'] or 'unknown'})"
    return [g for g in found if g["arch"]]


def parse_display_adapters(text: str) -> list:
    """Windows' AMD display adapters (the registry's display class as JSON, from windows_amd_adapters), numbered in
    its order - HIP's own with one AMD GPU."""
    try:
        rows = json.loads(text) if text.strip() else []
    except ValueError:
        return []
    found = []
    for r in rows if isinstance(rows, list) else [rows]:
        if not isinstance(r, dict):
            continue
        name = str(r.get("DriverDesc") or "AMD Radeon")
        dev = re.search(r"dev_([0-9a-f]{4})", str(r.get("MatchingDeviceId") or ""), re.I)
        arch = AMD_DEVICE_ARCHS.get(dev.group(1).lower(), "") if dev else ""
        arch = arch or next((a for pat, a in AMD_NAME_ARCHS if re.search(pat, name, re.I)), "unknown")
        try:
            vram = int(r.get("HardwareInformation.qwMemorySize") or 0) / 2**30
        except (TypeError, ValueError):
            vram = 0.0
        found.append({"index": len(found), "name": name, "vram_gb": vram, "arch": arch,
                      "driver": str(r.get("DriverVersion") or "windows"), "vendor": "amd", "source": "registry"})
    return found


def windows_amd_adapters() -> list:
    # the registry's display class has each adapter's full memory size (WMI's AdapterRAM stops at 4 GB)
    ps = ("Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}"
          "\\0*' -ErrorAction SilentlyContinue | Where-Object { $_.MatchingDeviceId -match 'ven_1002' } | "
          "Select-Object DriverDesc, MatchingDeviceId, DriverVersion, 'HardwareInformation.qwMemorySize' | "
          "ConvertTo-Json -Compress")
    return parse_display_adapters(out(["powershell", "-NoProfile", "-Command", ps]))


def amd_gpus(hipinfo=None):
    """AMD GPUs from the kernel's KFD topology (the amdgpu driver; no ROCm needed), numbered as HIP numbers them:
    the GPU nodes in order, the CPU nodes skipped.  Integrated GPUs are listed too (not supported).  On Windows:
    ROCm's hipInfo (`hipinfo`, its path), else Windows' AMD display adapters (windows_amd_adapters)."""
    if WIN:
        return (parse_hipinfo(out([str(hipinfo)])) if hipinfo else []) or windows_amd_adapters()
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    found = []
    if not base.is_dir():
        return found
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = v.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(f"/sys/class/drm/renderD{props.get('drm_render_minor', '')}/device")
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        if name == f"AMD Radeon ({arch})" and arch in AMD_NAMES:
            name = AMD_NAMES[arch]
        found.append({"index": len(found), "name": name, "vram_gb": vram, "arch": arch, "driver": "amdgpu",
                      "vendor": "amd"})
    return found


# ------------------------------------------------------------------------------------------------ the engine
def driver_major(gpu):
    try:
        return int(gpu["driver"].split(".")[0])
    except (ValueError, KeyError):
        return 0


ENGINE_SOURCES = ("CMakeLists.txt", "cmake", "src", "include", "third_party/ggml")


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def settings_path() -> Path:
    if WIN:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Strata" / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower()


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache)."""
    sel = cfg.get("gpu")
    gl = [gpu_info(i) or {} for i in sel] if isinstance(sel, list) else [gpu_info(sel) or {}]
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning Strata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  draft depth, the CPU threads). It takes about 5-10 minutes; the PC is busy meanwhile.")
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" for k, v in res["settings"].items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return (load_settings().get("calibration") or {}).get(hardware_key(cfg))
