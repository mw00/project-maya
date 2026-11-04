"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  NVML is called in a child process
  of its own (this file run with --nvml-child), never in the server's: a call that crashes there - issue #136, a
  SIGSEGV inside nvmlDeviceGetCurrPcieLinkGeneration that took the whole server down - ends only that reader.  The
  server starts it again without that call (its reading stays empty), and gives the GPU readings up after
  _Reader.MAX_CRASHES crashes; a call that hangs is ended the same way after _Reader.TIMEOUT seconds in it.
- GPU (AMD, HIP backend): the amdgpu driver's sysfs files (Linux only, no extra library) - load, VRAM, temperature,
  power, PCIe link; see the _Amd class. Plain file reads, so read in the server's own process. Picked explicitly by
  the server (its backend) or by auto-detection when NVML finds no card.  On Windows the same AMD card is read from
  DXGI (name, memory size) and Windows' GPU performance counters in pdh.dll (load, memory used), as Task Manager does:
  the _WinGpu class (no temperature, power or PCIe link there).
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time

HISTORY = 60
IDLE_S = 3.0      # the sampler's pause while no request runs and nothing has read the numbers lately (else 1 s)
WATCH_S = 15.0    # "lately": how long after a GET /metrics (or a request) the sampler keeps its once-a-second pace


# ------------------------------------------------------------------------------------------------ NVML (the server)
def gpu_pids():
    """Every process with a compute context on any NVIDIA GPU (NVML), or None when that can't be read (no NVIDIA driver,
    an AMD card, an old driver, a reader that crashed too often)."""
    r = _reader().call({"op": "pids"})
    return None if r is None or r.get("pids") is None else set(r["pids"])


class _Reader:
    """The NVML reader process, one per server (the sampler and gpu_pids() share it, one request at a time).  It
    names each NVML call on its stdout before making it ("@name") and once it returned ("~"), then answers with one
    line of JSON; a reader that ends while a call is open - or is silent for TIMEOUT seconds in one (INIT_TIMEOUT in
    nvmlInit, which can take a while on a box of many GPUs) - had that call crash (or hang): the next reader leaves it
    out (--skip), and the request is asked again once."""
    TIMEOUT = 20.0
    INIT_TIMEOUT = 120.0
    MAX_CRASHES = 5

    def __init__(self, argv=None):
        self.argv = list(argv) if argv else [sys.executable, "-I", "-u", os.path.abspath(__file__), "--nvml-child"]
        self.lock = threading.Lock()
        self.skip: set = set()        # the calls that crashed or hung a reader
        self.crashes = 0
        self.off = False              # no NVML here (no NVIDIA driver), or the reader crashed too often
        self.proc = None
        self.lines = None
        self.pump = None

    def call(self, req: dict):
        """The reader's answer (a dict) to `req`, or None: no NVML, or the reader crashed on it."""
        with self.lock:
            for _ in range(2):
                if self.off:
                    return None
                try:
                    if self.proc is None:
                        self._start()
                    self.proc.stdin.write(json.dumps(req) + "\n")
                    self.proc.stdin.flush()
                except (OSError, ValueError) as e:
                    if self.proc is None:             # (it could not be started at all)
                        self.off = True
                        print(f"[maya] the GPU readings are off: their NVML reader did not start ({e})", flush=True)
                        return None
                    rep, culprit, why = None, None, f"its pipe broke ({e})"
                else:
                    rep, culprit, why = self._answer()
                if rep is not None:
                    if not rep.get("ok"):             # no NVML: an AMD card, no NVIDIA driver - nothing to read
                        self.off = True
                        self._stop()
                        return None
                    return rep
                self._crashed(culprit, why)
            return None

    def _start(self):
        args = self.argv + (["--skip", ",".join(sorted(self.skip))] if self.skip else [])
        kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
        q: queue.Queue = queue.Queue()

        def pump(f=self.proc.stdout):
            try:
                for line in f:
                    q.put(line.rstrip("\r\n"))
            except (OSError, ValueError):
                pass
            q.put(None)
        self.pump = threading.Thread(target=pump, name="nvml-reader", daemon=True)
        self.pump.start()
        self.lines = q

    def _answer(self):
        """(reply, None, None), or (None, the NVML call open at the end or None, what happened)."""
        cur, wait = None, self.TIMEOUT
        deadline = time.monotonic() + wait
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None, cur, f"no answer in {wait:.0f} s"
            if line is None:
                try:
                    code = self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    code = None
                return None, cur, f"it ended, exit code {code}"
            if line.startswith("@"):                   # (each call has its own time)
                cur = line[1:]
                wait = self.INIT_TIMEOUT if cur.startswith("nvmlInit") else self.TIMEOUT
                deadline = time.monotonic() + wait
            elif line == "~":
                cur, wait = None, self.TIMEOUT
                deadline = time.monotonic() + wait
            else:
                try:
                    return json.loads(line), None, None
                except ValueError:
                    continue

    def _stop(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.kill()
            p.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.pump.join(timeout=2)                    # (its pipe's end once the process is gone)
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except (OSError, ValueError):
                pass

    def _crashed(self, culprit, why):
        self._stop()
        self.crashes += 1
        if culprit:
            self.skip.add(culprit)
        if self.crashes >= self.MAX_CRASHES:
            self.off = True
            print(f"[maya] the GPU readings are off: their NVML reader crashed {self.crashes} times ({why}) - the "
                  "server goes on without them", flush=True)
        elif culprit:
            print(f"[maya] the GPU readings' NVML call {culprit} crashed their reader ({why}): that reading is left "
                  "out from now on - the server goes on", flush=True)
        else:
            print(f"[maya] the GPU readings' NVML reader stopped ({why}); starting it again", flush=True)


_READER = None
_READER_LOCK = threading.Lock()


def _reader() -> _Reader:
    global _READER
    with _READER_LOCK:
        if _READER is None:
            _READER = _Reader()
        return _READER


# ------------------------------------------------------------------------------------------------ NVML (the reader)
class _Nvml:
    """One GPU's readings through NVML (`lib`: the library, initialised - in the reader process only)."""

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    class Proc1(ctypes.Structure):                       # nvmlProcessInfo_v1_t
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong)]

    class Proc2(ctypes.Structure):                       # nvmlProcessInfo_v2_t (the _v2 and _v3 calls: 24 bytes)
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong), ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]

    class Pci(ctypes.Structure):                         # nvmlPciInfo_t
        _fields_ = [("bus_legacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                    ("device", ctypes.c_uint), ("pci_device_id", ctypes.c_uint), ("pci_subsystem_id", ctypes.c_uint),
                    ("bus_id", ctypes.c_char * 32)]

    def __init__(self, lib, index=0):
        self.lib, self.dev = lib, None
        if lib is None:
            return
        h = ctypes.c_void_p()
        try:
            get = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None) or lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) == 0:
                self.dev = h
        except (AttributeError, OSError):
            pass

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def procs(self):
        """{pid: bytes it holds on this GPU (None: NVML can't say, as on Windows)} of the processes with a compute
        context on it, or None when NVML can't list them."""
        for fn, st in (("nvmlDeviceGetComputeRunningProcesses_v3", self.Proc2),
                       ("nvmlDeviceGetComputeRunningProcesses_v2", self.Proc2),
                       ("nvmlDeviceGetComputeRunningProcesses", self.Proc1)):
            f = getattr(self.lib, fn, None)
            if f is None:
                continue
            # (room for 256 records of up to twice this size: a driver whose record is larger stays inside it)
            n, buf = ctypes.c_uint(256), (st * 512)()
            try:
                if f(self.dev, ctypes.byref(n), buf) == 0:      # (NVML_VALUE_NOT_AVAILABLE: all ones)
                    return {buf[i].pid: None if buf[i].used >= 2**63 else buf[i].used for i in range(min(n.value, 256))}
            except OSError:
                pass
        return None

    def pids(self):
        """The processes with a compute context on this GPU, or None when NVML can't say."""
        p = self.procs()
        return None if p is None else set(p)

    def pci(self):
        """The card's PCI address as "0000:01:00" (domain:bus:device; the engine's VRAM lines carry CUDA's), or None."""
        info = self.Pci()
        for fn in ("nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetPciInfo_v2", "nvmlDeviceGetPciInfo"):
            f = getattr(self.lib, fn, None)
            try:
                if f is not None and f(self.dev, ctypes.byref(info)) == 0:
                    return f"{info.domain:04x}:{info.bus:02x}:{info.device:02x}"
            except OSError:
                pass
        return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# every NVML call's arguments and result (nvml.h), declared once on the library: none is default-typed
_U, _PU, _D = ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p
_SIGS = {
    "nvmlInit_v2": [], "nvmlInit": [],
    "nvmlDeviceGetCount_v2": [_PU], "nvmlDeviceGetCount": [_PU],
    "nvmlDeviceGetHandleByIndex_v2": [_U, ctypes.POINTER(_D)], "nvmlDeviceGetHandleByIndex": [_U, ctypes.POINTER(_D)],
    "nvmlDeviceGetUtilizationRates": [_D, ctypes.POINTER(_Nvml.Util)],
    "nvmlDeviceGetMemoryInfo": [_D, ctypes.POINTER(_Nvml.Mem)],
    "nvmlDeviceGetTemperature": [_D, _U, _PU],
    "nvmlDeviceGetPowerUsage": [_D, _PU], "nvmlDeviceGetEnforcedPowerLimit": [_D, _PU],
    "nvmlDeviceGetCurrPcieLinkGeneration": [_D, _PU], "nvmlDeviceGetMaxPcieLinkGeneration": [_D, _PU],
    "nvmlDeviceGetCurrPcieLinkWidth": [_D, _PU], "nvmlDeviceGetPcieThroughput": [_D, _U, _PU],
    "nvmlDeviceGetComputeRunningProcesses_v3": [_D, _PU, _D],
    "nvmlDeviceGetComputeRunningProcesses_v2": [_D, _PU, _D],
    "nvmlDeviceGetComputeRunningProcesses": [_D, _PU, _D],
    "nvmlDeviceGetPciInfo_v3": [_D, ctypes.POINTER(_Nvml.Pci)], "nvmlDeviceGetPciInfo_v2": [_D, ctypes.POINTER(_Nvml.Pci)],
    "nvmlDeviceGetPciInfo": [_D, ctypes.POINTER(_Nvml.Pci)],
    "nvmlDeviceGetName": [_D, ctypes.c_char_p, _U],
}


def _open_nvml():
    """NVIDIA's NVML library with every call above declared, or None (no NVIDIA driver)."""
    names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                      "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
        else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    for n in names:
        try:
            lib = ctypes.CDLL(n)
        except OSError:
            continue
        for fn, args in _SIGS.items():
            try:
                f = getattr(lib, fn)
            except AttributeError:
                continue
            f.restype, f.argtypes = ctypes.c_int, args
        return lib
    return None


class _Marked:
    """The library as the reader calls it: each call named on stdout before it runs ("@name") and once it returned
    ("~"), so the server knows the call a crash came in; the calls in `skip` are missing (AttributeError, as a call an
    older driver lacks)."""

    def __init__(self, lib, skip, out):
        self._lib, self._skip, self._out, self._fns = lib, set(skip), out, {}

    def __getattr__(self, name):
        if name.startswith("_") or name in self._skip:
            raise AttributeError(name)
        f = self._fns.get(name)
        if f is None:
            real, out = getattr(self._lib, name), self._out

            def f(*args, _real=real, _name=name):
                out.write(f"@{_name}\n")
                out.flush()
                try:
                    return _real(*args)
                finally:
                    out.write("~\n")
                    out.flush()
            self._fns[name] = f
        return f


def _child_main(args, lib=None):
    """The reader process: one JSON request a line on stdin ({"op": "info" | "read", "idx": [GPU, ...]} or
    {"op": "pids"}), one JSON answer a line on stdout; {"ok": false} when there is no NVML.  `lib`: a stand-in for
    the library (tests)."""
    skip = set()
    if "--skip" in args and args.index("--skip") + 1 < len(args):
        skip = {s for s in args[args.index("--skip") + 1].split(",") if s}
    if os.name == "nt":       # a crash ends the process at once (no error dialog that would hold it)
        try:
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        except (AttributeError, OSError):
            pass
    out = sys.stdout
    real = lib if lib is not None else _open_nvml()
    nv = _Marked(real, skip, out) if real is not None else None
    ok = False
    if nv is not None:
        try:
            init = getattr(nv, "nvmlInit_v2", None) or nv.nvmlInit
            ok = init() == 0
        except (AttributeError, OSError):
            ok = False
    devs: dict = {}

    def dev(i):
        if i not in devs:
            devs[i] = _Nvml(nv, i)
        return devs[i]

    def reading(d):
        if not d.ok():
            return None
        r = d.read()
        p = d.procs()
        r["procs"] = None if p is None else [[pid, used] for pid, used in p.items()]
        return r

    def all_pids():
        count = ctypes.c_uint()
        try:
            fn = getattr(nv, "nvmlDeviceGetCount_v2", None) or nv.nvmlDeviceGetCount
            if fn(ctypes.byref(count)) != 0:
                return None
        except (AttributeError, OSError):
            return None
        pids = set()
        for i in range(count.value):
            p = dev(i).pids() if dev(i).ok() else None
            if p is None:
                return None
            pids |= p
        return sorted(pids)

    while True:
        line = sys.stdin.readline()
        if not line:
            return
        try:
            req = json.loads(line)
        except ValueError:
            continue
        op, idx = req.get("op"), [int(i) for i in req.get("idx") or []]
        if not ok:
            rep = {"ok": False}
        elif op == "info":
            rep = {"ok": True, "gpus": [{"index": i, "ok": dev(i).ok(), "pci": dev(i).pci() if dev(i).ok() else None,
                                         "name": dev(i).name() if dev(i).ok() else None} for i in idx]}
        elif op == "read":
            rep = {"ok": True, "gpus": [reading(dev(i)) for i in idx]}
        elif op == "pids":
            rep = {"ok": True, "pids": all_pids()}
        else:
            rep = {"ok": True}
        out.write(json.dumps(rep) + "\n")
        out.flush()


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`, or None
    when there is no such card (or no amdgpu).  HIP orders the cards as the KFD topology lists its GPU nodes (the
    CPU nodes skipped) - the same order setup's amd_gpus() and HIP_VISIBLE_DEVICES use - and KFD links each node to
    its render node by drm_render_minor, which is the bridge between the two numberings."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """An AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap).
    Every read is tolerant: a file the driver or kernel version does not have leaves its value None."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    def pci(self):
        """The card's own PCI address as _Nvml's "0000:03:00" (domain:bus:device, what the GPU map matches the engine's
        VRAM lines by), or None: the last device of the sysfs path - the ones before it are the bridges above it."""
        try:
            bdfs = [part for part in os.path.realpath(self.dev or "").split("/") if self._bdf(part)]
            return bdfs[-1][:10] if bdfs else None
        except OSError:
            return None

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` keeps the card's own hop aside: the gap between the two is what a user has to see.

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_path": "%d/%d" % (read, len(hops))}

    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        return out


# ------------------------------------------------------------------------------------------------ AMD (Windows)
def _dxgi_adapters():
    """[(name, vendor_id, dedicated_bytes, luid_high, luid_low)] of the hardware graphics adapters, in DXGI's order
    (CreateDXGIFactory1 + IDXGIFactory1::EnumAdapters1 + IDXGIAdapter1::GetDesc1 through COM vtables; software
    adapters skipped); [] when DXGI cannot be asked."""
    class Desc1(ctypes.Structure):                       # DXGI_ADAPTER_DESC1
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                    ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("LuidLow", ctypes.c_uint32), ("LuidHigh", ctypes.c_int32), ("Flags", ctypes.c_uint)]

    class Guid(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]

    def method(obj, slot, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[slot])

    out = []
    try:
        iid = Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        factory = ctypes.c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory:
            return []
        try:
            for i in range(64):
                adapter = ctypes.c_void_p()
                if method(factory, 12, ctypes.c_uint, ctypes.c_void_p)(factory, i, ctypes.byref(adapter)) != 0:
                    break                                # DXGI_ERROR_NOT_FOUND: the last one
                try:
                    d = Desc1()
                    if method(adapter, 10, ctypes.c_void_p)(adapter, ctypes.byref(d)) == 0 and not d.Flags & 2:
                        out.append((d.Description, d.VendorId, int(d.DedicatedVideoMemory),
                                    d.LuidHigh & 0xFFFFFFFF, d.LuidLow))
                finally:
                    method(adapter, 2)(adapter)          # Release
        finally:
            method(factory, 2)(factory)
    except (OSError, AttributeError, ValueError):
        return []
    return out


class _Pdh:
    """Windows' GPU performance counters (pdh.dll, what Task Manager shows): one query with the GPU Engine
    utilisation and the GPU Adapter Memory usage of every adapter; values() -> {counter: [(instance, value)]}."""
    COUNTERS = {"engine": r"\GPU Engine(*)\Utilization Percentage",
                "dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
                "shared": r"\GPU Adapter Memory(*)\Shared Usage"}

    class Item(ctypes.Structure):                        # PDH_FMT_COUNTERVALUE_ITEM_W (the value as a double)
        _fields_ = [("name", ctypes.c_wchar_p), ("status", ctypes.c_uint32), ("value", ctypes.c_double)]

    def __init__(self):
        self.query, self.counters = ctypes.c_void_p(), {}
        try:
            self.lib = ctypes.windll.pdh
            if self.lib.PdhOpenQueryW(None, None, ctypes.byref(self.query)) != 0:
                self.query = None
                return
            for key, path in self.COUNTERS.items():
                c = ctypes.c_void_p()
                if self.lib.PdhAddEnglishCounterW(self.query, path, None, ctypes.byref(c)) == 0:
                    self.counters[key] = c
            self.lib.PdhCollectQueryData(self.query)     # (a rate counter needs two collections)
        except (OSError, AttributeError):
            self.query = None

    def ok(self):
        return bool(self.query) and "dedicated" in self.counters

    def values(self):
        out = {}
        if not self.ok() or self.lib.PdhCollectQueryData(self.query) != 0:
            return out
        for key, c in self.counters.items():
            size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
            fmt = 0x200 | 0x8000                         # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
            if self.lib.PdhGetFormattedCounterArrayW(c, fmt, ctypes.byref(size), ctypes.byref(count), None) \
                    != ctypes.c_long(0x800007D2).value:  # PDH_MORE_DATA: the size it needs
                continue
            buf = (ctypes.c_byte * size.value)()
            if self.lib.PdhGetFormattedCounterArrayW(c, fmt, ctypes.byref(size), ctypes.byref(count), buf) != 0:
                continue
            items = ctypes.cast(buf, ctypes.POINTER(self.Item))
            out[key] = [(items[i].name or "", items[i].value) for i in range(count.value)
                        if items[i].status in (0, 1)]    # PDH_CSTATUS_VALID_DATA / _NEW_DATA
        return out


def win_gpu_reading(values, luid):
    """The readings of the adapter `luid` ("luid_0x00000000_0x0000d1a3") from _Pdh.values(): the load is the busiest
    engine's (each engine's processes summed, as Task Manager counts it - not Linux's whole-card busy percent), the
    memory its dedicated (the VRAM, or an APU's carve-out) and shared usage in bytes; None where nothing was read."""
    luid = luid.lower()
    engines = {}
    for name, v in values.get("engine", ()):
        name = name.lower()
        at = name.find(luid)
        key = name[at:].split("_engtype")[0]            # luid_..._phys_0_eng_3: one engine, any process
        if at < 0 or "_eng_" not in key:
            continue
        engines[key] = engines.get(key, 0.0) + v
    out = {"util": round(min(100.0, max(engines.values())), 1) if engines else None}
    for key, field in (("dedicated", "mem_used"), ("shared", "mem_shared")):
        vals = [v for name, v in values.get(key, ()) if name.lower().startswith(luid)]
        out[field] = int(sum(vals)) if vals else None
    return out


class _WinGpu:
    """An AMD card on Windows (the HIP backend), with _Amd's interface: the name and total memory from DXGI, the load
    and memory used from Windows' GPU performance counters (_Pdh). No temperature, power or PCIe link (Windows has no
    such counters for it). `index` counts the AMD adapters in DXGI's order. Every failure leaves ok() False."""
    VENDOR_AMD = 0x1002

    def __init__(self, index=0, adapters=None, pdh=None):
        amd = [a for a in (_dxgi_adapters() if adapters is None else adapters) if a[1] == self.VENDOR_AMD]
        self.card = amd[index] if 0 <= index < len(amd) else None
        self.pdh = pdh if pdh is not None or self.card is None else _Pdh()
        self.luid = "luid_0x%08x_0x%08x" % (self.card[3], self.card[4]) if self.card else None

    def ok(self):
        return self.card is not None and self.pdh is not None and self.pdh.ok()

    def name(self):
        return (self.card[0] if self.card else "") or "AMD Radeon"

    def pci(self):
        return None

    def read(self):
        out = win_gpu_reading(self.pdh.values(), self.luid) if self.ok() else {}
        out["mem_total"] = self.card[2] if self.card and self.card[2] else None
        return out


WINDOWS = os.name == "nt"


def amd_card(index=0):
    """An AMD card's reader for the HIP backend: the amdgpu sysfs files (Linux), else Windows' counters (_WinGpu)."""
    g = _Amd(index)
    return g if g.ok() or not WINDOWS else _WinGpu(index)


class _NvmlCard:
    """One NVIDIA card through the NVML reader process (_Reader), with _Amd's interface (gpu_reader)."""

    def __init__(self, index=0, reader=None):
        self.index, self.nv = index, reader or _reader()
        g = ((self.nv.call({"op": "info", "idx": [index]}) or {}).get("gpus") or [None])[0] or {}
        self._ok, self._pci, self._name = bool(g.get("ok")), g.get("pci"), g.get("name")

    def ok(self):
        return self._ok

    def pci(self):
        return self._pci

    def name(self):
        return self._name

    def _reading(self):
        r = ((self.nv.call({"op": "read", "idx": [self.index]}) or {}).get("gpus") or [None])[0]
        return r if isinstance(r, dict) else {}

    def read(self):
        return {k: v for k, v in self._reading().items() if k != "procs"}

    def procs(self):
        p = self._reading().get("procs")
        return None if p is None else {int(pid): b for pid, b in p}


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML through the reader process (NVIDIA), or with the AMD/HIP backend the amdgpu sysfs
    files (Linux, plain file reads) or Windows' GPU counters (amd_card), read here."""
    return amd_card(index) if amd else _NvmlCard(index)


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read.  `amd`: an AMD card on the HIP backend, read from
    the sysfs files (see gpu_reader)."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False, reader: _Reader | None = None,
                 start=True, busy=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the cards are
        AMD ones read from the sysfs files (the HIP backend); when False and NVML finds no card at all, AMD is
        tried anyway (auto-detection), so the Monitor works on a HIP setup that runs this file's older caller.
        `reader`: the NVML reader process (the server's own by default); `start`: sample in a thread of its own
        (else sample() only); `busy()` -> whether a request is running: while none is and nobody has read the numbers
        for WATCH_S, the sampler pauses IDLE_S between readings instead of 1 s (issue #685: every reading wakes the
        server and the NVML process - for numbers nobody looks at); a reader of the numbers (snapshot) or touch()
        returns it to once a second at once."""
        self.extra = extra
        self.busy = busy
        self.watched_at = time.monotonic()           # (starts watched: the first readings come at the usual pace)
        self._wake = threading.Event()
        self._slow = False
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.nv = None
        self.amd = None                              # [(index, _Amd / _WinGpu)]: AMD cards (sysfs / Windows counters)
        cards = []
        if amd:
            self.amd = [(i, amd_card(i)) for i in idx]
        else:
            self.nv = reader or _reader()
            info = self.nv.call({"op": "info", "idx": idx}) or {}
            cards = [g for g in info.get("gpus") or [] if g and g.get("ok")]
            if not cards:
                # auto-detect: NVML found no card at all - try the amdgpu sysfs (a HIP setup that does not say so)
                found = [(i, amd_card(i)) for i in idx]
                if any(g.ok() for _, g in found):
                    self.amd = found
        if self.amd is not None:
            self.amd = [(i, g) for i, g in self.amd if g.ok()]
            cards = [{"index": i, "pci": g.pci(), "name": g.name()} for i, g in self.amd]
        self.idx = [int(g["index"]) for g in cards]
        self.cards = [(int(g["index"]), g.get("pci"), g.get("name")) for g in cards]   # (the GPU map's)
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": " + ".join(g.get("name") or "?" for g in cards) if cards else None,
            "gpu_count": len(cards) or 1,
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def _gpu_reads(self):
        """[(index, readings)] of the cards that answered (the reader's, or the sysfs files'; [] when nothing).  An AMD
        card's processes are not listed (procs None: amdgpu's fdinfo names them, in a layout not read here)."""
        if not self.idx:
            return []
        if self.amd is not None:
            return [(i, dict(g.read(), procs=None)) for i, g in self.amd]
        rep = self.nv.call({"op": "read", "idx": self.idx}) or {}
        return [(i, r) for i, r in zip(self.idx, rep.get("gpus") or []) if isinstance(r, dict)]

    def sample(self):
        s = {}
        reads = self._gpu_reads()
        if reads:
            g = {k: v for k, v in reads[0][1].items() if k != "procs"}
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power"),
                              "power_limit": r.get("power_limit"), "pcie_gen": r.get("pcie_gen"),
                              "pcie_gen_max": r.get("pcie_gen_max"), "pcie_width": r.get("pcie_width"),
                              "pcie_rx_mb": r.get("pcie_rx_mb")}
                             for i, r in reads]
            s.update({f"gpu_{k}": v for k, v in g.items()})
            # each card's memory and the processes holding it, for the Monitor's map of a GPU (serve/gpu_map.py)
            by = dict(reads)
            s["vram"] = [{"index": i, "pci": pci, "name": name, "used": by[i].get("mem_used"),
                          "total": by[i].get("mem_total"),
                          "procs": None if by[i].get("procs") is None else {int(p): b for p, b in by[i]["procs"]}}
                         for i, pci, name in self.cards if i in by]
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def touch(self) -> None:
        """Someone looks at the numbers, or a request starts: back to once a second, now."""
        self.watched_at = time.monotonic()
        if self._slow:
            self._wake.set()

    def pause(self) -> float:
        """The seconds to the next reading."""
        quiet = self.busy is not None and time.monotonic() - self.watched_at > WATCH_S
        try:
            return IDLE_S if quiet and not self.busy() else 1.0
        except Exception:  # noqa: BLE001 - not known: the usual pace
            return 1.0

    def _loop(self):
        last = time.monotonic()
        while True:
            try:
                s = self.sample()
            except Exception:  # noqa: BLE001 - a reading that fails is empty this second; the sampler goes on
                s = {}
            now = time.monotonic()
            # (a reading after a long pause stands for each of its seconds: the series stay a second apart)
            held = max(1, min(int(IDLE_S) + 1, int(round(now - last))))
            last = now
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    for _ in range(held):
                        self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            wait = self.pause()
            self._slow = wait > 1.0
            if self._slow:
                self._wake.wait(wait)
                self._wake.clear()
                self._slow = False
            else:
                time.sleep(wait)

    def snapshot(self):
        self.touch()                                 # (somebody reads the numbers)
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}


if __name__ == "__main__" and sys.argv[1:2] == ["--nvml-child"]:
    _child_main(sys.argv[2:])
