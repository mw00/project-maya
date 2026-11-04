"""The GPU readings' NVML calls run in a reader process of their own (issue #136: a SIGSEGV inside
nvmlDeviceGetCurrPcieLinkGeneration took the whole server down).  These tests run the real reader loop
(telemetry._child_main) over a stand-in for the NVML library whose calls can crash the process for real (a NULL read),
hang, or be missing - no GPU or driver needed.  AmdSysfs: #269's amdgpu sysfs readings (a fake /sys)."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import telemetry as T  # noqa: E402

# the reader process over a fake NVML: argv[1] = {"crash": [calls], "hang": [calls], "absent": bool, "die": bool}
FAKE = r"""
import ctypes, faulthandler, json, os, sys, time
sys.path.insert(0, ROOT)
from serve import telemetry as T
spec = json.loads(sys.argv[1])

def crash():
    if os.name == "nt":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    getattr(faulthandler, "_sigsegv", os.abort)()     # a real access violation / SIGSEGV
    os.abort()

class Fake:
    def __getattribute__(self, name):
        f = object.__getattribute__(self, name)
        if name in spec.get("crash", []):
            return lambda *a: crash()
        if name in spec.get("hang", []):
            return lambda *a: time.sleep(3600)
        if name in spec.get("slow", {}):
            return lambda *a: (time.sleep(spec["slow"][name]), f(*a))[1]
        return f
    def nvmlInit_v2(self):
        return 0
    def nvmlDeviceGetCount_v2(self, n):
        n._obj.value = 2
        return 0
    def nvmlDeviceGetHandleByIndex_v2(self, i, h):
        h._obj.value = 100 + i.value
        return 0
    def nvmlDeviceGetUtilizationRates(self, d, u):
        u._obj.gpu = 42
        return 0
    def nvmlDeviceGetMemoryInfo(self, d, m):
        m._obj.total, m._obj.used = 32 << 30, (8 + d.value - 100) << 30
        return 0
    def _v(self, ref, v):
        ref._obj.value = v
        return 0
    def nvmlDeviceGetTemperature(self, d, s, v):
        return self._v(v, 60 + d.value - 100)
    def nvmlDeviceGetPowerUsage(self, d, v):
        return self._v(v, 150000)
    def nvmlDeviceGetEnforcedPowerLimit(self, d, v):
        return self._v(v, 250000)
    def nvmlDeviceGetCurrPcieLinkGeneration(self, d, v):
        return self._v(v, 4)
    def nvmlDeviceGetMaxPcieLinkGeneration(self, d, v):
        return self._v(v, 4)
    def nvmlDeviceGetCurrPcieLinkWidth(self, d, v):
        return self._v(v, 16)
    def nvmlDeviceGetPcieThroughput(self, d, which, v):
        return self._v(v, 2048)
    def nvmlDeviceGetComputeRunningProcesses_v3(self, d, n, buf):
        buf[0].pid, buf[0].used, n._obj.value = 4242, 1 << 30, 1
        return 0
    def nvmlDeviceGetPciInfo_v3(self, d, p):
        p._obj.domain, p._obj.bus, p._obj.device = 0, d.value - 99, 0
        return 0
    def nvmlDeviceGetName(self, d, buf, n):
        buf.value = b"Fake V100"
        return 0

if spec.get("die"):                  # a reader that ends at its first request, outside any NVML call
    sys.stdin.readline()
    os._exit(3)
if spec.get("absent"):               # no NVML library on this machine (not the real one either)
    T._open_nvml = lambda: None
    T._child_main(sys.argv[2:])
else:
    T._child_main(sys.argv[2:], lib=Fake())
"""


def reader(**spec):
    return T._Reader([sys.executable, "-I", "-u", "-c", "ROOT = " + repr(str(ROOT)) + "\n" + FAKE, json.dumps(spec)])


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class ReaderProcess(unittest.TestCase):
    def tearDown(self):
        for r in getattr(self, "readers", []):
            r._stop()

    def keep(self, r):
        self.readers = getattr(self, "readers", []) + [r]
        return r

    def test_readings(self):
        r = self.keep(reader())
        info = r.call({"op": "info", "idx": [0, 1]})
        self.assertEqual([(g["index"], g["pci"], g["name"]) for g in info["gpus"]],
                         [(0, "0000:01:00", "Fake V100"), (1, "0000:02:00", "Fake V100")])
        g = r.call({"op": "read", "idx": [0]})["gpus"][0]
        self.assertEqual((g["util"], g["mem_used"], g["temp"], g["power"], g["power_limit"], g["pcie_gen"],
                          g["pcie_width"], g["pcie_rx_mb"]), (42, 8 << 30, 60, 150.0, 250.0, 4, 16, 2.0))
        self.assertEqual(g["procs"], [[4242, 1 << 30]])
        self.assertEqual(r.call({"op": "pids"})["pids"], [4242])
        self.assertEqual((r.crashes, r.skip), (0, set()))

    def test_a_crashing_call_is_left_out(self):
        """The issue's crash: the reader dies inside one call; the server's side starts it again without that call,
        answers the same request with everything else, and keeps that reader."""
        r = self.keep(reader(crash=["nvmlDeviceGetCurrPcieLinkGeneration"]))
        with quiet() as out:
            g = r.call({"op": "read", "idx": [0]})["gpus"][0]
        self.assertIsNone(g["pcie_gen"])
        self.assertEqual((g["util"], g["power"], g["pcie_gen_max"]), (42, 150.0, 4))
        self.assertEqual((r.crashes, r.skip, r.off), (1, {"nvmlDeviceGetCurrPcieLinkGeneration"}, False))
        self.assertIn("nvmlDeviceGetCurrPcieLinkGeneration crashed", out.getvalue())
        pid = r.proc.pid
        for _ in range(3):
            self.assertEqual(r.call({"op": "read", "idx": [0]})["gpus"][0]["util"], 42)
        self.assertEqual((r.proc.pid, r.crashes), (pid, 1))

    def test_a_hanging_call_is_left_out(self):
        r = self.keep(reader(hang=["nvmlDeviceGetPowerUsage"]))
        r.TIMEOUT = 3.0
        with quiet():
            g = r.call({"op": "read", "idx": [0]})["gpus"][0]
        self.assertIsNone(g["power"])
        self.assertEqual((g["util"], g["pcie_gen"]), (42, 4))
        self.assertEqual(r.skip, {"nvmlDeviceGetPowerUsage"})

    def test_a_slow_init_and_a_slow_read_are_waited_for(self):
        """Each call has its own time (a slow call is not a hung one), nvmlInit a longer one (many GPUs)."""
        r = self.keep(reader(slow={"nvmlInit_v2": 3, "nvmlDeviceGetMemoryInfo": 1, "nvmlDeviceGetPowerUsage": 1}))
        r.TIMEOUT, r.INIT_TIMEOUT = 1.5, 6.0
        g = r.call({"op": "read", "idx": [0]})["gpus"][0]
        self.assertEqual((g["util"], g["mem_used"], g["power"]), (42, 8 << 30, 150.0))
        self.assertEqual((r.crashes, r.skip), (0, set()))

    def test_a_crash_in_the_process_list_falls_back(self):
        r = self.keep(reader(crash=["nvmlDeviceGetComputeRunningProcesses_v3"]))
        with quiet():
            g = r.call({"op": "read", "idx": [0]})["gpus"][0]
        self.assertIsNone(g["procs"])            # (the fake has no _v2 / v1 call to fall back to)
        self.assertEqual(g["util"], 42)

    def test_a_crash_in_init_turns_the_readings_off(self):
        r = self.keep(reader(crash=["nvmlInit_v2"]))
        with quiet():
            self.assertIsNone(r.call({"op": "read", "idx": [0]}))
        self.assertTrue(r.off)
        self.assertIsNone(r.proc)
        self.assertIsNone(r.call({"op": "pids"}))   # (nothing started again)
        self.assertIsNone(r.proc)

    def test_no_nvml(self):
        r = self.keep(reader(absent=True))
        with quiet() as out:
            self.assertIsNone(r.call({"op": "info", "idx": [0]}))
        self.assertTrue(r.off)
        self.assertEqual((r.crashes, out.getvalue()), (0, ""))   # (an AMD card: nothing to say)

    def test_a_reader_that_keeps_dying_is_given_up(self):
        r = self.keep(reader(die=True))
        with quiet() as out:
            for _ in range(4):
                self.assertIsNone(r.call({"op": "read", "idx": [0]}))
        self.assertTrue(r.off)
        self.assertEqual(r.crashes, r.MAX_CRASHES)
        self.assertIn("the GPU readings are off", out.getvalue())


class Sampler(unittest.TestCase):
    def test_sample_through_the_reader(self):
        r = reader(crash=["nvmlDeviceGetCurrPcieLinkGeneration"])
        self.addCleanup(r._stop)
        with quiet():
            t = T.Telemetry(gpu_indices=[0, 1], reader=r, start=False)
            s = t.sample()
        self.assertEqual(t.static["gpu_name"], "Fake V100 + Fake V100")
        self.assertEqual(t.static["gpu_count"], 2)
        self.assertEqual((s["gpu_util"], s["gpu_mem_used"], s["gpu_temp"], s["gpu_power"]), (42, 17 << 30, 61, 300.0))
        self.assertEqual([g["pcie_gen"] for g in s["gpus"]], [None, None])
        self.assertEqual([(v["index"], v["pci"], v["procs"]) for v in s["vram"]],
                         [(0, "0000:01:00", {4242: 1 << 30}), (1, "0000:02:00", {4242: 1 << 30})])
        self.assertNotIn("gpu_procs", s)
        json.dumps(t.snapshot())

    def test_without_nvml(self):
        r = reader(absent=True)
        # (nor an AMD card: AmdSysfs and AmdWindows have those)
        with mock.patch.object(T, "SYSFS", str(ROOT / "no-such-sysfs")), mock.patch.object(T, "WINDOWS", False):
            t = T.Telemetry(gpu_index=0, reader=r, start=False)
            s = t.sample()
        self.assertEqual((t.static["gpu_name"], t.static["gpu_count"], t.cards), (None, 1, []))
        self.assertNotIn("gpu_util", s)
        self.assertIn("ram_total", s)

    def test_an_idle_unwatched_server_samples_slower_and_wakes_at_a_look(self):
        """Issue #685: no request and nobody reading the numbers for a while: a reading every IDLE_S, not every second;
        a read of the numbers (snapshot), a request (touch) or work in hand brings back the second at once."""
        r = reader(absent=True)
        busy = [False]
        with mock.patch.object(T, "SYSFS", str(ROOT / "no-such-sysfs")), mock.patch.object(T, "WINDOWS", False):
            t = T.Telemetry(gpu_index=0, reader=r, start=False, busy=lambda: busy[0])
        self.assertEqual(t.pause(), 1.0)                       # (just started: watched)
        t.watched_at -= T.WATCH_S + 1
        self.assertEqual(t.pause(), T.IDLE_S)                  # unwatched and idle
        busy[0] = True
        self.assertEqual(t.pause(), 1.0)                       # a request runs
        busy[0] = False
        t.snapshot()
        self.assertEqual(t.pause(), 1.0)                       # somebody read the numbers
        self.assertEqual(T.Telemetry(gpu_index=0, reader=r, start=False).pause(), 1.0)   # no busy(): as before

    def test_the_loop_keeps_the_series_a_second_apart_and_wakes_from_its_long_pause(self):
        import threading
        import time
        r = reader(absent=True)
        with mock.patch.object(T, "SYSFS", str(ROOT / "no-such-sysfs")), mock.patch.object(T, "WINDOWS", False), \
             mock.patch.object(T, "IDLE_S", 3.0), mock.patch.object(T, "WATCH_S", 0.0):
            t = T.Telemetry(gpu_index=0, reader=r, start=False, busy=lambda: False, extra=lambda: {"tok_s": 7.0})
            time.sleep(0.01)
            threading.Thread(target=t._loop, daemon=True).start()
            time.sleep(0.2)
            n = len(t.hist["tok_s"])                           # (the first reading; then it pauses IDLE_S)
            self.assertGreaterEqual(n, 1)
            self.assertTrue(t._slow)
            t0 = time.monotonic()
            t.touch()                                          # a request: the next reading comes now, not in 3 s
            for _ in range(100):
                if len(t.hist["tok_s"]) > n:
                    break
                time.sleep(0.01)
            self.assertGreater(len(t.hist["tok_s"]), n)
            self.assertLess(time.monotonic() - t0, 0.5)


    def test_gpu_pids(self):
        r = reader()
        self.addCleanup(r._stop)
        with mock.patch.object(T, "_READER", r):
            self.assertEqual(T.gpu_pids(), {4242})
        r2 = reader(absent=True)
        with mock.patch.object(T, "_READER", r2):
            self.assertIsNone(T.gpu_pids())

    def test_this_machine(self):
        """The real reader on this machine (NVML or none): it answers or says there is nothing - never raises."""
        r = T._Reader()
        self.addCleanup(r._stop)
        t = T.Telemetry(gpu_index=0, reader=r, start=False)
        s = t.sample()
        self.assertIsInstance(s, dict)
        pids = None if r.off else r.call({"op": "pids"})
        self.assertTrue(pids is None or isinstance(pids.get("pids"), (list, type(None))))


class AmdSysfs(unittest.TestCase):
    """#269: an AMD card on the HIP backend is read from the amdgpu sysfs files, in the server's own process (plain
    file reads); the NVML reader is not started for it.  A fake /sys: the KFD topology (a CPU node, then the GPU node
    with its render minor) and the card's files; on Linux the card sits behind a bridge chain with a Gen3 hop."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.sysfs = str(root)

        def put(path, text):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        nodes = root / "class/kfd/kfd/topology/nodes"
        put(nodes / "0/properties", "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\n")
        put(nodes / "1/properties", "simd_count 192\ngfx_target_version 110000\ndrm_render_minor 128\n")
        hops = [("0000:00:01.1", "16.0 GT/s PCIe", 16), ("0000:01:00.0", "8.0 GT/s PCIe", 16),
                ("0000:02:00.0", "16.0 GT/s PCIe", 16), ("0000:03:00.0", "16.0 GT/s PCIe", 16)]
        d = root / "devices/pci0000:00"
        if os.name == "nt":                # (no ':' in a Windows file name: the card alone, its address not read)
            hops, d = [], root / "devices/card0"
        for name, speed, width in hops:
            d = d / name
            put(d / "max_link_speed", speed)
            put(d / "max_link_width", str(width))
        card = d
        for f, v in (("gpu_busy_percent", "37"), ("mem_info_vram_used", str(4 << 30)),
                     ("mem_info_vram_total", str(24 << 30)), ("product_name", "AMD Radeon RX 7900 XTX"),
                     ("current_link_speed", "16.0 GT/s PCIe"), ("current_link_width", "16"),
                     ("hwmon/hwmon3/temp1_input", "55000"), ("hwmon/hwmon3/power1_average", "120000000"),
                     ("hwmon/hwmon3/power1_cap", "300000000")):
            put(card / f, v)
        dev = root / "class/drm/renderD128/device"
        dev.parent.mkdir(parents=True)
        self.linked = False
        if os.name != "nt":                # (sysfs is Linux's: the card's path is a link into /sys/devices)
            try:
                os.symlink(card, dev)
                self.linked = True
            except OSError:
                pass
        if not self.linked:
            shutil.copytree(card, dev)

    def check(self, t, s):
        self.assertEqual((t.static["gpu_name"], t.static["gpu_count"]), ("AMD Radeon RX 7900 XTX", 1))
        self.assertEqual((s["gpu_util"], s["gpu_mem_used"], s["gpu_mem_total"], s["gpu_temp"], s["gpu_power"],
                          s["gpu_power_limit"]), (37, 4 << 30, 24 << 30, 55.0, 120.0, 300.0))
        self.assertIsNone(s["vram"][0]["procs"])
        if self.linked:                    # the card's own address; the Gen3 hop is the link it gets
            self.assertEqual(t.cards[0][1], "0000:03:00")
            self.assertEqual((s["gpu_pcie_gen"], s["gpu_pcie_own_gen"], s["gpu_pcie_path"]), (3, 4, "4/4"))

    def test_hip_backend(self):
        with mock.patch.object(T, "SYSFS", self.sysfs), \
                mock.patch.object(T, "_reader", side_effect=AssertionError("NVML for an AMD card")):
            t = T.Telemetry(gpu_index=0, amd=True, start=False)
            self.check(t, t.sample())
            self.assertEqual(T.free_vram_mib(0, amd=True), 20 << 10)
            self.assertIsNone(T.free_vram_mib(1, amd=True))      # (no second GPU node)

    def test_auto_detected_when_nvml_has_no_card(self):
        r = reader(absent=True)
        with mock.patch.object(T, "SYSFS", self.sysfs):
            t = T.Telemetry(gpu_index=0, reader=r, start=False)
            self.check(t, t.sample())

    def test_nvidia_card_keeps_nvml(self):
        r = reader()
        self.addCleanup(r._stop)
        with mock.patch.object(T, "SYSFS", self.sysfs):
            t = T.Telemetry(gpu_index=0, reader=r, start=False)
        self.assertIsNone(t.amd)
        self.assertEqual(t.static["gpu_name"], "Fake V100")


class AmdWindows(unittest.TestCase):
    """#102 (B): an AMD card on Windows, read from DXGI and Windows' GPU counters (_WinGpu; fake ones here - the
    real calls are checked on a Windows + AMD PC)."""
    LUID = "luid_0x00000000_0x0000d1a3"
    ADAPTERS = [("Intel(R) UHD Graphics", 0x8086, 128 << 20, 0, 0x1111),
                ("AMD Radeon(TM) 8060S Graphics", 0x1002, 96 << 30, 0, 0xD1A3)]

    class FakePdh:
        def __init__(self, values):
            self.v = values

        def ok(self):
            return True

        def values(self):
            return self.v

    VALUES = {
        "engine": [("pid_10_luid_0x00000000_0x0000D1A3_phys_0_eng_0_engtype_3D", 30.0),
                   ("pid_11_luid_0x00000000_0x0000D1A3_phys_0_eng_0_engtype_3D", 25.5),    # same engine: summed
                   ("pid_10_luid_0x00000000_0x0000D1A3_phys_0_eng_3_engtype_Compute", 40.0),
                   ("pid_12_luid_0x00000000_0x00001111_phys_0_eng_0_engtype_3D", 99.0)],   # another adapter
        "dedicated": [("luid_0x00000000_0x0000D1A3_phys_0", 90.0 * 2**30),
                      ("luid_0x00000000_0x00001111_phys_0", 1.0 * 2**20)],
        "shared": [("luid_0x00000000_0x0000D1A3_phys_0", 2.0 * 2**30)]}

    def test_reading_of_one_adapter(self):
        r = T.win_gpu_reading(self.VALUES, self.LUID)
        self.assertEqual(r, {"util": 55.5, "mem_used": 90 << 30, "mem_shared": 2 << 30})
        self.assertEqual(T.win_gpu_reading({}, self.LUID), {"util": None, "mem_used": None, "mem_shared": None})

    def test_card_by_amd_index(self):
        g = T._WinGpu(0, adapters=self.ADAPTERS, pdh=self.FakePdh(self.VALUES))
        self.assertTrue(g.ok())
        self.assertEqual((g.name(), g.pci(), g.luid), ("AMD Radeon(TM) 8060S Graphics", None, self.LUID))
        r = g.read()
        self.assertEqual((r["util"], r["mem_used"], r["mem_total"]), (55.5, 90 << 30, 96 << 30))
        self.assertFalse(T._WinGpu(1, adapters=self.ADAPTERS, pdh=self.FakePdh(self.VALUES)).ok())   # one AMD card
        self.assertFalse(T._WinGpu(0, adapters=[]).ok())

    def test_windows_picks_it_when_sysfs_has_no_card(self):
        fake = T._WinGpu(0, adapters=self.ADAPTERS, pdh=self.FakePdh(self.VALUES))
        with mock.patch.object(T, "SYSFS", "/nonexistent-sysfs"), mock.patch.object(T, "_WinGpu", return_value=fake), \
                mock.patch.object(T, "_reader", side_effect=AssertionError("NVML for an AMD card")):
            with mock.patch.object(T, "WINDOWS", True):
                t = T.Telemetry(gpu_index=0, amd=True, start=False)
                s = t.sample()
                self.assertEqual(T.free_vram_mib(0, amd=True), 6 << 10)
            self.assertEqual((t.static["gpu_name"], s["gpu_util"], s["gpu_mem_used"], s["gpu_mem_total"]),
                             ("AMD Radeon(TM) 8060S Graphics", 55.5, 90 << 30, 96 << 30))
            self.assertIsNone(s.get("gpu_temp"))
            # Linux keeps the sysfs reader only: no card there means no AMD readings
            with mock.patch.object(T, "WINDOWS", False):
                self.assertFalse(T.amd_card(0).ok())


if __name__ == "__main__":
    unittest.main()
