"""setup.py's unified-memory detection (issue #671): a discrete NVIDIA card on a PC without Resizable BAR (BAR1 of 256 MiB)
is not "shared with the system"; a GB10 / Grace-class device whose memory is the system's still is.

    python -m unittest tools.test_unified_memory -v
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import setup as S  # noqa: E402

A4000_X2 = ("0, NVIDIA RTX A4000, 16376, 8.6, 550.120, 00000000:17:00.0\n"
            "1, NVIDIA RTX A4000, 16376, 8.6, 550.120, 00000000:65:00.0\n")
GB10 = "0, NVIDIA GB10, [N/A], 12.1, 580.95, 0000000F:01:00.0\n"
GB10_NUMBERED = "0, NVIDIA GB10, 122570, 12.1, 580.95, 0000000F:01:00.0\n"      # (a driver that did report a number)
SMALL_BAR = {"0000:17:00.0", "0000:65:00.0"}
GB10_ID = {"000f:01:00.0"}


class UnifiedMemory(unittest.TestCase):
    def found(self, smi, candidates, integrated):
        with patch.object(S, "out", return_value=smi), patch.object(S, "unified_memory_pci_ids", return_value=candidates), \
             patch.object(S, "cuda_integrated_pci", return_value=integrated), patch.object(S, "ram_gb", return_value=119.7):
            return S.gpus()

    def test_a_discrete_card_without_resizable_bar_is_not_shared(self):
        for integrated in ({"0000:17:00.0": False, "0000:65:00.0": False}, {}):     # (CUDA says no / cannot be asked)
            with self.subTest(cuda=integrated):
                g = self.found(A4000_X2, SMALL_BAR, integrated)
                self.assertEqual([x["unified_memory"] for x in g], [False, False])
                self.assertEqual([round(x["vram_gb"]) for x in g], [16, 16])         # nvidia-smi's own number

    def test_a_card_with_a_big_bar_is_not_a_candidate(self):
        g = self.found(A4000_X2, set(), {})
        self.assertEqual([x["unified_memory"] for x in g], [False, False])

    def test_the_gb10_is_shared_with_the_system(self):
        for integrated in ({"000f:01:00.0": True}, {}):                             # (CUDA says yes / cannot be asked)
            with self.subTest(cuda=integrated):
                g = self.found(GB10, GB10_ID, integrated)
                self.assertTrue(g[0]["unified_memory"])
                self.assertAlmostEqual(g[0]["vram_gb"], 119.7)                      # the RAM

    def test_cuda_decides_when_it_can_be_asked(self):
        g = self.found(GB10_NUMBERED, GB10_ID, {"000f:01:00.0": True})              # numbered memory, integrated
        self.assertTrue(g[0]["unified_memory"])
        self.assertAlmostEqual(g[0]["vram_gb"], 122570 / 1024.0)
        g = self.found(GB10, GB10_ID, {"000f:01:00.0": False})                       # no number, but CUDA: not integrated
        self.assertFalse(g[0]["unified_memory"])

    def test_a_mixed_box(self):
        smi = A4000_X2.splitlines()[0] + "\n" + "1, NVIDIA GB10, [N/A], 12.1, 580.95, 0000000F:01:00.0\n"
        g = self.found(smi, SMALL_BAR | GB10_ID, {"0000:17:00.0": False, "000f:01:00.0": True})
        self.assertEqual([x["unified_memory"] for x in g], [False, True])

    def test_cuda_integrated_pci_reads_the_driver(self):
        class Lib:
            def cuInit(self, flags): return 0
            def cuDeviceGetCount(self, p): p._obj.value = 2; return 0
            def cuDeviceGet(self, p, i): p._obj.value = i; return 0
            def cuDeviceGetPCIBusId(self, buf, n, dev): buf.value = (b"0000:17:00.0", b"000F:01:00.0")[dev.value]; return 0
            def cuDeviceGetAttribute(self, p, attr, dev): p._obj.value = int(dev.value == 1 and attr == 18); return 0
        with patch.object(S.ctypes, "CDLL", return_value=Lib()), patch.object(S, "WIN", False), patch.object(S.sys, "platform", "linux"):
            self.assertEqual(S.cuda_integrated_pci(), {"0000:17:00.0": False, "000f:01:00.0": True})

    def test_cuda_integrated_pci_without_a_driver_is_empty(self):
        with patch.object(S.ctypes, "CDLL", side_effect=OSError("no libcuda")), patch.object(S, "WIN", False), \
             patch.object(S.sys, "platform", "linux"):
            self.assertEqual(S.cuda_integrated_pci(), {})


if __name__ == "__main__":
    unittest.main()
