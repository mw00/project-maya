"""tools/test_ling_fit.py - Unit tests for the unified Ling-3.2-air planner (tools/ling_fit.py)
and its parity with serve/models.py.
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import ling_fit as LINGF  # noqa: E402
from serve import models  # noqa: E402


class TestLingFit(unittest.TestCase):
    def test_exact_kv_and_kda_bytes(self):
        self.assertEqual(LINGF.KV_BYTES["f16"], 8064)
        self.assertEqual(LINGF.KV_BYTES["q8"], 4704)
        self.assertEqual(LINGF.DEFAULT_KV, "f16")
        self.assertAlmostEqual(LINGF.kv_gib(262144, "f16"), 1.96875, places=4)
        self.assertAlmostEqual(LINGF.kv_gib(262144, "q8"), 1.1484375, places=4)
        self.assertAlmostEqual(LINGF.KDA_STATE_GIB, 0.073165, places=4)

    def test_single_and_multi_gpu_plans(self):
        # 1x 24 GB at 32K FP16 with ~55.6 GiB GGUF: ~6400-7000 VRAM expert slots, rest in RAM
        p1 = LINGF.plan(24.0, 32768, "f16", ram_gb=128.0, gguf_gib=55.6)
        self.assertTrue(p1["fits_trunk"])
        self.assertFalse(p1["resident"])
        self.assertGreater(p1["vram_slots"], 5000)
        self.assertEqual(p1["ssd_slots"], 0)
        self.assertEqual(p1["vision_permanent_gib"], 0.0)

        # 4x 24 GB at 32K FP16: ALL 20,480 experts fit in VRAM!
        p4 = LINGF.plan([24.0, 24.0, 24.0, 24.0], 32768, "f16", ram_gb=128.0, gguf_gib=55.6)
        self.assertTrue(p4["fits_trunk"])
        self.assertTrue(p4["resident"])
        self.assertEqual(p4["vram_slots"], LINGF.TOTAL_TRUNK_EXPERTS)

    def test_emulated_small_cards_and_ram_cap_ssd(self):
        # Emulated 8 GB card: zero-expert VRAM pool must still fit trunk & say "runs slowly"
        p8 = LINGF.plan(8.0, 131072, "f16", ram_gb=128.0, reserve_gb=1.2, gguf_gib=55.6)
        self.assertTrue(p8["fits_trunk"])
        self.assertEqual(p8["vram_slots"], 0)
        word, line = LINGF.verdict(8.0, 128.0, 131072, "f16", reserve_gb=1.2, gguf_gib=55.6)
        self.assertEqual(word, "runs slowly")
        self.assertIn("zero-expert", line)

        # RAM below model size -> SSD streaming verdict
        word_ssd, line_ssd = LINGF.verdict(24.0, 32.0, 32768, "f16", gguf_gib=55.6)
        self.assertEqual(word_ssd, "runs slowly")
        self.assertIn("NVMe SSD", line_ssd)
        # (the hint names the smaller size that exists - the table's - with what it changes)
        self.assertIn("try Ling-3.2-air-Maya-S", line_ssd)
        self.assertIn("SSD", line_ssd.split("try ", 1)[1])
        with mock.patch.object(LINGF, "download_manifest", lambda: {"quants": {}}):   # (no size listed: the plain hint)
            self.assertIn("smaller quant", LINGF.verdict(24.0, 32.0, 32768, "f16", gguf_gib=55.6)[1])

        # Card too small -> "does not fit"
        word_no, _ = LINGF.verdict(4.0, 64.0, 32768, "f16", gguf_gib=55.6)
        self.assertEqual(word_no, "does not fit")

    def test_windows_nvidia_supported_amd_refused(self):
        # (v1.1.83: Windows + NVIDIA as Linux, the same card rules; AMD on Windows refused in one line)
        self.assertIsNone(LINGF.unsupported("cuda", True, [{"vram_gb": 8, "cc": 86}], 64, True))
        self.assertIn("sm_70", LINGF.unsupported("cuda", True, [{"vram_gb": 24, "cc": 61}], 128, True))
        self.assertIn("VRAM", LINGF.unsupported("cuda", True, [{"vram_gb": 6, "cc": 86}], 128, True))
        self.assertIn("AVX2", LINGF.unsupported("cuda", True, [{"vram_gb": 24, "cc": 86}], 128, False))
        amd = LINGF.unsupported("hip", True, [{"vram_gb": 24}], 128, True)
        self.assertIn("Windows", amd)
        self.assertIn("AMD", amd)

    def test_unsupported_refusals(self):
        self.assertIsNotNone(LINGF.unsupported("hip", True, [{"vram_gb": 24, "cc": 86}], 128, True))
        self.assertIsNotNone(LINGF.unsupported("cuda", False, [{"vram_gb": 24, "cc": 61}], 128, True))
        self.assertIsNotNone(LINGF.unsupported("cuda", False, [{"vram_gb": 6, "cc": 86}], 128, True))
        self.assertIsNotNone(LINGF.unsupported("cuda", False, [{"vram_gb": 24, "cc": 86}], 128, False))
        self.assertIsNone(LINGF.unsupported("cuda", False, [{"vram_gb": 24, "cc": 86}], 128, True))

    def test_amd_linux(self):
        # AMD (HIP) on Linux: one RDNA 3 / 3.5 / 4 GPU - a card of 8 GB or more, or an APU whose memory is the RAM
        apu = {"name": "AMD Radeon 8065S", "arch": "gfx1151", "vram_gb": 0.5, "unified_memory": True}
        card = {"name": "Radeon RX 7900 XT", "arch": "gfx1100", "vram_gb": 20.0}
        for gpus in ([apu], [card], [dict(card, arch="gfx1201")], []):
            self.assertIsNone(LINGF.unsupported("hip", False, gpus, 187, True), gpus)
        for gpus in ([dict(card, arch="gfx1030")], [dict(card, arch="gfx942")], [dict(card, vram_gb=6.0)],
                     [card, card]):
            self.assertIsNotNone(LINGF.unsupported("hip", False, gpus, 187, True), gpus)
        self.assertIn("Linux only", LINGF.unsupported("hip", True, [card], 64, True))
        self.assertIn("AVX2", LINGF.unsupported("hip", False, [apu], 187, False))
        # (no NVIDIA rule reaches an AMD card: it has no compute capability)
        self.assertIsNone(LINGF.unsupported("hip", False, [dict(card, cc=11)], 64, True))

    def test_models_py_uses_same_planner_and_defaults(self):
        cfg = {
            "exe": str(ROOT / "build" / "maya-ling"),
            "backend": "cuda",
            "gpu": [0],
            "args": ["--serve", "--gguf", "/tmp/Ling-3.2-air.gguf", "--max-context", "32768"],
            "installer": {"family": "ling", "quant": "local"},
        }
        shown = models.shown_settings(cfg)
        self.assertEqual(shown["kv"], "f16")
        self.assertEqual(shown["parallel"], 1)
        with self.assertRaises(ValueError) as ctx:
            models.apply(cfg, {"parallel": 2})
        self.assertIn("one request at a time", str(ctx.exception))
        applied = models.apply(cfg, {"parallel": 1, "kv": "q8"})
        self.assertIn("--kv", applied["args"])
        self.assertEqual(applied["args"][applied["args"].index("--kv") + 1], "q8")


if __name__ == "__main__":
    unittest.main()
