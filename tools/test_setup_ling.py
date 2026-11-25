"""tools/test_setup_ling.py - Unit tests for maya.py set_up_ling(), backup preservation,
calibration reset, and setup-claims vs planner consistency.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import maya  # noqa: E402
import ling_fit as LINGF  # noqa: E402
import maya_quants as Q  # noqa: E402


class TestSetupLing(unittest.TestCase):
    def test_choose_family_and_title(self):
        class A:
            model = maya.LING
            gguf_dir = None
            draft = None
            model_dir = None
        self.assertEqual(maya.choose_family(A(), {}), "ling")
        self.assertEqual(maya.setup_title({"installer": {"family": "ling"}}), "Ling-3.2-air")

    def test_model_resolution(self):
        self.assertEqual(Q.resolve("ling"), ("ling", None))
        self.assertEqual(Q.resolve("Ling-3.2-air"), ("ling", None))
        self.assertEqual(Q.resolve("ling-maya-s"), ("ling", "Ling-3.2-air-Maya-S"))
        self.assertEqual(Q.config_stem("ling", None), "maya-ling-3.2-air")

    def test_write_config_preserves_choices_and_writes_bak(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_root = td / "repo"
            fake_root.mkdir()
            (fake_root / "data").mkdir()
            gguf = td / "Ling-3.2-air-R3-test.gguf"
            gguf.write_bytes(b"GGUF")
            mmproj = td / "Ling-3.2-air-mmproj-BF16.gguf"
            mmproj.write_bytes(b"GGUF")
            tok = td / "tok"
            tok.mkdir()
            (tok / "chat_template.jinja").write_text("tpl", encoding="utf-8")

            class Args:
                port = 8099
                no_vision = False
                env = []
                host = None
                api_key = None

            pc = {"backend": "cuda", "gpus": [{"index": 0, "vram_gb": 24.0, "cc": 86}]}
            with mock.patch.object(maya, "ROOT", fake_root), \
                 mock.patch.object(maya, "LING_EXE", fake_root / "build" / "maya-ling"), \
                 mock.patch.object(maya, "write_run_script", return_value=fake_root / "run.sh"):
                p1 = maya.write_config_ling(Args(), pc, {}, gguf, tok, 32768, "f16", "auto",
                                            td, td, mmproj, "Ling-3.2-air-Maya-S", {})
                c1 = json.loads(p1.read_text(encoding="utf-8"))
                self.assertEqual(c1["port"], 8099)
                self.assertEqual(c1["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
                self.assertEqual(c1["reasoning_effort"], "high")
                self.assertEqual(c1["installer"]["family"], "ling")
                self.assertIn("--kv", c1["args"])
                self.assertEqual(c1["args"][c1["args"].index("--kv") + 1], "fp16")

                # Re-run with a new context -> preserves port/sampling and creates .bak-*
                Args.port = None
                p2 = maya.write_config_ling(Args(), pc, {}, gguf, tok, 65536, "f16", "auto",
                                            td, td, mmproj, "Ling-3.2-air-Maya-S", c1)
                c2 = json.loads(p2.read_text(encoding="utf-8"))
                self.assertEqual(c2["port"], 8099)
                baks = list(fake_root.glob("maya-ling-3.2-air.json.bak-*"))
                self.assertEqual(len(baks), 1)

    def test_reset_ling_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            cf = Path(td) / "ling-calibration.txt"
            cf.write_text("key\tmode=gpu\n", encoding="utf-8")
            cfg = {"installer": {"family": "ling", "cal_file": str(cf)}}
            self.assertTrue(maya.reset_ling_calibration(cfg))
            self.assertFalse(cf.exists())

    def test_download_json_manifest_valid(self):
        m = LINGF.download_manifest()
        self.assertEqual(m["name"], "Ling-3.2-air")
        self.assertEqual(m["default_kv"], "f16")
        self.assertIn("Ling-3.2-air-Maya-S", m["quants"])


if __name__ == "__main__":
    unittest.main()
