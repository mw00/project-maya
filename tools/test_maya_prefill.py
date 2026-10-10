"""Maya's prompt-chunk setting in the configs it writes (Strata's --prefill engine arg: auto, kept when edited),
Strata's --prefill tips, and KV streaming (--kv-streaming on in every config); no GPU, network or
model downloads required.

    python -m unittest tools.test_maya_prefill
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import maya
from gguf_writer import GGUFWriter


def gguf(path: Path, names: list[str]) -> Path:
    w = GGUFWriter()
    w.add("general.architecture", "glm5next")
    for n in names:
        w.add_f32(n, np.zeros(4, dtype=np.float32))
    return w.write(path)


class PrefillConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "models" / "mine"
        (self.model / "pack").mkdir(parents=True)
        # a "model": its embedding and many layers' tensors
        self.first = gguf(self.model / "GLM-5.3-Flash-test.gguf",
                          ["token_embd.weight"] + [f"blk.{i}.ffn_norm.weight" for i in range(250)])
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "EXE", self.root / "build/strata"),
            patch.object(maya, "CAL_STORE", self.root / "calibration.json"),
            patch.object(maya, "WIN", False),
            patch.object(maya, "say"),
            patch.object(maya, "step"),
            patch.object(maya.S, "gpus", return_value=[]),
            patch.object(maya.S, "cpu_info", return_value=("test CPU", True, True)),
            patch.object(maya.S, "ram_gb", return_value=177),
            patch.object(maya, "mem_gb", return_value=(177.0, 150.0)),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(env=[], port=None, host=None, api_key=None, gguf_dir=None)

    def write(self, gpus, ctx=32768):
        pc = {"gpus": [{"index": i, "name": "GPU", "vram_gb": 24} for i in gpus]}
        p = maya.write_config(self.a, pc, {}, self.model / "pack", "test", ctx, self.root / "models", None,
                              ("local", self.first))
        return p, json.loads(p.read_text())["args"]

    def test_tuned_settings_follow_a_new_context(self):
        # #57: tuned at 32K, a setup for 128K wrote the engine's own settings (half the speed there)
        p, _ = self.write([0])
        store = {maya.hardware_key(json.loads(p.read_text())): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"},
                                                                  "date": "2026-10-09"}}
        maya.CAL_STORE.write_text(json.dumps(store))
        p, _ = self.write([0], 131072)
        c = json.loads(p.read_text())
        self.assertEqual(c["env"]["STRATA_GLM_PCIE_SHARE"], "0.25")
        self.assertIsNone(maya.saved_calibration(c))                        # this context is not tuned itself
        self.assertEqual(maya.saved_calibration(c, any_context=True)["context"], 32768)
        # the nearest tuned context wins, and this context's own tuning over any other
        store[maya.hardware_key(c).replace("|131072|", "|65536|")] = {"settings": {"STRATA_GLM_PCIE_SHARE": "0.30"}}
        maya.CAL_STORE.write_text(json.dumps(store))
        self.assertEqual(maya.saved_calibration(c, any_context=True)["settings"]["STRATA_GLM_PCIE_SHARE"], "0.30")
        store[maya.hardware_key(c)] = {"settings": {"STRATA_GLM_PCIE_SHARE": "0.35"}}
        maya.CAL_STORE.write_text(json.dumps(store))
        self.assertNotIn("context", maya.saved_calibration(c, any_context=True))
        # another model on the same PC takes nothing from this one
        self.assertIsNone(maya.saved_calibration(dict(c, installer=dict(c["installer"], quant="other")),
                                                 any_context=True))

    def test_prefill_auto_is_written_and_an_edit_kept(self):
        for gpus in ([0], [0, 1]):
            p, args = self.write(gpus)
            self.assertEqual(args[args.index("--prefill") + 1], "auto")
            p.unlink()
        p, args = self.write([0])
        c = json.loads(p.read_text())
        c["args"][c["args"].index("--prefill") + 1] = "32768"  # a bigger chunk, set by hand
        p.write_text(json.dumps(c))
        _, again = self.write([0])
        self.assertEqual(again[again.index("--prefill") + 1], "32768")
        self.assertEqual(again.count("--prefill"), 1)

    def test_prefill_tips_follow_strata(self):
        self.assertEqual(maya.prefill_tips(["--prefill", "auto"], 177, 1), [])      # one GPU: auto takes 32768
        self.assertEqual(maya.prefill_tips(["--prefill", "32768"], 32, 1), [])
        self.assertIn("--prefill 32768", maya.prefill_tips(["--prefill", "auto"], 177)[0])
        self.assertEqual(maya.prefill_tips(["--prefill", "auto"], 64), [])
        self.assertIn("warning", maya.prefill_tips(["--prefill", "32768"], 32)[0])
        self.assertEqual(maya.prefill_tips(["--prefill", "32768"], 177), [])
        self.assertEqual(maya.prefill_tips(["--prefill", "4096"], 32), [])

    def test_kv_streaming_in_every_config(self):
        # "--kv-streaming", "on" in every config, whatever the context or the PC
        for ctx in (32768, 131072):
            p, args = self.write([0], ctx)
            self.assertEqual(args[args.index("--kv-streaming") + 1], "on")
            self.assertEqual(args.count("--kv-streaming"), 1)
            self.assertNotIn("--kv-resident", args)
            p.unlink()
        # what was set by hand (off, or a window of its own) is kept by a setup again
        p, _ = self.write([0], 131072)
        c = json.loads(p.read_text())
        c["args"][c["args"].index("--kv-streaming") + 1] = "off"
        c["args"] += ["--kv-resident", "65536"]
        p.write_text(json.dumps(c))
        _, again = self.write([0], 131072)
        self.assertEqual(again[again.index("--kv-streaming") + 1], "off")
        self.assertEqual(again[again.index("--kv-resident") + 1], "65536")
        self.assertEqual(again.count("--kv-streaming"), 1)


if __name__ == "__main__":
    unittest.main()
