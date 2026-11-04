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
            patch.object(maya, "EXE", self.root / "build/maya"),
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
        return p, maya.read_json(p)["args"]

    def test_tuned_settings_follow_a_new_context(self):
        # #57: tuned at 32K, a setup for 128K wrote the engine's own settings (half the speed there)
        p, _ = self.write([0])
        store = {maya.hardware_key(maya.read_json(p)): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"},
                                                                  "date": "2026-10-09"}}
        maya.CAL_STORE.write_text(json.dumps(store))
        p, _ = self.write([0], 131072)
        c = maya.read_json(p)
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

    def test_tuning_measured_again_after_a_driver_or_p2p_change(self):
        # #419: STRATA_GLM_PCIE_SHARE tuned before a P2P driver was installed halved the decode, nothing said why
        topo = (" \t\x1b[4mGPU0\tGPU1\t\x1b[0m\n GPU0\tX\t{s}\t\n GPU1\t{s}\tX\t\n\nLegend:\n  X    = Self\n  OK   = Status Ok\n")
        state = {"drv": "595.58", "p2p": "NS"}

        def gpus():
            return [{"index": i, "name": "RTX 3090", "vram_gb": 24, "driver": state["drv"], "pci": f"0000:0{i}:00.0"}
                    for i in (0, 1)]

        def out(cmd):
            return topo.format(s=state["p2p"]) if cmd[1:3] == ["topo", "-p2p"] else ""
        with patch.object(maya.S, "gpus", side_effect=gpus), patch.object(maya.S, "out", side_effect=out):
            cfg = {"gpu": [0, 1], "args": ["--max-context", "32768"], "installer": {"quant": "Maya-S"},
                   "env": {"STRATA_GLM_PCIE_SHARE": "0.25", "STRATA_GLM_CPU_LANE": "12", "STRATA_GLM_FOO": "1"}}
            self.assertEqual(maya.gpu_env(cfg)["p2p"], {"0-1": "NS"})
            # an older config: this PC's state is recorded, nothing changes
            self.assertEqual(maya.stale_tuning(cfg), "")
            self.assertEqual(cfg["installer"]["tuned_on"]["driver"], "595.58")
            self.assertIsNone(maya.stale_tuning(cfg))
            # the store's record: valid while nothing changed
            store = {maya.hardware_key(cfg): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"},
                                              "gpu_env": maya.gpu_env(cfg)}}
            maya.CAL_STORE.write_text(json.dumps(store))
            self.assertIsNotNone(maya.saved_calibration(cfg))
            # the P2P driver: P2P on, a new version - the tuned settings go, the others stay; a setup offers it again
            state.update(drv="615.71", p2p="OK")
            msg = maya.stale_tuning(cfg)
            self.assertIn("driver 595.58 -> 615.71", msg)
            self.assertIn("P2P 0-1:NS -> 0-1:OK", msg)
            self.assertEqual(cfg["env"], {"STRATA_GLM_FOO": "1"})
            self.assertNotIn("tuned_on", cfg["installer"])
            self.assertIsNone(maya.saved_calibration(cfg))
            self.assertIsNone(maya.saved_calibration(cfg, any_context=True))
            # nvidia-smi silent once (no P2P answer): no change seen
            cfg["env"] = {"STRATA_GLM_PCIE_SHARE": "0.3"}
            cfg["installer"]["tuned_on"] = maya.gpu_env(cfg)
            state["p2p"] = ""
            self.assertIsNone(maya.stale_tuning(cfg))
            self.assertEqual(cfg["env"], {"STRATA_GLM_PCIE_SHARE": "0.3"})

    def test_prefill_auto_is_written_and_an_edit_kept(self):
        for gpus in ([0], [0, 1]):
            p, args = self.write(gpus)
            self.assertEqual(args[args.index("--prefill") + 1], "auto")
            p.unlink()
        p, args = self.write([0])
        c = maya.read_json(p)
        c["args"][c["args"].index("--prefill") + 1] = "32768"  # a bigger chunk, set by hand
        p.write_text(json.dumps(c))
        _, again = self.write([0])
        self.assertEqual(again[again.index("--prefill") + 1], "32768")
        self.assertEqual(again.count("--prefill"), 1)

    def test_parallel_is_written_kept_and_given_again(self):
        # --parallel N: absent by default, written when given, kept by a later setup, a new value wins
        _, args = self.write([0])
        self.assertNotIn("--parallel", args)
        self.a.parallel = 4
        _, args = self.write([0])
        self.assertEqual(args[args.index("--parallel") + 1], "4")
        self.a.parallel = None
        _, args = self.write([0])
        self.assertEqual(args[args.index("--parallel") + 1], "4")
        self.a.parallel = 2
        _, args = self.write([0])
        self.assertEqual(args[args.index("--parallel") + 1], "2")
        self.assertEqual(args.count("--parallel"), 1)

    def test_expert_cache_auto_and_the_vram_reserve(self):
        # "--expert-cache", "auto" in every config; a number set by hand (and a reserve) kept by a setup again
        for gpus in ([0], [0, 1]):
            p, args = self.write(gpus)
            self.assertEqual(args[args.index("--expert-cache") + 1], "auto")
            self.assertNotIn("--vram-reserve-mib", args)      # the engine's default unless asked for
            p.unlink()
        p, _ = self.write([0])
        c = maya.read_json(p)
        c["args"][c["args"].index("--expert-cache") + 1] = "1200"
        c["args"] += ["--vram-reserve-later-mib", "300"]
        p.write_text(json.dumps(c))
        _, again = self.write([0])
        self.assertEqual(again[again.index("--expert-cache") + 1], "1200")
        self.assertEqual(again[again.index("--vram-reserve-later-mib") + 1], "300")
        self.assertEqual(again.count("--expert-cache"), 1)
        # --vram-reserve-mib N at setup: written, kept by a later setup, a new value wins
        self.a.vram_reserve_mib = 1500
        _, args = self.write([0])
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "1500")
        self.a.vram_reserve_mib = None
        _, args = self.write([0])
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "1500")
        self.a.vram_reserve_mib = 0
        _, args = self.write([0])
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "0")
        self.assertEqual(args.count("--vram-reserve-mib"), 1)

    def test_dflash2_drafter_in_the_config(self):
        # a DFlash2 setup: --dflash with the drafter's file, the answer kept for the next setup, its own tuning key
        pc = {"gpus": [{"index": i, "name": "GPU", "vram_gb": 24} for i in (0, 1)]}
        drafter = self.root / "models" / "GLM-5.3-Flash-DFlash2" / maya.DFLASH["file"]
        p = maya.write_config(self.a, pc, {}, self.model / "pack", "test", 32768, self.root / "models", None,
                              ("local", self.first), drafter)
        c = maya.read_json(p)
        self.assertEqual(c["args"][c["args"].index("--dflash") + 1], str(drafter))
        self.assertEqual(c["installer"]["draft"], "dflash2")
        key = maya.hardware_key(c)
        # the MTP block (the default): no --dflash, not even one kept from the config before
        p = maya.write_config(self.a, pc, {}, self.model / "pack", "test", 32768, self.root / "models", None,
                              ("local", self.first))
        c = maya.read_json(p)
        self.assertNotIn("--dflash", c["args"])
        self.assertEqual(c["installer"]["draft"], "mtp")
        self.assertNotEqual(maya.hardware_key(c), key)

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
        c = maya.read_json(p)
        c["args"][c["args"].index("--kv-streaming") + 1] = "off"
        c["args"] += ["--kv-resident", "65536"]
        p.write_text(json.dumps(c))
        _, again = self.write([0], 131072)
        self.assertEqual(again[again.index("--kv-streaming") + 1], "off")
        self.assertEqual(again[again.index("--kv-resident") + 1], "65536")
        self.assertEqual(again.count("--kv-streaming"), 1)


if __name__ == "__main__":
    unittest.main()
