"""Maya's expert profile in the configs it writes (Strata's --expert-profile engine arg: the shipped
data/expert-profile-glm.bin, a hand-wired one and --expert-profile-save kept); no GPU, network or model downloads
required.

    python -m unittest tools.test_maya_profile
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


class ProfileArgsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "models" / "mine"
        (self.model / "pack").mkdir(parents=True)
        self.first = gguf(self.model / "GLM-5.3-Flash-test.gguf",
                          ["token_embd.weight"] + [f"blk.{i}.ffn_norm.weight" for i in range(250)])
        (self.root / "data").mkdir()
        self.shipped = self.root / "data" / "expert-profile-glm.bin"
        self.shipped.write_bytes(b"STRP")
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "EXE", self.root / "build/strata"),
            patch.object(maya, "PROFILE", self.shipped),
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

    def write(self):
        pc = {"gpus": [{"index": 0, "name": "GPU", "vram_gb": 24}]}
        p = maya.write_config(self.a, pc, {}, self.model / "pack", "test", 32768, self.root / "models", None,
                              ("local", self.first))
        return p, json.loads(p.read_text())["args"]

    def test_the_shipped_profile_is_passed(self):
        _, args = self.write()
        self.assertEqual(maya.flag_value(args, "--expert-profile"), str(self.shipped))
        self.assertNotIn("--expert-profile-save", args)                    # learning across starts: opt-in

    def test_no_profile_file_no_argument(self):
        self.shipped.unlink()
        self.assertIsNone(maya.flag_value(self.write()[1], "--expert-profile"))

    def test_a_hand_wired_profile_and_save_are_kept(self):
        mine = self.root / "learned.bin"
        mine.write_bytes(b"STRP")
        p, args = self.write()                                # the first setup
        args[args.index("--expert-profile") + 1] = str(mine)  # the user points it at their learned profile ...
        args += ["--expert-profile-save", str(mine), "--expert-profile-save-every", "5"]
        c = json.loads(p.read_text())
        c["args"] = args
        p.write_text(json.dumps(c))
        _, again = self.write()                               # ... and a setup again keeps all three
        self.assertEqual(maya.flag_value(again, "--expert-profile"), str(mine))
        self.assertEqual(maya.flag_value(again, "--expert-profile-save"), str(mine))
        self.assertEqual(maya.flag_value(again, "--expert-profile-save-every"), "5")
        self.assertEqual(again.count("--expert-profile"), 1)

    def test_an_older_config_gains_the_shipped_profile_at_its_next_start(self):
        # an update restarts the installed config without a setup: one written before the profile gets it
        cfg = {"args": ["--glm-pack", "p", "--max-context", "32768"]}
        self.assertTrue(maya.add_profile(cfg))
        self.assertEqual(maya.flag_value(cfg["args"], "--expert-profile"), str(self.shipped))
        self.assertFalse(maya.add_profile(cfg))                            # once
        mine = {"args": ["--glm-pack", "p", "--expert-profile", "/elsewhere/learned.bin"]}
        self.assertFalse(maya.add_profile(mine))                           # a profile of its own stays
        self.assertEqual(maya.flag_value(mine["args"], "--expert-profile"), "/elsewhere/learned.bin")
        self.assertFalse(maya.add_profile({"args": ["--pack", "q"]}))      # not a GLM pack
        self.shipped.unlink()
        self.assertFalse(maya.add_profile({"args": ["--glm-pack", "p"]}))  # no shipped profile: nothing to add

    def test_a_missing_hand_wired_profile_falls_back_to_the_shipped_one(self):
        old = ["--glm-pack", "p", "--expert-profile", str(self.root / "gone.bin")]
        new = ["--glm-pack", "p", "--expert-profile", str(self.shipped)]
        self.assertEqual(maya.keep_profile(old, new), [])
        self.assertEqual(maya.flag_value(new, "--expert-profile"), str(self.shipped))


if __name__ == "__main__":
    unittest.main()
