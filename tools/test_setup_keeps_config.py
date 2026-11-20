"""#1066: running the setup again rewrote an existing config - it added --ablate (the refusal direction a new config
gets) and GPU 1 (the GPU question's recommended answer, all the cards).  A setup again now keeps what the config had:
its GPUs are the default answer, a config without a direction stays without (--ablation on|off picks anew), a GLM
config's own environment stays, and each config the run changed keeps a copy of itself next to it, said."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ablate_fit as AF  # noqa: E402
import maya  # noqa: E402


def card(i, name="RTX 5090", vram=32, arch="120"):
    return {"index": i, "name": name, "vram_gb": vram, "arch": arch}


class GpusKept(unittest.TestCase):
    """The GPU question: the config before's GPUs are the default (Enter and --yes take them)."""

    def setUp(self):
        for ctx in (patch.object(maya, "say"), patch.object(maya, "ok")):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(gpus=None, gpu=None, yes=True, check=False)
        self.three = [card(0, "Tesla V100", 32, "70"), card(1, "Tesla V100", 32, "70"), card(2, "Tesla V100", 32, "70")]

    def ids(self, gs):
        return [g["index"] for g in gs]

    def test_a_new_setup_takes_the_recommended_gpus(self):
        self.assertEqual(self.ids(maya.choose_gpus(self.a, self.three)), [0, 1, 2])

    def test_a_setup_again_keeps_the_config_gpus(self):
        self.assertEqual(self.ids(maya.choose_gpus(self.a, self.three, before=[0])), [0])
        self.assertEqual(self.ids(maya.choose_gpus(self.a, self.three, before=[0, 1])), [0, 1])
        self.assertEqual(self.ids(maya.choose_gpus(self.a, self.three, before=[2, 0])), [2, 0])   # (any set, its order)

    def test_the_setups_hand_it_over_on_the_answers(self):
        # (set_up_* set a.gpus_before from the config before; check_pc's question reads it)
        a = SimpleNamespace(gpus=None, gpu=None, yes=True, check=False, gpus_before=[1])
        self.assertEqual(self.ids(maya.choose_gpus(a, self.three)), [1])

    def test_the_users_two_cards_case(self):
        # #1066: an RTX 5090 (GPU 0) + an RTX 4070 SUPER (GPU 1), the config on the 5090 alone
        pcs = [card(0), card(1, "RTX 4070 SUPER", 12, "89")]
        self.assertEqual(self.ids(maya.choose_gpus(self.a, pcs, before=[0])), [0])
        self.assertEqual(self.ids(maya.choose_gpus(self.a, pcs)), [0, 1])

    def test_a_card_that_is_gone_or_given_now(self):
        self.assertEqual(self.ids(maya.choose_gpus(self.a, self.three, before=[0, 5])), [0, 1, 2])   # (5 is gone)
        a = SimpleNamespace(gpus="1,2", gpu=None, yes=True, check=False)
        self.assertEqual(self.ids(maya.choose_gpus(a, self.three, before=[0])), [1, 2])               # (--gpus wins)

    def test_asked_the_config_gpus_are_the_default_answer(self):
        seen = {}

        def fake_choose(q, intro, options, default, yes, outro=()):
            seen["labels"], seen["default"] = [o[0] for o in options], default
            seen["recommended"] = [i for i, o in enumerate(options) if o[1] == "recommended"]
            return default
        a = SimpleNamespace(gpus=None, gpu=None, yes=False, check=False)
        with patch.object(maya, "choose", side_effect=fake_choose):
            self.assertEqual(self.ids(maya.choose_gpus(a, self.three, before=[0])), [0])
        self.assertIn("as set up before", seen["labels"][seen["default"]])
        self.assertEqual(seen["recommended"], [0])                       # (all three: still marked recommended)


class AblationKept(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.models = self.d / "models"
        (self.models / maya.ABLATE_DIR).mkdir(parents=True)
        self.direction = AF.write_direction(self.models / maya.ABLATE_DIR / "DeepSeek-V4.1-Flash-refusal-direction.gguf",
                                            np.ones(16), "DeepSeek-V4.1-Flash", "Maya-S", 30)
        self.cfg = self.d / "maya-ds4.json"
        for ctx in (patch.object(maya, "say"), patch.object(maya, "ok"),
                    patch.object(maya, "fetch_ablate_direction", return_value=self.direction)):
            ctx.start()
            self.addCleanup(ctx.stop)

    def new_config(self):
        """What the DeepSeek writer leaves before ablate_step: no direction (the writers start anew)."""
        self.cfg.write_text(json.dumps({"installer": {"family": "deepseek"}, "args": ["--pack", "/p"]}))

    def run_step(self, before: dict | None, ablation=None):
        self.new_config()
        with patch.object(maya, "SETUP_BEFORE", {} if before is None else {str(self.cfg.resolve()): json.dumps(before)}):
            maya.ablate_step(self.cfg, "deepseek", self.models, SimpleNamespace(ablation=ablation))
        return json.loads(self.cfg.read_text())

    def test_a_new_config_loads_the_direction(self):
        c = self.run_step(None)
        self.assertIn("--ablate", c["args"])
        self.assertEqual(c["installer"]["ablate"], "loaded")

    def test_a_setup_again_over_a_config_without_one_keeps_it_without(self):
        c = self.run_step({"installer": {"family": "deepseek"}, "args": ["--pack", "/p"]})
        self.assertNotIn("--ablate", c["args"])
        self.assertNotIn("ablate", c)
        self.assertEqual(c["installer"]["ablate"], "off")
        c = self.run_step(c)                                              # (and the next setup again too)
        self.assertNotIn("--ablate", c["args"])

    def test_ablation_on_and_off_pick_anew(self):
        c = self.run_step({"args": ["--pack", "/p"]}, ablation="on")
        self.assertIn("--ablate", c["args"])
        c = self.run_step(c, ablation="off")
        self.assertNotIn("--ablate", c["args"])
        self.assertEqual(c["installer"]["ablate"], "off")

    def test_a_config_whose_setup_found_none_tries_again(self):
        c = self.run_step({"args": ["--pack", "/p"], "installer": {"ablate": "none"}})
        self.assertIn("--ablate", c["args"])

    def test_the_models_page_default_stays(self):
        before = {"args": ["--pack", "/p", "--ablate", str(self.direction), "--ablate-default", "off"],
                  "ablate": {"file": str(self.direction), "default": True, "scale": 0.5}}
        c = self.run_step(before)
        self.assertTrue(c["ablate"]["default"])
        self.assertEqual(c["ablate"]["scale"], 0.5)
        self.assertEqual(c["args"][c["args"].index("--ablate-scale") + 1], "0.5")

    def test_a_file_set_by_hand_is_written_back(self):
        hand = AF.write_direction(self.d / "mine.gguf", np.ones(16), "DeepSeek-V4.1-Flash", "Maya-S", 30)
        c = self.run_step({"args": ["--pack", "/p", "--ablate", str(hand)], "ablate": {"file": str(hand)}})
        self.assertEqual(c["ablate"]["file"], str(hand))
        self.assertEqual(c["args"][c["args"].index("--ablate") + 1], str(hand))


class Backups(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.said = []
        for ctx in (patch.object(maya, "ROOT", self.root), patch.object(maya, "say"),
                    patch.object(maya, "ok", side_effect=lambda m, *x: self.said.append(m))):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.cfg = self.root / "maya-ds4.json"
        self.old = {"exe": "build/maya-ds4", "gpu": [0], "args": ["--pack", "/p", "--max-context", "65536"],
                    "env": {"MY_VAR": "1"}, "installer": {"family": "deepseek", "written": "2026-10-23 16:57"}}
        self.cfg.write_text(json.dumps(self.old, indent=1))

    def setup_writes(self, new):
        def fake(a, prev):
            self.cfg.write_text(json.dumps(new, indent=1))
            return self.cfg
        with patch.object(maya, "_set_up", side_effect=fake):
            return maya.set_up(SimpleNamespace(), {})

    def test_a_changed_config_keeps_a_copy_and_says_what_changed(self):
        new = {**self.old, "gpu": [0, 1], "args": self.old["args"] + ["--ablate", "/d.gguf", "--ablate-default", "off"],
               "installer": {**self.old["installer"], "written": "2026-11-20 19:00"}}
        self.assertEqual(self.setup_writes(new), self.cfg)
        baks = sorted(self.root.glob("maya-ds4.json.bak-*"))
        self.assertEqual(len(baks), 1)
        self.assertEqual(json.loads(baks[0].read_text()), self.old)
        line = next(m for m in self.said if "changed by this setup" in m)
        self.assertIn("GPUs [0] -> [0, 1]", line)
        self.assertIn("--ablate added", line)
        self.assertIn(baks[0].name, line)
        self.assertEqual([p.name for p in maya.configs()], ["maya-ds4.json"])   # (the copy is not a config)

    def test_only_the_date_changed_no_copy(self):
        self.setup_writes({**self.old, "installer": {**self.old["installer"], "written": "2026-11-20 19:00"}})
        self.assertEqual(list(self.root.glob("*.bak-*")), [])

    def test_a_setup_that_stops_half_way_still_keeps_the_copy(self):
        def fake(a, prev):
            self.cfg.write_text(json.dumps({**self.old, "gpu": [1]}))
            raise SystemExit(1)
        with patch.object(maya, "_set_up", side_effect=fake), self.assertRaises(SystemExit):
            maya.set_up(SimpleNamespace(), {})
        self.assertEqual(len(list(self.root.glob("maya-ds4.json.bak-*"))), 1)

    def test_the_models_page_delete_takes_the_copies_too(self):
        from serve import models as SM
        (self.root / "maya-ds4.json.bak-20261120-190000").write_text("{}")
        SM.delete(self.root, self.cfg, False)
        self.assertEqual(list(self.root.iterdir()), [])


class GlmEnvKept(unittest.TestCase):
    """A GLM setup again keeps the config's own variables (the other models' setups did already); the ones the setup and
    the tuning work out are worked out again."""

    def setUp(self):
        sys.path.insert(0, str(HERE))
        from test_maya_prefill import gguf
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "models" / "mine"
        (self.model / "pack").mkdir(parents=True)
        self.first = gguf(self.model / "GLM-5.3-Flash-test.gguf",
                          ["token_embd.weight"] + [f"blk.{i}.ffn_norm.weight" for i in range(250)])
        for ctx in (patch.object(maya, "ROOT", self.root), patch.object(maya, "EXE", self.root / "build/maya"),
                    patch.object(maya, "CAL_STORE", self.root / "calibration.json"), patch.object(maya, "WIN", False),
                    patch.object(maya, "say"), patch.object(maya, "step"),
                    patch.object(maya.S, "gpus", return_value=[]),
                    patch.object(maya.S, "cpu_info", return_value=("test CPU", True, True)),
                    patch.object(maya.S, "ram_gb", return_value=177),
                    patch.object(maya, "mem_gb", return_value=(177.0, 150.0))):
            ctx.start()
            self.addCleanup(ctx.stop)

    def write(self, env=()):
        a = SimpleNamespace(env=list(env), port=None, host=None, api_key=None, gguf_dir=None)
        pc = {"gpus": [{"index": 0, "name": "GPU", "vram_gb": 24}]}
        return maya.write_config(a, pc, {}, self.model / "pack", "test", 32768, self.root / "models", None,
                                 ("local", self.first))

    def test_own_variables_stay_and_the_setups_are_worked_out_again(self):
        p = self.write()
        c = maya.read_json(p)
        c["env"] = {"MY_VAR": "1", "MAYA_GLM_PCIE_SHARE": "0.25", "MAYA_GLM_MTP_DECODE": "0", "OTHER": "x"}
        p.write_text(json.dumps(c))
        env = maya.read_json(self.write(["OTHER=y"])).get("env") or {}
        self.assertEqual(env.get("MY_VAR"), "1")                         # (the user's own: kept)
        self.assertEqual(env.get("OTHER"), "y")                          # (an --env given now wins)
        self.assertNotIn("MAYA_GLM_PCIE_SHARE", env)                     # (the tuning's: none saved for this PC)
        self.assertNotIn("MAYA_GLM_MTP_DECODE", env)                     # (the setup's / speed check's own)


if __name__ == "__main__":
    unittest.main()
