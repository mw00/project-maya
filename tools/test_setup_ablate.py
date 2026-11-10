"""The setup's refusal direction (maya.ablate_step): downloaded from the model's own repo and recorded as the config's
"ablate" with --ablate FILE --ablate-default off; a repo without one, no connection or a wrong file - the setup goes
on without it (the switch stays off); an earlier copy and a file set by hand are kept."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ablate_fit as AF  # noqa: E402
import maya  # noqa: E402


class AblateStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.models = self.d / "models"
        self.cfg = self.d / "maya-ds4.json"
        self.cfg.write_text(json.dumps({"installer": {"family": "deepseek"}, "args": ["--pack", "/p", "--ctx", "8192"]}))
        self.good = AF.write_direction(self.d / "good.gguf", np.ones(16), "DeepSeek-V4.1-Flash", "Maya-S", 30)
        self.other = AF.write_direction(self.d / "other.gguf", np.ones(16), "MiniMax-M3.1-Flash", "Maya-S", 30)
        self.urls = []

    def tearDown(self):
        self.tmp.cleanup()

    def curl(self, src: pathlib.Path | None, rc: int = 0):
        def run(cmd, **kw):
            self.urls.append(cmd[-1])
            if src is not None:
                pathlib.Path(cmd[cmd.index("-o") + 1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(cmd, rc, "", "" if rc in (0, 22) else "curl: (6) Could not resolve host")
        return patch.object(maya.subprocess, "run", side_effect=run)

    def step(self, src, rc=0):
        with self.curl(src, rc), patch.object(maya.shutil, "which", return_value="curl"), patch.object(maya, "ok"):
            maya.ablate_step(self.cfg, "deepseek", self.models)
        return json.loads(self.cfg.read_text())

    def test_downloaded_and_recorded_off_by_default(self):
        cfg = self.step(self.good)
        f = self.models / "refusal-directions" / "DeepSeek-V4.1-Flash-refusal-direction.gguf"
        self.assertTrue(f.is_file())
        self.assertEqual(self.urls, ["https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/resolve/main/"
                                     "DeepSeek-V4.1-Flash-refusal-direction.gguf"])
        self.assertEqual(cfg["ablate"], {"file": str(f), "model": "DeepSeek-V4.1-Flash", "fitted_on": "Maya-S"})
        a = cfg["args"]
        self.assertEqual(a[a.index("--ablate") + 1], str(f))
        self.assertEqual(a[a.index("--ablate-default") + 1], "off")
        self.assertFalse(list(f.parent.glob("*.part")))

    def test_a_repo_without_one_changes_nothing(self):
        cfg = self.step(None, 22)
        self.assertNotIn("ablate", cfg)
        self.assertEqual(cfg["args"], ["--pack", "/p", "--ctx", "8192"])
        self.assertFalse(list((self.models / "refusal-directions").glob("*")))

    def test_a_wrong_file_is_not_taken(self):
        cfg = self.step(self.other)
        self.assertNotIn("ablate", cfg)
        self.assertNotIn("--ablate", cfg["args"])
        self.assertFalse(list((self.models / "refusal-directions").glob("*")))

    def test_offline_keeps_the_earlier_copy_and_the_default(self):
        self.step(self.good)
        cfg = json.loads(self.cfg.read_text())
        cfg["ablate"]["default"] = True
        self.cfg.write_text(json.dumps(cfg))
        cfg = self.step(None, 6)
        self.assertTrue(cfg["ablate"]["default"])
        self.assertIn("--ablate", cfg["args"])

    def test_a_file_set_by_hand_stays(self):
        cfg = json.loads(self.cfg.read_text())
        cfg["ablate"] = {"file": str(self.good)}
        cfg["args"] += ["--ablate", str(self.good), "--ablate-default", "off"]
        self.cfg.write_text(json.dumps(cfg))
        with patch.object(maya.subprocess, "run", side_effect=AssertionError("no download")), patch.object(maya, "ok"):
            maya.ablate_step(self.cfg, "deepseek", self.models)
        self.assertEqual(json.loads(self.cfg.read_text())["ablate"], {"file": str(self.good)})

    def test_every_model_has_its_own_repo_and_name(self):
        self.assertEqual(set(maya.ABLATE_DIRECTIONS), {"glm", "glm-dsa", "deepseek", "minimax"})
        for model, repo in maya.ABLATE_DIRECTIONS.values():
            self.assertTrue(repo.startswith("peasantsmith/") and repo.endswith("-Maya-GGUF"), repo)
            self.assertTrue(repo.split("/")[1].startswith(model), (model, repo))


if __name__ == "__main__":
    unittest.main()
