"""Reject GLM-only concurrency options before changing another model's config; no GPU needed."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


class ParallelOptionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "model.json"
        self.args = SimpleNamespace(parallel=4, port=None, host=None, api_key=None, gpus=None, gpu=None)
        self.mocks = {}
        for name in ("say", "ok", "select_build_backend", "refresh_engine", "ds4_seed_profile", "add_profile"):
            mocked = patch.object(maya, name, return_value=None)
            self.mocks[name] = mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(maya, "needs_speed_calibration", return_value=False)
        mocked.start()
        self.addCleanup(mocked.stop)

    def config(self, family):
        exe = self.root / {"glm": "maya", "deepseek": "maya-ds4", "minimax": "maya-mm3"}[family]
        exe.touch()
        pack = self.root / "pack"
        pack.mkdir(exist_ok=True)
        tok = self.root / "tokenizer"
        tok.mkdir(exist_ok=True)
        (tok / "vocab.json").write_text("{}", encoding="utf-8")
        cfg = {"exe": str(exe), "tokenizer": str(tok), "gpu": [0],
               "installer": {"family": family},
               "args": ["--glm-pack" if family == "glm" else "--pack", str(pack)]}
        self.path.write_text(json.dumps(cfg), encoding="utf-8")
        return cfg

    def test_start_rejects_parallel_before_mutating_non_glm_config(self):
        for family in ("deepseek", "minimax"):
            for parallel in (1, 4):
                with self.subTest(family=family, parallel=parallel):
                    self.config(family)
                    self.args.parallel = parallel
                    before = self.path.read_bytes()
                    with patch.object(maya, "rename_programs") as rename, \
                            self.assertRaises(SystemExit):
                        maya.server_command(self.path, self.args)
                    rename.assert_not_called()
                    self.assertEqual(self.path.read_bytes(), before)
                    self.mocks["refresh_engine"].assert_not_called()

    def test_legacy_deepseek_config_is_rejected_without_rewriting_it(self):
        cfg = self.config("deepseek")
        cfg.pop("installer")
        cfg["exe"] = str(self.root / "strata-ds4")
        self.path.write_text(json.dumps(cfg), encoding="utf-8")
        before = self.path.read_bytes()
        with patch.object(maya, "rename_programs") as rename, self.assertRaises(SystemExit):
            maya.server_command(self.path, self.args)
        rename.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_non_glm_start_without_parallel_is_unchanged(self):
        self.args.parallel = None
        for family in ("deepseek", "minimax"):
            with self.subTest(family=family):
                cfg = self.config(family)
                with patch.object(maya, "rename_programs"):
                    command, _, _ = maya.server_command(self.path, self.args)
                self.assertIn(str(self.path), command)
                self.assertEqual(maya.read_json(self.path)["args"], cfg["args"])

    def test_glm_start_still_saves_parallel_once(self):
        self.config("glm")
        for parallel in (4, 2):
            self.args.parallel = parallel
            with patch.object(maya, "rename_programs"):
                maya.server_command(self.path, self.args)
            args = maya.read_json(self.path)["args"]
            self.assertEqual(args.count("--parallel"), 1)
            self.assertEqual(args[args.index("--parallel") + 1], str(parallel))

    def test_setup_rejects_parallel_before_non_glm_setup(self):
        for family, setup_name in (("deepseek", "set_up_ds4"), ("minimax", "set_up_mm3")):
            with self.subTest(family=family), \
                    patch.object(maya, "choose_family", return_value=family), \
                    patch.object(maya, setup_name, return_value=None) as setup, \
                    self.assertRaises(SystemExit):
                maya.set_up(self.args, {"installer": {"family": family}})
            setup.assert_not_called()

    def test_cli_rejects_parallel_before_calibration(self):
        for family in ("deepseek", "minimax"):
            for action in ("--calibrate", "--speed-calibration"):
                with self.subTest(family=family, action=action):
                    self.config(family)
                    before = self.path.read_bytes()
                    with patch.dict(os.environ, MAYA_SH="1"), \
                            patch.object(sys, "argv", ["maya.py", "--config", str(self.path),
                                                       "--parallel", "4", action, "--no-start"]), \
                            patch.object(maya, "stop_signals"), \
                            patch.object(maya, "calibrate_config", return_value=True) as tune, \
                            patch.object(maya, "speed_calibration", return_value=True) as speed, \
                            self.assertRaises(SystemExit):
                        maya.main()
                    tune.assert_not_called()
                    speed.assert_not_called()
                    self.mocks["refresh_engine"].assert_not_called()
                    self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
