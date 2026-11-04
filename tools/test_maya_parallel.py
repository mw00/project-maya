"""--parallel: GLM-5.3-Flash, DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash take it (kept in the config); a family that
does not is refused before its config changes; no GPU needed."""
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

    def test_start_rejects_parallel_before_mutating_the_config_of_a_family_without_it(self):
        # (every family of today takes it: the gate stays for the next one)
        for family in ("glm", "deepseek", "minimax"):
            self.assertIn(family, maya.PARALLEL_FAMILIES)
        for parallel in (1, 4):
            with self.subTest(parallel=parallel):
                self.config("minimax")
                self.args.parallel = parallel
                before = self.path.read_bytes()
                with patch.object(maya, "PARALLEL_FAMILIES", ("glm", "deepseek")), \
                        patch.object(maya, "rename_programs") as rename, self.assertRaises(SystemExit):
                    maya.server_command(self.path, self.args)
                rename.assert_not_called()
                self.assertEqual(self.path.read_bytes(), before)
                self.mocks["refresh_engine"].assert_not_called()

    def test_minimax_start_saves_parallel_once(self):
        self.config("minimax")
        for parallel in (4, 2, 1):
            self.args.parallel = parallel
            with patch.object(maya, "rename_programs"):
                maya.server_command(self.path, self.args)
            args = maya.read_json(self.path)["args"]
            self.assertEqual(args.count("--parallel"), 1)
            self.assertEqual(args[args.index("--parallel") + 1], str(parallel))

    def test_deepseek_start_saves_parallel_once(self):
        self.config("deepseek")
        for parallel in (4, 2):
            self.args.parallel = parallel
            with patch.object(maya, "rename_programs"):
                maya.server_command(self.path, self.args)
            args = maya.read_json(self.path)["args"]
            self.assertEqual(args.count("--parallel"), 1)
            self.assertEqual(args[args.index("--parallel") + 1], str(parallel))

    def test_deepseek_context_shared_by_the_conversations(self):
        # 256K alone where the memory left beside the model holds its caches; 4 or 8 conversations share that room
        need, kv = 100.0, maya.ds4_kv_gb(262144)
        self.assertEqual(maya.ds4_recommended_context(0, need + 1.5 * kv, need), 262144)
        self.assertEqual(maya.ds4_recommended_context(0, need + 1.5 * kv, need, parallel=4), 65536)
        self.assertEqual(maya.ds4_recommended_context(0, need + 1.5 * kv, need, parallel=8), 32768)
        self.assertEqual(maya.ds4_recommended_context(32, 0, need, parallel=4), maya.DS4_DEFAULT_CONTEXT)

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

    def test_setup_takes_parallel_for_deepseek(self):
        with patch.object(maya, "choose_family", return_value="deepseek"),                 patch.object(maya, "set_up_ds4", return_value=None) as setup:
            maya.set_up(self.args, {"installer": {"family": "deepseek"}})
        setup.assert_called_once()

    def test_setup_takes_parallel_for_minimax(self):
        with patch.object(maya, "choose_family", return_value="minimax"), \
                patch.object(maya, "set_up_mm3", return_value=None) as setup:
            maya.set_up(self.args, {"installer": {"family": "minimax"}})
        setup.assert_called_once()

    def test_setup_rejects_parallel_before_a_setup_of_a_family_without_it(self):
        with patch.object(maya, "PARALLEL_FAMILIES", ("glm", "deepseek")), \
                patch.object(maya, "choose_family", return_value="minimax"), \
                patch.object(maya, "set_up_mm3", return_value=None) as setup, self.assertRaises(SystemExit):
            maya.set_up(self.args, {"installer": {"family": "minimax"}})
        setup.assert_not_called()

    def test_cli_calibrates_a_minimax_config_with_parallel(self):
        # (--parallel on a start is kept in the config; the speed calibration then declines for MiniMax - no DSpark
        #  with --parallel above 1 - which its own test below pins)
        for action, mocked in (("--calibrate", "calibrate_config"), ("--speed-calibration", "speed_calibration")):
            with self.subTest(action=action):
                self.config("minimax")
                with patch.dict(os.environ, MAYA_SH="1"), \
                        patch.object(sys, "argv", ["maya.py", "--config", str(self.path),
                                                   "--parallel", "4", action, "--no-start"]), \
                        patch.object(maya, "stop_signals"), \
                        patch.object(maya, mocked, return_value=True) as run:
                    self.assertEqual(maya.main(), 0)
                run.assert_called_once()


class MinimaxCalibrationTests(unittest.TestCase):
    """maya-mm3 --parallel above 1 decodes without DSpark: the speed calibration has nothing to measure."""

    def test_declines_with_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            cfg = {"exe": "maya-mm3", "installer": {"family": "minimax"}, "args": ["--pack", "pack", "--dspark", "off",
                                                                                 "--parallel", "4"]}
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with patch.object(maya.CDS, "has_drafter", return_value=True),                     patch.object(maya.CDS, "run") as run, patch.object(maya, "warn") as warn:
                self.assertFalse(maya.needs_speed_calibration(cfg))
                self.assertFalse(maya.speed_calibration(path, "asked for"))
            run.assert_not_called()
            self.assertIn("--parallel", warn.call_args[0][0])
            cfg["args"][cfg["args"].index("--parallel") + 1] = "1"
            with patch.object(maya.CDS, "has_drafter", return_value=True):
                self.assertTrue(maya.needs_speed_calibration(cfg))           # (one conversation: measured as before)


if __name__ == "__main__":
    unittest.main()
