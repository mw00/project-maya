"""An unattended start (no terminal on stdin, systemd, --yes) never runs a calibration by itself: it loads the model two
more times, and a service killed at its start timeout would repeat that forever (#690).  An interactive start measures
as before.  No GPU needed."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


class FakeStdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


class UnattendedStartTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "model.json"
        self.said = []
        self.mocks = {}
        for name, effect in (("ok", None), ("select_build_backend", None), ("refresh_engine", None),
                             ("ds4_seed_profile", None), ("add_profile", None), ("rename_programs", None),
                             ("speed_calibration", False), ("prefill_calibration", False)):
            mocked = patch.object(maya, name, return_value=effect)
            self.mocks[name] = mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(maya, "say", side_effect=lambda m="": self.said.append(m))
        mocked.start()
        self.addCleanup(mocked.stop)
        mocked = patch.object(maya.CDS, "has_drafter", return_value=True)
        mocked.start()
        self.addCleanup(mocked.stop)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("INVOCATION_ID", None)
        os.environ.pop("NOTIFY_SOCKET", None)
        self.args = SimpleNamespace(parallel=None, port=None, host=None, api_key=None, gpus=None, gpu=None)

    def config(self, family):
        exe = self.root / {"glm": "maya", "deepseek": "maya-ds4", "minimax": "maya-mm3"}[family]
        exe.touch()
        pack = self.root / "pack"
        pack.mkdir(exist_ok=True)
        tok = self.root / "tokenizer"
        tok.mkdir(exist_ok=True)
        (tok / "vocab.json").write_text("{}", encoding="utf-8")
        args = ["--glm-pack" if family == "glm" else "--pack", str(pack)]
        if family == "glm":
            args += ["--dflash", str(self.root / "drafter")]       # (DFlash2 chosen, never measured)
        cfg = {"exe": str(exe), "tokenizer": str(tok), "gpu": [0], "installer": {"family": family}, "args": args}
        self.path.write_text(json.dumps(cfg), encoding="utf-8")
        return cfg

    def start(self, tty):
        with patch.object(maya.sys, "stdin", FakeStdin(tty)):
            maya.server_command(self.path, self.args)

    def skipped_line(self):
        return [m for m in self.said if "Calibration skipped" in m]

    def test_every_family_needs_a_calibration_here(self):
        for family in ("glm", "deepseek", "minimax"):
            self.assertTrue(maya.needs_speed_calibration(self.config(family)), family)
        self.assertTrue(maya.needs_prefill_calibration(self.config("minimax")))

    def test_no_terminal_skips_and_says_the_command(self):
        for family in ("glm", "deepseek", "minimax"):
            with self.subTest(family=family):
                self.said.clear()
                cfg = self.config(family)
                before = self.path.read_bytes()
                self.start(tty=False)
                self.mocks["speed_calibration"].assert_not_called()
                self.mocks["prefill_calibration"].assert_not_called()
                line = self.skipped_line()
                self.assertEqual(len(line), 1)
                self.assertIn("--speed-calibration --no-start", line[0])
                self.assertIn(str(self.path), line[0])
                self.assertEqual("--calibrate --no-start" in line[0], family == "minimax")
                self.assertEqual(self.path.read_bytes(), before)       # (nothing recorded: the next interactive start measures)
                self.assertEqual(maya.read_json(self.path)["args"], cfg["args"])

    def test_systemd_skips_even_with_a_terminal(self):
        for var in ("INVOCATION_ID", "NOTIFY_SOCKET"):
            with self.subTest(var=var):
                self.said.clear()
                self.config("deepseek")
                with patch.dict(os.environ, {var: "x"}):
                    self.start(tty=True)
                self.mocks["speed_calibration"].assert_not_called()
                self.assertEqual(len(self.skipped_line()), 1)

    def test_yes_skips(self):
        self.config("minimax")
        self.args.yes = True
        self.start(tty=True)
        self.mocks["speed_calibration"].assert_not_called()
        self.mocks["prefill_calibration"].assert_not_called()
        self.assertEqual(len(self.skipped_line()), 1)

    def test_interactive_start_measures_as_before(self):
        for family in ("glm", "deepseek", "minimax"):
            with self.subTest(family=family):
                self.said.clear()
                for m in (self.mocks["speed_calibration"], self.mocks["prefill_calibration"]):
                    m.reset_mock()
                self.config(family)
                self.start(tty=True)
                self.mocks["speed_calibration"].assert_called_once()
                self.assertEqual(self.mocks["prefill_calibration"].call_count, 1 if family == "minimax" else 0)
                self.assertEqual(self.skipped_line(), [])

    def test_unattended_start_with_a_saved_calibration_says_nothing(self):
        cfg = self.config("deepseek")
        cfg["installer"]["speed_calibration"] = {"kept": "dspark"}
        self.path.write_text(json.dumps(cfg), encoding="utf-8")
        self.start(tty=False)
        self.mocks["speed_calibration"].assert_not_called()
        self.assertEqual(self.skipped_line(), [])

    def test_stdin_closed_counts_as_no_terminal(self):
        broken = MagicMock()
        broken.isatty.side_effect = ValueError("I/O operation on closed file")
        with patch.object(maya.sys, "stdin", broken):
            self.assertEqual(maya.unattended_start(self.args), "no terminal")
        with patch.object(maya.sys, "stdin", FakeStdin(True)):
            self.assertIsNone(maya.unattended_start(self.args))


if __name__ == "__main__":
    unittest.main()
