"""MAYA_PAR_MAX: engine_args caps the config's --parallel before the engine starts; unset, 0 or not a number leaves
the arguments as they are.

    python -m unittest serve.test_par_max
"""
from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import engine_args  # noqa: E402

ARGS = ["--pack", "/p", "--parallel", "3"]


class ParMax(unittest.TestCase):
    def args(self, env: dict | None = None, os_env: dict | None = None) -> list[str]:
        with unittest.mock.patch.dict(os.environ):   # (restored after)
            os.environ.pop("MAYA_PAR_MAX", None)
            os.environ.update(os_env or {})
            return engine_args({"args": list(ARGS), "env": env or {}})

    def test_caps_from_the_config(self):
        self.assertEqual(self.args({"MAYA_PAR_MAX": "2"}), ["--pack", "/p", "--parallel", "2"])

    def test_caps_from_the_environment(self):
        self.assertEqual(self.args(os_env={"MAYA_PAR_MAX": "1"}), ["--pack", "/p", "--parallel", "1"])

    def test_unchanged_when_not_lower_unset_zero_or_bad(self):
        for env in ({}, {"MAYA_PAR_MAX": "4"}, {"MAYA_PAR_MAX": "0"}, {"MAYA_PAR_MAX": "two"}):
            self.assertEqual(self.args(env), ARGS, env)


if __name__ == "__main__":
    unittest.main()
