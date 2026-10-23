"""tools/test_slot_keep.py - tools/ds4/slot_keep_test.py's environment, without a GPU: a model config's DeepSeek
settings reach the engine the test starts (a unified-memory APU's DS4_POOL_GB), and the test's own slot settings win
over the config's.

    python -m unittest tools.test_slot_keep -v
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parent / "ds4")]
import slot_keep_test as SK  # noqa: E402


class ConfigEnv(unittest.TestCase):
    def env_of(self, env: dict) -> dict:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "maya-ds4.json"
            path.write_text(json.dumps({"exe": "engine", "args": [], "env": env}), encoding="utf-8")
            return SK.config_env(str(path))

    def test_the_engines_settings_reach_it_under_its_names(self):
        self.assertEqual(self.env_of({"DS4_POOL_GB": "88", "MAYA_DS4_STREAM": "0", "STRATA_DS4_SLOT_CAP": 8}),
                         {"DS4_POOL_GB": "88", "STRATA_DS4_STREAM": "0", "STRATA_DS4_SLOT_CAP": "8"})

    def test_another_engines_settings_stay_out(self):
        self.assertEqual(self.env_of({"STRATA_GLM_RAM_GB": "65", "MAYA_GLM_SLOTS": "0", "CUDA_VISIBLE_DEVICES": "1"}),
                         {})


class EngineEnv(unittest.TestCase):
    def test_the_tests_slot_settings_win_over_a_config(self):
        self.assertEqual(SK.engine_env({"MAYA_DS4_SLOT_KEEP": "0", "MAYA_DS4_SLOT_DIR": "/elsewhere"}, True,
                                       "/tmp/slots", None),
                         {"MAYA_DS4_SLOT_KEEP": "1", "MAYA_DS4_SLOT_DIR": "/tmp/slots"})

    def test_the_cap_is_passed_only_when_the_run_sets_one(self):
        self.assertEqual(SK.engine_env({"DS4_POOL_GB": "88"}, False, "/tmp/slots", 0.05),
                         {"DS4_POOL_GB": "88", "MAYA_DS4_SLOT_KEEP": "0", "MAYA_DS4_SLOT_DIR": "/tmp/slots",
                          "MAYA_DS4_SLOT_GB": "0.05"})
        self.assertNotIn("MAYA_DS4_SLOT_GB", SK.engine_env({}, False, "/tmp/slots", None))


if __name__ == "__main__":
    unittest.main()
