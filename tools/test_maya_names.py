"""tools/test_maya_names.py - the settings' names: MAYA_* on disk and on screen, the older STRATA_* still read.

    python -m unittest tools.test_maya_names -v
"""
from pathlib import Path
import json
import os
import re
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import maya_names as N  # noqa: E402


class Names(unittest.TestCase):
    def test_environment(self):
        env = {"MAYA_GLM_CPU_LANE": "8", "STRATA_GLM_SPLIT": "0", "STRATA_GLM_PREFILL_MB": "1", "MAYA_GLM_PREFILL_MB": "2",
               "PATH": "/bin"}
        N.alias_env(env)
        self.assertEqual(env["STRATA_GLM_CPU_LANE"], "8")       # the name the code reads
        self.assertEqual(env["STRATA_GLM_SPLIT"], "0")          # an older name alone still counts
        self.assertEqual(env["STRATA_GLM_PREFILL_MB"], "2")     # both: the MAYA_ one wins
        self.assertEqual(env["PATH"], "/bin")

    def test_config_round_trip(self):
        disk = {"exe": "x", "env": {"MAYA_GLM_CPU_LANE": "8", "STRATA_GLM_SPLIT": "0", "HIP_VISIBLE_DEVICES": "0"}}
        cfg = N.env_in(json.loads(json.dumps(disk)))
        self.assertEqual(cfg["env"], {"STRATA_GLM_CPU_LANE": "8", "STRATA_GLM_SPLIT": "0", "HIP_VISIBLE_DEVICES": "0"})
        out = N.env_out(cfg)
        self.assertEqual(out["env"], {"MAYA_GLM_CPU_LANE": "8", "MAYA_GLM_SPLIT": "0", "HIP_VISIBLE_DEVICES": "0"})
        self.assertEqual(cfg["env"]["STRATA_GLM_SPLIT"], "0")   # (the copy written; the config in memory unchanged)
        self.assertEqual(N.env_out({"exe": "x"}), {"exe": "x"})

    def test_setup_reads_and_writes_configs_so(self):
        p = Path(tempfile.mkdtemp()) / "maya-x.json"
        p.write_text(json.dumps({"env": {"MAYA_GLM_RESERVE_MB": "3072"}}))
        self.assertEqual(maya.read_json(p)["env"], {"STRATA_GLM_RESERVE_MB": "3072"})
        self.assertEqual(N.shown("STRATA_GLM_RESERVE_MB"), "MAYA_GLM_RESERVE_MB")

    def test_both_names_and_lookup(self):
        self.assertEqual(N.both_names("MAYA_DS4_READ_ORDER"), ("MAYA_DS4_READ_ORDER", "STRATA_DS4_READ_ORDER"))
        self.assertEqual(N.both_names("STRATA_DS4_READ_ORDER"), ("MAYA_DS4_READ_ORDER", "STRATA_DS4_READ_ORDER"))
        self.assertEqual(N.both_names("PATH"), ("PATH", "PATH"))
        env = {"STRATA_A": "1", "MAYA_B": "2", "STRATA_C": "old", "MAYA_C": "new"}
        self.assertEqual([N.lookup(env, n) for n in ("MAYA_A", "STRATA_A", "MAYA_B", "STRATA_B", "MAYA_C")],
                         ["1", "1", "2", "2", "new"])
        self.assertIsNone(N.lookup(env, "MAYA_NONE"))
        self.assertEqual(N.lookup(env, "MAYA_NONE", "d"), "d")

    def test_export_env_gives_both_names(self):
        out = {"STRATA_X": "stale", "MAYA_Y": "shell"}
        N.export_env(out, {"STRATA_X": "1", "MAYA_Y": "2", "HIP_VISIBLE_DEVICES": "0", "N": 3})
        self.assertEqual(out, {"STRATA_X": "1", "MAYA_X": "1", "STRATA_Y": "2", "MAYA_Y": "2",
                               "HIP_VISIBLE_DEVICES": "0", "N": "3"})
        both = {}                                                  # (a config naming one setting both ways: MAYA_ wins)
        N.export_env(both, {"MAYA_Z": "new", "STRATA_Z": "old"})
        self.assertEqual(both, {"MAYA_Z": "new", "STRATA_Z": "new"})


class EngineSeesMayaOnlySettings(unittest.TestCase):
    """Issue #686: a setting the engine reads as getenv("MAYA_...") alone must reach it when it is in a config's "env"."""

    @staticmethod
    def engine_maya_names():
        """Every MAYA_* name the engines' sources read (getenv / env_count / env_int), so a new one is covered too."""
        src = Path(__file__).resolve().parents[1]
        names = set()
        for top in ("src", "include"):
            for f in (src / top).rglob("*"):
                if f.suffix in (".cpp", ".cu", ".cuh", ".h", ".hpp", ".mm"):
                    names.update(re.findall(r'(?:getenv|env_count|env_int|env)\("(MAYA_[A-Z0-9_]+)"',
                                            f.read_text(encoding="utf-8", errors="replace")))
        return sorted(names)

    def test_config_env_reaches_the_engine_under_the_maya_name(self):
        from serve.server import child_env
        names = self.engine_maya_names()
        self.assertIn("MAYA_DS4_READ_ORDER", names)               # (the audit finds the reporter's setting)
        self.assertGreater(len(names), 30)
        for backend in ("cuda", "hip", "metal"):
            with self.subTest(backend=backend):
                disk = {"exe": "x", "backend": backend, "gpu": [0], "env": {n: "7" for n in names}}
                cfg = N.env_in(json.loads(json.dumps(disk)))      # as read from a config on disk
                self.assertTrue(all(k.startswith("STRATA_") for k in cfg["env"]))
                env = child_env(cfg)
                for n in names:
                    self.assertEqual(env.get(n), "7", n)           # what getenv("MAYA_...") reads
                    self.assertEqual(env.get(N.read_as(n)), "7", n)   # and the older name

    def test_a_started_engine_reads_the_setting(self):
        """The real thing: a child started with the server's environment prints what getenv would give it."""
        import subprocess
        import sys as _sys
        from serve.server import child_env
        p = Path(tempfile.mkdtemp()) / "maya-x.json"
        p.write_text(json.dumps({"exe": "x", "env": {"MAYA_DS4_READ_ORDER": "arrival", "MAYA_EXPERT_STAGERS": "3"}}))
        cfg = maya.read_json(p)
        out = subprocess.run([_sys.executable, "-c", "import os;print(os.environ.get('MAYA_DS4_READ_ORDER'),"
                              "os.environ.get('MAYA_EXPERT_STAGERS'),os.environ.get('STRATA_DS4_READ_ORDER'))"],
                             env=child_env(cfg), capture_output=True, text=True, timeout=60).stdout.split()
        self.assertEqual(out, ["arrival", "3", "arrival"])

    def test_the_server_applies_its_own_settings_under_both_names(self):
        from unittest import mock
        from serve import server as S
        with mock.patch.dict(os.environ):
            S.apply_server_env(N.env_in({"env": {"MAYA_QUEUE_PING_S": "0.5", "MAYA_UPDATE_CHECK": "0"}}))
            self.assertEqual(S.queue_ping_seconds(), 0.5)
            self.assertEqual(os.environ["MAYA_UPDATE_CHECK"], "0")


class EngineReadsTheConfigName(unittest.TestCase):
    def test_the_in_place_trim_settings_are_read_under_the_config_name(self):
        # #728: env_in leaves a config's MAYA_DS4_INPLACE_* settings under STRATA_; the engine must read that name too
        src = (Path(__file__).resolve().parents[1] / "src" / "ds4" / "ds4_model.cu").read_text(encoding="utf-8")
        for name in ("MAYA_DS4_INPLACE_TRIM_GB", "MAYA_DS4_INPLACE_TRIM_MS"):
            self.assertIn(f'"{N.read_as(name)}"', src)


if __name__ == "__main__":
    unittest.main()
