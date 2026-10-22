"""tools/test_bench.py - --bench's choice of engine binary on HIP (maya.py). No GPU, no network, no engine process.

    python -m unittest tools.test_bench -v
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402


class ExeForTarget(unittest.TestCase):
    """exe_for_target: the one place EXE vs DS4_EXE gets picked by name - bench_exe and every build step
    (compile_engine, compile_engine_hip, build_step) all go through it."""

    def setUp(self):
        for ctx in (patch.object(maya, "EXE", Path("/build-hip/maya")),
                    patch.object(maya, "DS4_EXE", Path("/build-hip/maya-ds4"))):
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_maya_target_is_the_glm_binary(self):
        self.assertEqual(maya.exe_for_target("maya"), maya.EXE)

    def test_maya_ds4_target_is_the_ds4_binary(self):
        self.assertEqual(maya.exe_for_target("maya-ds4"), maya.DS4_EXE)

    def test_unknown_target_raises_instead_of_guessing(self):
        # the old bug's failure mode was silently picking the wrong binary; an unknown target should fail loudly.
        with self.assertRaises(KeyError):
            maya.exe_for_target("maya-glm-typo")

    def test_reads_exe_and_ds4_exe_live_not_at_import_time(self):
        # select_build_backend() reassigns EXE/DS4_EXE once the backend is known; a dict built once at import time
        # would freeze the pre-backend-selection values and silently go stale.
        with patch.object(maya, "EXE", Path("/build/maya")), patch.object(maya, "DS4_EXE", Path("/build/maya-ds4")):
            self.assertEqual(maya.exe_for_target("maya"), Path("/build/maya"))
        with patch.object(maya, "EXE", Path("/build-hip/maya")), patch.object(maya, "DS4_EXE",
                                                                                Path("/build-hip/maya-ds4")):
            self.assertEqual(maya.exe_for_target("maya"), Path("/build-hip/maya"))


class BenchExeSelection(unittest.TestCase):
    """#350: --bench used to hardcode the GLM binary for every HIP config, crashing a DeepSeek setup with
    FileNotFoundError (build-hip/maya was never built there - build-hip/maya-ds4 was)."""

    def setUp(self):
        for ctx in (patch.object(maya, "EXE", Path("/build-hip/maya")),
                    patch.object(maya, "DS4_EXE", Path("/build-hip/maya-ds4"))):
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_deepseek_config_gets_the_ds4_binary(self):
        cfg = {"exe": "/build-hip/maya-ds4", "backend": "hip",
               "args": ["--pack", "/m/maya-ds4", "--max-context", "65536"]}
        self.assertEqual(maya.bench_exe(cfg), maya.DS4_EXE)

    def test_glm_config_gets_the_glm_binary(self):
        cfg = {"exe": "/build-hip/maya", "backend": "hip", "args": ["--glm-pack", "/m/pack"]}
        self.assertEqual(maya.bench_exe(cfg), maya.EXE)

    def test_old_pre_1_1_16_ds4_name_still_recognised(self):
        cfg = {"exe": "/build-hip/strata-ds4", "backend": "hip", "args": ["--pack", "/m/maya-ds4"]}
        self.assertEqual(maya.bench_exe(cfg), maya.DS4_EXE)

    def test_agrees_with_config_family_by_construction(self):
        # bench_exe must never answer differently than config_family - that drift is exactly what caused #350.
        ds4 = {"exe": "/build-hip/maya-ds4", "args": ["--pack", "/m/maya-ds4"]}
        glm = {"exe": "/build-hip/maya", "args": ["--glm-pack", "/m/pack"]}
        for cfg in (ds4, glm):
            expect = maya.DS4_EXE if maya.config_family(cfg) == "deepseek" else maya.EXE
            self.assertEqual(maya.bench_exe(cfg), expect)


if __name__ == "__main__":
    unittest.main()
