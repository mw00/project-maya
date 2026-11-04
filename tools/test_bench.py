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

    def test_minimax_config_gets_the_mm3_binary(self):
        with patch.object(maya, "config_family", return_value="minimax"), \
                patch.object(maya, "MM3_EXE", Path("/build-hip/maya-mm3")):
            self.assertEqual(maya.bench_exe({"exe": "/build-hip/maya-mm3", "args": []}), Path("/build-hip/maya-mm3"))


class BenchRequests(unittest.TestCase):
    """#549: each family's own chat format and short answers out of the mean; #420 / #509: a --parallel engine's GEN
    carries rid=0 and its DONE line its id."""

    def test_chat_formats(self):
        self.assertEqual(maya.bench_chat("glm", Path("/none"))("hi"), "[gMASK]<sop><|user|>\nhi<|assistant|>\n</think>")
        self.assertEqual(maya.bench_chat("deepseek", Path("/none"))("hi"),
                         "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>")

    def test_minimax_uses_its_own_template(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2 not installed")
        s = maya.bench_chat("minimax", Path("/none"))("hi")
        self.assertNotIn("[gMASK]", s)
        self.assertIn("hi", s)

    def test_gen_line(self):
        self.assertEqual(maya.bench_gen_line(256, [1, 2], False), "GEN 256 temperature=0 1,2\n")
        self.assertEqual(maya.bench_gen_line(256, [1, 2], True), "GEN 256 temperature=0 rid=0 1,2\n")

    def test_reply(self):
        self.assertEqual(maya.bench_reply("DONE 256 9 1.0 2.0 length\n", False)[:2], ["DONE", "256"])
        self.assertEqual(maya.bench_reply("DONE 0 256 9 1.0 2.0 length\n", True)[:2], ["DONE", "256"])
        self.assertIsNone(maya.bench_reply("DONE 3 256 9 1.0 2.0 length\n", True))   # (another request's)
        self.assertEqual(maya.bench_reply("ERR this engine runs several conversations: GEN needs rid=<id>", True)[0],
                         "ERR")
        self.assertIsNone(maya.bench_reply("PP 64 128 1.0 2.0 0", False))

    def test_short_answer_not_in_the_mean(self):
        r = [("decode", "a", 20.0, 256), ("decode", "b", 40.0, 11), ("decode", "c", 22.0, 256), ("prefill", "x", 900.0)]
        self.assertEqual(maya.bench_decode_mean(r), (21.0, 2))
        self.assertEqual(maya.bench_decode_mean([("decode", "b", 40.0, 11)]), (40.0, 1))   # (all short: all of them)


if __name__ == "__main__":
    unittest.main()
