"""#664: a GLM setup and a DeepSeek setup share one build folder and one build stamp.  After a git pull the first start
compiles only its own engine; the other model's next start must compile its own engine too (CUDA path; the HIP path
is covered in test_maya_hip.py).  The real compile_engine writes the stamp here, cmake is replaced.  No GPU needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


class RefreshTargetsTests(unittest.TestCase):
    def test_second_model_compiles_its_own_engine_after_an_update(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            build = root / "build"
            build.mkdir()
            nvcc = root / "nvcc"
            nvcc.touch()
            (root / "llama/ggml").mkdir(parents=True)
            (root / "llama/ggml/CMakeLists.txt").touch()
            stamp = build / "MAYA-BUILD.json"
            stamp.write_text(json.dumps({"src": "old", "archs": ["70"], "nvcc": str(nvcc), "gpu_ids": [0],
                                         "llama": str(root / "llama"), "targets": ["maya", "maya-ds4"]}))
            built = []

            def fake_steps(conf, build_cmd, env, bat, nvcc=None):
                built.append(build_cmd[build_cmd.index("--target") + 1])
                return None

            glm = {"exe": str(build / "maya"), "gpu": [0]}
            ds4 = {"exe": str(build / "maya-ds4"), "gpu": [0]}
            source = ["new"]
            with patch.object(maya, "EXE", build / "maya"), patch.object(maya, "DS4_EXE", build / "maya-ds4"), \
                    patch.object(maya, "MM3_EXE", build / "maya-mm3"), patch.object(maya, "BUILD", build), \
                    patch.object(maya, "STAMP", stamp), patch.object(maya, "say"), patch.object(maya, "ok"), \
                    patch.object(maya.S, "source_hash", side_effect=lambda *_: source[0]), \
                    patch.object(maya, "pick_cmake", return_value="cmake"), \
                    patch.object(maya, "cached_generator", return_value=None), \
                    patch.object(maya, "venv_tool", return_value=None), \
                    patch.object(maya, "toolkit_root_defs", return_value=[]), \
                    patch.object(maya, "cuda_lib_dirs", return_value=[]), \
                    patch.object(maya, "cmake_steps", side_effect=fake_steps):
                maya.refresh_engine(glm)              # the first model starts after the pull
                self.assertEqual(built, ["maya"])
                self.assertEqual(json.loads(stamp.read_text())["targets"], ["maya"])
                maya.refresh_engine(glm)              # starting it again: nothing more to build
                self.assertEqual(built, ["maya"])
                maya.refresh_engine(ds4)              # the other model: its own engine is still the old build
                self.assertEqual(built, ["maya", "maya-ds4"])
                self.assertEqual(json.loads(stamp.read_text())["targets"], ["maya", "maya-ds4"])
                maya.refresh_engine(ds4)
                maya.refresh_engine(glm)
                self.assertEqual(built, ["maya", "maya-ds4"])


class FailedUpdateBuildTests(unittest.TestCase):
    """#805 on NVIDIA: an update whose engine rebuild fails stops with the reason and the fix - the engine compiled
    before is from the older source and never starts under this version's server and config."""
    def test_failed_update_build_never_starts_the_engine_from_before(self):
        with tempfile.TemporaryDirectory() as d:
            build = Path(d) / "build"
            build.mkdir()
            with patch.object(maya, "BUILD", build), patch.object(maya, "STAMP", build / "MAYA-BUILD.json"), \
                    patch.object(maya, "say"), patch.object(maya, "pick_cmake", return_value="cmake"), \
                    patch.object(maya, "cached_generator", return_value=None), \
                    patch.object(maya, "venv_tool", return_value=None), \
                    patch.object(maya, "toolkit_root_defs", return_value=[]), \
                    patch.object(maya, "cmake_steps", return_value="compiling"), \
                    patch.object(maya, "warn", side_effect=AssertionError("no warning-and-go-on")), \
                    patch.object(maya, "fail", side_effect=SystemExit(1)) as stop:
                with self.assertRaises(SystemExit):
                    maya.compile_engine(["70"], [0], "nvcc", None, Path(d) / "llama", "source-sha", soft=True)
            self.assertIn("not started", stop.call_args.args[0])
            self.assertIn("start Maya again", stop.call_args.args[1])

    def test_refresh_without_the_cuda_compiler_stops(self):
        with tempfile.TemporaryDirectory() as d:
            build = Path(d) / "build"
            build.mkdir()
            stamp = build / "MAYA-BUILD.json"
            stamp.write_text(json.dumps({"src": "old", "archs": ["70"], "nvcc": str(Path(d) / "gone" / "nvcc"),
                                         "gpu_ids": [0], "llama": str(Path(d) / "llama"), "targets": ["maya"]}))
            with patch.object(maya, "EXE", build / "maya"), patch.object(maya, "DS4_EXE", build / "maya-ds4"), \
                    patch.object(maya, "MM3_EXE", build / "maya-mm3"), patch.object(maya, "STAMP", stamp), \
                    patch.object(maya, "say"), patch.object(maya.S, "source_hash", return_value="new"), \
                    patch.object(maya, "compile_engine", side_effect=AssertionError("nothing to compile with")), \
                    patch.object(maya, "fail", side_effect=SystemExit(1)) as stop:
                with self.assertRaises(SystemExit):
                    maya.refresh_engine({"exe": str(build / "maya"), "gpu": [0]})
            self.assertIn("not started", stop.call_args.args[0])
            self.assertIn("--setup --rebuild", stop.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
