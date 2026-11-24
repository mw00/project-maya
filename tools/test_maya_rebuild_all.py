"""#1066: `./maya.sh --rebuild` compiled only the engine of the model it set up (build/maya), so maya-hy4 and
maya-ds4 stayed the builds from before the update.  It now compiles every engine the PC uses - the configs' engines,
the targets the stamp lists, the engine binaries compiled before - in one build with the same compiler, and for the
GPUs every installed model runs on (the engines share one CMake cache).  cmake is replaced; no GPU needed."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya  # noqa: E402

REAL_CMAKE_STEPS = maya.cmake_steps
V100 ={"index": 0, "name": "Tesla V100", "vram_gb": 32, "arch": "70"}


class RebuildAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.stamp = self.build / "MAYA-BUILD.json"
        self.nvcc = self.root / "nvcc"
        self.nvcc.touch()
        self.llama = self.root / "llama"
        self.built, self.archs, self.warned = [], [], []

        def fake_steps(conf, build_cmd, env, bat, nvcc=None, vcvars=None):
            i = build_cmd.index("--target")
            self.built.append(build_cmd[i + 1:build_cmd.index("-j")])
            self.archs.append(next((c.split("=", 1)[1] for c in conf if c.startswith("-DCMAKE_CUDA_ARCHITECTURES=")
                                    or c.startswith("-DCMAKE_HIP_ARCHITECTURES=")), None))
            return None
        for ctx in (patch.object(maya, "ROOT", self.root), patch.object(maya, "BUILD", self.build),
                    patch.object(maya, "STAMP", self.stamp), patch.object(maya, "EXE", self.build / "maya"),
                    patch.object(maya, "DS4_EXE", self.build / "maya-ds4"),
                    patch.object(maya, "MM3_EXE", self.build / "maya-mm3"),
                    patch.object(maya, "HY4_EXE", self.build / "maya-hy4"),
                    patch.object(maya, "say"), patch.object(maya, "ok"), patch.object(maya, "step"),
                    patch.object(maya, "warn", side_effect=lambda m, *x: self.warned.append(m)),
                    patch.object(maya.S, "source_hash", return_value="new"),
                    patch.object(maya.S, "gpus", return_value=[V100]),
                    patch.object(maya.S, "nvcc_version", return_value=(12, 4)),
                    patch.object(maya, "pick_cmake", return_value="cmake"),
                    patch.object(maya, "cached_generator", return_value=None),
                    patch.object(maya, "venv_tool", return_value=None),
                    patch.object(maya, "toolkit_root_defs", return_value=[]),
                    patch.object(maya, "cuda_lib_dirs", return_value=[]),
                    patch.object(maya, "toolchain_for", side_effect=lambda archs, n, h: (str(self.nvcc), None)),
                    patch.object(maya, "cmake_steps", side_effect=fake_steps)):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.pc = {"backend": "cuda", "gpus": [V100], "archs": [70], "nvcc": str(self.nvcc)}

    def a(self, rebuild):
        return SimpleNamespace(rebuild=rebuild, nvcc=None, host_compiler=None)

    def updated_pc(self):
        """After an update: the three engines compiled before, the stamp of the GLM start after the pull (it lists only
        maya), a Hy4-Flash config beside the GLM one."""
        for t in ("maya", "maya-ds4", "maya-hy4"):
            (self.build / t).touch()
        self.stamp.write_text(json.dumps({"src": "new", "archs": [70], "nvcc": str(self.nvcc), "gpu_ids": [0],
                                          "llama": str(self.llama), "targets": ["maya"]}))
        (self.root / "maya-glm.json").write_text(json.dumps({"exe": str(self.build / "maya"), "gpu": [0]}))
        (self.root / "maya-hy4-flash-maya-s.json").write_text(json.dumps({"exe": str(self.build / "maya-hy4"),
                                                                          "gpu": [0]}))

    def test_rebuild_compiles_every_engine_the_pc_uses(self):
        self.updated_pc()
        maya.build_step(self.a(True), self.pc, self.llama, "maya")
        self.assertEqual(self.built, [["maya", "maya-ds4", "maya-hy4"]])    # (one build, this model's engine first)
        self.assertEqual(json.loads(self.stamp.read_text())["targets"], ["maya", "maya-ds4", "maya-hy4"])

    def test_a_config_engine_not_compiled_yet_is_compiled_too(self):
        self.updated_pc()
        (self.build / "maya-ds4").unlink()
        (self.root / "maya-ds4.json").write_text(json.dumps({"exe": str(self.build / "maya-ds4"), "gpu": [0]}))
        maya.build_step(self.a(True), self.pc, self.llama, "maya-hy4")
        self.assertEqual(self.built, [["maya-hy4", "maya", "maya-ds4"]])

    def test_a_setup_without_rebuild_compiles_only_its_own_engine(self):
        self.updated_pc()
        maya.build_step(self.a(False), self.pc, self.llama, "maya-hy4")    # (not in the stamp: compiled)
        self.assertEqual(self.built, [["maya-hy4"]])
        maya.build_step(self.a(False), self.pc, self.llama, "maya")         # (in the stamp: nothing to do)
        self.assertEqual(self.built, [["maya-hy4"]])

    def test_the_cards_of_every_installed_model_are_compiled_for(self):
        # an RTX 5090 (GPU 0) + an RTX 4070 SUPER (GPU 1): GLM set up on the 5090, Hy4-Flash on both - a rebuild for
        # GLM keeps sm_89 in maya-hy4
        cards = [{"index": 0, "name": "RTX 5090", "vram_gb": 32, "arch": "120"},
                 {"index": 1, "name": "RTX 4070 SUPER", "vram_gb": 12, "arch": "89"}]
        self.updated_pc()
        (self.root / "maya-hy4-flash-maya-s.json").write_text(json.dumps({"exe": str(self.build / "maya-hy4"),
                                                                          "gpu": [0, 1]}))
        pc = {"backend": "cuda", "gpus": cards[:1], "archs": [120], "nvcc": str(self.nvcc)}
        with patch.object(maya.S, "gpus", return_value=cards), patch.object(maya.S, "nvcc_version",
                                                                             return_value=(13, 0)):
            maya.build_step(self.a(True), pc, self.llama, "maya")
        self.assertEqual(self.archs, ["89;120"])
        self.assertEqual(json.loads(self.stamp.read_text())["archs"], [89, 120])

    def test_a_toolkit_that_cannot_build_for_both_keeps_this_models_cards(self):
        # a V100 config beside a 5090 set up now, with only CUDA 13 (no Volta): the 5090's build, said
        cards = [{"index": 0, "name": "RTX 5090", "vram_gb": 32, "arch": "120"}, dict(V100, index=1)]
        self.updated_pc()
        (self.root / "maya-hy4-flash-maya-s.json").write_text(json.dumps({"exe": str(self.build / "maya-hy4"),
                                                                          "gpu": [1]}))
        pc = {"backend": "cuda", "gpus": cards[:1], "archs": [120], "nvcc": str(self.nvcc)}
        with patch.object(maya.S, "gpus", return_value=cards), patch.object(maya.S, "nvcc_version",
                                                                             return_value=(13, 0)):
            maya.build_step(self.a(False), pc, self.llama, "maya")
        self.assertEqual(self.archs, ["120"])
        self.assertTrue(any("sm_70" in w for w in self.warned))

    def test_a_start_after_an_update_compiles_for_its_own_cards(self):
        # the last setup was GLM's on the 5090 alone (sm_120); Hy4-Flash runs on both: its update build takes sm_89 too
        cards = [{"index": 0, "name": "RTX 5090", "vram_gb": 32, "arch": "120"},
                 {"index": 1, "name": "RTX 4070 SUPER", "vram_gb": 12, "arch": "89"}]
        (self.llama / "ggml").mkdir(parents=True)
        (self.llama / "ggml" / "CMakeLists.txt").touch()
        self.stamp.write_text(json.dumps({"src": "old", "archs": [120], "nvcc": str(self.nvcc), "gpu_ids": [0],
                                          "llama": str(self.llama), "targets": ["maya", "maya-hy4"]}))
        with patch.object(maya.S, "gpus", return_value=cards), patch.object(maya.S, "nvcc_version",
                                                                             return_value=(13, 0)):
            maya.refresh_engine({"exe": str(self.build / "maya-hy4"), "gpu": [0, 1]})
        self.assertEqual(self.built, [["maya-hy4"]])
        self.assertEqual(self.archs, ["89;120"])

    def test_hip_rebuild_compiles_every_engine(self):
        self.updated_pc()
        self.stamp.write_text(json.dumps({"backend": "hip", "src": "new", "archs": ["gfx1151"], "rocm": "/opt/rocm",
                                          "targets": ["maya-ds4"]}))
        pc = {"backend": "hip", "gpus": [{"index": 0}], "archs": ["gfx1151"], "rocm": "/opt/rocm"}
        with patch.object(maya, "hip_toolchain", return_value=(Path("/opt/rocm"), Path("/opt/rocm/llvm/bin/clang++"),
                                                               Path("/opt/rocm/bin"), {})), \
                patch.object(maya, "WIN", False):
            maya.build_step(self.a(True), pc, self.llama, "maya-hy4")
        self.assertEqual(self.built, [["maya-hy4", "maya", "maya-ds4"]])
        self.assertEqual(json.loads(self.stamp.read_text())["targets"], ["maya", "maya-ds4", "maya-hy4"])

    # the HIP build: one binary per engine for every installed model's AMD GPUs, as the CUDA build (an RX 7900 XTX
    # (HIP GPU 0) + an RX 9070 XT (GPU 1): GLM on the 7900, Hy4-Flash on both)
    AMD = [{"index": 0, "name": "Radeon RX 7900 XTX", "vram_gb": 24, "arch": "gfx1100"},
           {"index": 1, "name": "Radeon RX 9070 XT", "vram_gb": 16, "arch": "gfx1201"}]

    def hip(self, bitcode=None):
        env = {"HIP_DEVICE_LIB_PATH": str(bitcode)} if bitcode else {}
        return (patch.object(maya, "hip_toolchain", return_value=(self.root / "rocm", self.root / "rocm/llvm/bin/clang++",
                                                                  self.root / "rocm/bin", env)),
                patch.object(maya, "WIN", False), patch.object(maya.S, "amd_gpus", return_value=self.AMD))

    def hip_configs(self):
        self.updated_pc()
        for p in self.root.glob("maya-*.json"):
            p.unlink()
        (self.root / "maya-glm.json").write_text(json.dumps({"exe": str(self.build / "maya"), "gpu": [0]}))
        (self.root / "maya-hy4-flash-maya-s.json").write_text(json.dumps({"exe": str(self.build / "maya-hy4"),
                                                                          "gpu": [0, 1]}))
        return {"backend": "hip", "gpus": self.AMD[:1], "archs": ["gfx1100"], "rocm": str(self.root / "rocm"),
                "amd_found": self.AMD}

    def test_hip_the_gpus_of_every_installed_model_are_compiled_for(self):
        pc = self.hip_configs()
        a, b, c = self.hip()
        with a, b, c:
            maya.build_step(self.a(True), pc, self.llama, "maya")
        self.assertEqual(self.archs, ["gfx1100;gfx1201"])
        self.assertEqual(json.loads(self.stamp.read_text())["archs"], ["gfx1100", "gfx1201"])
        self.assertEqual(pc["archs"], ["gfx1100"])                         # (the setup's own choice is kept)

    def test_hip_a_rocm_without_the_other_gpu_keeps_this_models(self):
        pc = self.hip_configs()
        bc = self.root / "bitcode"
        bc.mkdir()
        (bc / "oclc_isa_version_1100.bc").touch()                          # (no 1201 in this ROCm's device libraries)
        a, b, c = self.hip(bc)
        with a, b, c:
            maya.build_step(self.a(True), pc, self.llama, "maya")
        self.assertEqual(self.archs, ["gfx1100"])
        self.assertTrue(any("gfx1201" in w for w in self.warned))
        (bc / "oclc_isa_version_1201.bc").touch()
        self.archs.clear()
        with a, b, c:
            maya.build_step(self.a(True), pc, self.llama, "maya")
        self.assertEqual(self.archs, ["gfx1100;gfx1201"])

    def test_hip_a_build_for_both_is_not_narrowed_by_a_setup_on_one(self):
        pc = self.hip_configs()
        self.stamp.write_text(json.dumps({"backend": "hip", "src": "new", "archs": ["gfx1100", "gfx1201"],
                                          "rocm": pc["rocm"], "targets": ["maya", "maya-hy4"]}))
        a, b, c = self.hip()
        with a, b, c:
            maya.build_step(self.a(False), pc, self.llama, "maya")
        self.assertEqual(self.built, [])
        self.stamp.write_text(json.dumps({"backend": "hip", "src": "new", "archs": ["gfx1100"],   # (GLM's alone before)
                                          "rocm": pc["rocm"], "targets": ["maya", "maya-hy4"]}))
        with a, b, c:
            maya.build_step(self.a(False), pc, self.llama, "maya")
        self.assertEqual(self.archs, ["gfx1100;gfx1201"])

    def test_hip_a_start_after_an_update_compiles_for_every_models_gpus(self):
        self.hip_configs()
        (self.llama / "ggml").mkdir(parents=True)
        (self.llama / "ggml" / "CMakeLists.txt").touch()
        self.stamp.write_text(json.dumps({"backend": "hip", "src": "old", "archs": ["gfx1100"], "rocm": "/opt/rocm",
                                          "llama": str(self.llama), "targets": ["maya", "maya-hy4"]}))
        a, b, c = self.hip()
        with a, b, c, patch.object(maya, "hip_clang", return_value=Path("/opt/rocm/llvm/bin/clang++")):
            maya.refresh_engine({"exe": str(self.build / "maya-hy4"), "gpu": [0, 1], "backend": "hip"})
        self.assertEqual(self.built, [["maya-hy4"]])
        self.assertEqual(self.archs, ["gfx1100;gfx1201"])

    def test_windows_build_script_builds_every_target(self):
        # the .bat cmake_steps writes on Windows carries every target of the one build
        seen = {}

        def fake_run(cmd, env=None, check=True, echo=True, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0)
        with patch.object(maya, "cmake_steps", REAL_CMAKE_STEPS), patch.object(maya, "WIN", True), \
                patch.object(maya.S, "find_vcvars", return_value=Path(r"C:\VS\vcvars64.bat")), \
                patch.object(maya.S, "full_build_output", return_value=True), \
                patch.object(maya, "run", side_effect=fake_run), patch.object(maya, "mem_gb", return_value=(64, 60)):
            self.updated_pc()
            maya.build_step(self.a(True), self.pc, self.llama, "maya")
        bat = (self.root / "build-maya.bat").read_text(encoding="utf-8")
        self.assertIn("--target maya maya-ds4 maya-hy4 -j", bat)


if __name__ == "__main__":
    unittest.main()
