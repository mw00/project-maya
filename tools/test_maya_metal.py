"""Maya's Metal (Apple Silicon) setup regressions; no GPU, Mac, network or model downloads required."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


class MetalSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        sysctl = {"machdep.cpu.brand_string": "Apple M4 Max", "hw.memsize": str(64 * 2**30)}
        out = lambda cmd: (sysctl[cmd[-1]] if cmd[0] == "sysctl" else "/usr/bin/clang++\n" if cmd[0] == "xcrun"
                           else "/Library/Developer/CommandLineTools\n")      # noqa: E731
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "PROFILE", self.root / "expert-profile-glm.bin"),
            patch.object(maya, "WIN", False),
            patch.object(maya, "apple_silicon", return_value=True),
            patch.object(maya, "mem_gb", return_value=(64.0, 20.0)),
            patch.object(maya, "pick_cmake", return_value="/usr/bin/cmake"),
            patch.object(maya, "tool_version", return_value=(3, 30)),
            patch.object(maya.S, "out", side_effect=out),
            patch.object(maya.S, "gpus", side_effect=AssertionError("Metal must not query NVIDIA")),
            patch.object(maya.S, "amd_gpus", side_effect=AssertionError("Metal must not query AMD")),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(backend="metal", gpu=None, gpus=None, port=8099, host=None, api_key=None, env=[],
                                 gguf_dir=None, parallel=None, vram_reserve_mib=None, no_start=True, yes=True)

    def test_build_folder_and_program(self):
        maya.select_build_backend("metal")
        self.assertEqual((maya.BUILD, maya.EXE), (self.root / "build-metal", self.root / "build-metal/maya-metal"))
        self.assertEqual(maya.STAMP, self.root / "build-metal/MAYA-BUILD.json")
        maya.select_build_backend("cuda")                       # (the other backends keep theirs)
        self.assertEqual(maya.EXE, self.root / "build/maya")

    def test_check_describes_the_apple_gpu(self):
        pc = maya.check_pc(self.a)
        self.assertEqual(pc["backend"], "metal")
        self.assertEqual(pc["gpus"][0]["index"], 0)
        self.assertEqual(pc["gpus"][0]["name"], "Apple M4 Max")
        self.assertAlmostEqual(pc["gpus"][0]["vram_gb"], 0.85 * 64)
        self.assertEqual(maya.EXE.name, "maya-metal")

    def test_check_needs_apple_silicon_and_a_compiler(self):
        with patch.object(maya, "apple_silicon", return_value=False), self.assertRaises(SystemExit):
            maya.check_pc(self.a)
        with patch.object(maya.S, "out", return_value=""), self.assertRaises(SystemExit):   # no Xcode command line tools
            maya.check_pc(self.a)

    def test_deepseek_has_no_metal_engine(self):
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a, "deepseek", gguf=True)

    def test_exl3_quants_are_not_offered(self):
        self.a.model = "EXL3-3.05bpw"
        with self.assertRaises(SystemExit):
            maya.choose_model(self.a, self.root, {})
        self.a.model = None
        with patch.object(maya, "say"):
            kind, quant = maya.choose_model(self.a, self.root, {})
        self.assertFalse(maya.is_exl3(quant))

    def test_first_run_on_a_mac_picks_metal(self):
        with patch.object(maya, "configs", return_value=[]), patch.dict(os.environ, MAYA_SH="1"), \
             patch.object(sys, "argv", ["maya.py", "--check"]), \
             patch.object(maya, "check_pc", return_value={"gpus": [{"index": 0}]}) as check, \
             patch.object(maya, "say"):
            self.assertEqual(maya.main(), 0)
        self.assertEqual(check.call_args.args[0].backend, "metal")

    def write(self, **kw):
        gguf = self.root / "models" / "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf"
        pc = maya.check_pc(self.a)
        return maya.write_config(self.a, pc, {}, self.root / "models" / "metal", "Maya-S-v2-IQ2_XXS", 16384,
                                 self.root / "data", kw.pop("vision", None), gguf=gguf, **kw), gguf

    def test_set_up_writes_the_gguf_once(self):
        # a merge once left a second write_config (without gguf=) after the Metal one: "--gguf", "None"
        shard = self.root / "models" / "m-00001-of-00003.gguf"
        self.a.calibrate, self.a.models_dir = False, None
        with patch.object(maya, "choose_family", return_value="glm"), \
             patch.object(maya, "choose_context", return_value=16384), \
             patch.object(maya, "choose_model", return_value=("yours", "m")), \
             patch.object(maya, "choose_draft", return_value="mtp"), \
             patch.object(maya, "tools_step"), patch.object(maya, "build_step", return_value={}), \
             patch.object(maya, "model_step", return_value=(shard.parent, [shard], "Maya-S-v2-IQ2_XXS")), \
             patch.object(maya, "metal_tokenizer_step", return_value=self.root / "models" / "metal"), \
             patch.object(maya, "vision_step", return_value=None), \
             patch.object(maya, "write_config", wraps=maya.write_config) as wc, patch.object(maya, "say"):
            p = maya.set_up(self.a, {})
        self.assertEqual(wc.call_count, 1)
        self.assertEqual(maya.read_json(p)["args"][:2], ["--gguf", str(shard)])

    def test_config_has_no_pack_and_no_cuda_arguments(self):
        p, gguf = self.write()
        cfg = maya.read_json(p)
        self.assertEqual(p.name, "maya-maya-s-v2-iq2_xxs-metal.json")
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["gpu"], [0])
        self.assertEqual(Path(cfg["exe"]).name, "maya-metal")
        # the learned expert profile lives with the tokenizer, in <model folder>/metal/
        self.assertEqual(cfg["args"], ["--gguf", str(gguf), "--max-context", "16384", "--expert-profile-save",
                                       str(self.root / "models" / "metal" / "expert-profile.bin")])
        self.assertEqual(cfg["tokenizer"], str(self.root / "models" / "metal" / "tokenizer"))
        self.assertNotIn("--glm-pack", cfg["args"])
        self.assertEqual(cfg["installer"]["draft"], "mtp")

    def test_config_carries_the_shipped_expert_profile_and_env(self):
        maya.PROFILE.write_bytes(b"STRP")
        self.a.env = ["MAYA_METAL_POOL_SLOTS=15"]
        p, gguf = self.write()
        cfg = maya.read_json(p)
        self.assertEqual(cfg["args"][-2:], ["--expert-profile", str(maya.PROFILE)])
        self.assertEqual(cfg["args"][:4], ["--gguf", str(gguf), "--max-context", "16384"])
        self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["env"]["MAYA_METAL_POOL_SLOTS"], "15")

    def test_dflash_drafter_is_a_metal_argument_and_mtp_stays_the_default(self):
        drafter = self.root / "GLM-5.3-Flash-DFlash2" / "d.gguf"
        cfg = maya.read_json(self.write(dflash=drafter)[0])
        self.assertEqual(maya.flag_value(cfg["args"], "--dflash"), str(drafter))
        self.assertEqual(cfg["installer"]["draft"], "dflash2")
        self.assertNotIn("--dflash", maya.read_json(self.write()[0])["args"])        # (a setup again: MTP)
        for given, want in ((None, "mtp"), ("dflash2", "dflash2"), ("mtp", "mtp")):
            self.a.draft = given
            self.a.yes = True
            self.assertEqual(maya.choose_draft(self.a, {"backend": "metal"}, {}), want)

    def test_dflash_on_metal_has_no_speed_calibration(self):
        cfg = {"backend": "metal", "args": ["--gguf", "m.gguf", "--dflash", "d.gguf"]}
        self.assertFalse(maya.needs_speed_calibration(cfg))
        path = self.root / "maya-metal.json"
        path.write_text(json.dumps(cfg))
        with patch.object(maya, "warn") as warn:
            self.assertFalse(maya.speed_calibration(path))
        self.assertIn("Metal", warn.call_args.args[0])

    def test_a_setup_again_keeps_the_learned_profile_file(self):
        p, _ = self.write()
        cfg = maya.read_json(p)
        mine = str(self.root / "mine.bin")
        maya.set_arg(cfg["args"], "--expert-profile-save", mine)
        p.write_text(json.dumps(cfg))
        self.assertEqual(maya.flag_value(maya.read_json(self.write()[0])["args"], "--expert-profile-save"), mine)

    def test_parallel_and_vram_reserve_do_not_reach_the_engine(self):
        self.a.parallel, self.a.vram_reserve_mib = 4, 2048
        cfg = maya.read_json(self.write()[0])
        self.assertNotIn("--parallel", cfg["args"])
        self.assertNotIn("--vram-reserve-mib", cfg["args"])

    def test_an_older_metal_config_gets_the_profile_at_a_start(self):
        maya.PROFILE.write_bytes(b"STRP")
        cfg = {"args": ["--gguf", "m.gguf", "--max-context", "8192"]}
        self.assertTrue(maya.add_profile(cfg))
        self.assertEqual(cfg["args"][-2:], ["--expert-profile", str(maya.PROFILE)])
        self.assertFalse(maya.add_profile(cfg))

    def test_vision_entry_on_metal_is_the_resident_encoder(self):
        a = SimpleNamespace(no_vision=False, download_model=False, yes=True, rebuild=False)
        quant = "Maya-S-v2-IQ2_XXS"
        spec = maya.MODELS[quant]["vision"]
        vd = self.root / "models" / spec["folder"]
        vd.mkdir(parents=True)
        for f in (spec["mmproj"], spec["vocab"]):
            (vd / f).write_bytes(b"x")
        exe = self.root / "build-vision" / "bin" / "maya-vision"
        pc = maya.check_pc(self.a)
        with patch.object(maya, "VEXE", exe), patch.object(maya, "VSTAMP", exe.parent.parent / "MAYA-BUILD.json"), \
             patch.object(maya.S, "source_hash", return_value="abc"), patch.object(maya, "compile_vision") as comp:
            comp.side_effect = lambda *x: exe.parent.mkdir(parents=True) or exe.touch() or True
            entry = maya.vision_step(a, pc, {}, self.root / "llama.cpp", self.root / "models", quant)
            comp.assert_called_once()
        self.assertEqual(entry, {"exe": str(exe), "mmproj": str(vd / spec["mmproj"]),
                                 "model": str(vd / spec["vocab"]), "gpu": True})
        cfg = maya.read_json(self.write(vision=entry)[0])
        self.assertEqual(cfg["vision"], entry)

    def test_vision_compiles_for_metal_without_cuda_flags(self):
        steps = []
        vb = self.root / "build-vision"
        vb.mkdir()                                     # (cmake makes it)
        with patch.object(maya, "VBUILD", vb), patch.object(maya, "VEXE", vb / "bin" / "maya-vision"), \
             patch.object(maya, "VSTAMP", vb / "MAYA-BUILD.json"), patch.object(maya, "run"), \
             patch.object(maya, "venv_tool", return_value="/usr/bin/ninja"), \
             patch.object(maya, "cmake_steps", side_effect=lambda *x, **k: steps.append(x) or None), \
             patch.object(maya, "say"):
            maya.compile_vision(maya.check_pc(self.a), {}, self.root / "llama.cpp", "abc")
        conf, build, env = steps[0][:3]
        self.assertEqual(conf[:3], ["/usr/bin/cmake", "-G", "Ninja"])
        for flag in ("-DCMAKE_BUILD_TYPE=Release", f"-DLLAMA_DIR={self.root / 'llama.cpp'}", str(vb)):
            self.assertIn(flag, conf)
        self.assertFalse([c for c in conf if "CUDA" in c], conf)
        self.assertEqual(build[build.index("--target") + 1], "maya-vision")
        self.assertIsNotNone(env)                      # (the shell's environment, not a Visual Studio .bat)
        self.assertEqual(maya.read_json(vb / "MAYA-BUILD.json")["src"], "abc")

    def test_no_tuning_for_metal(self):
        self.assertIsNone(maya.saved_calibration({"backend": "metal"}, any_context=True))
        cfg = self.root / "maya-metal.json"
        cfg.write_text(json.dumps({"backend": "metal"}))
        with patch.object(maya, "warn") as warn:
            self.assertFalse(maya.calibrate_config(cfg))
        self.assertIn("Metal", warn.call_args.args[0])

    def test_build_is_skipped_when_the_source_is_unchanged(self):
        maya.select_build_backend("metal")
        maya.EXE.parent.mkdir(parents=True)
        maya.EXE.touch()
        with patch.object(maya.S, "source_hash", return_value="abc"):
            meta = {"backend": "metal", "src": "abc"}
            maya.STAMP.write_text(json.dumps(meta))
            a = SimpleNamespace(rebuild=False)
            with patch.object(maya, "compile_engine_metal") as compile_:
                self.assertEqual(maya.build_step(a, {"backend": "metal"}, self.root / "llama.cpp"), meta)
                compile_.assert_not_called()
                maya.STAMP.write_text(json.dumps({"backend": "metal", "src": "old"}))
                maya.build_step(a, {"backend": "metal"}, self.root / "llama.cpp")
                compile_.assert_called_once_with(self.root / "llama.cpp", "abc")

    def test_compile_commands_and_stamp(self):
        maya.select_build_backend("metal")
        maya.BUILD.mkdir(parents=True)
        with patch.object(maya, "cmake_steps", return_value=None) as steps, patch.object(maya, "venv_tool", return_value=None), \
             patch.object(maya.shutil, "which", return_value=None), patch.object(maya, "say"), patch.object(maya, "ok"):
            meta = maya.compile_engine_metal(self.root / "llama.cpp", "abc")
        conf, build = steps.call_args.args[:2]
        for flag in ("-DMAYA_ENABLE_METAL=ON", "-DMAYA_METAL_TESTS=OFF", f"-DMAYA_GGML_DIR={self.root / 'llama.cpp'}",
                     "-DCMAKE_BUILD_TYPE=Release", str(self.root / "build-metal")):
            self.assertIn(flag, conf)
        self.assertEqual(build[build.index("--target") + 1], "maya-metal")
        self.assertEqual((meta["backend"], meta["src"]), ("metal", "abc"))
        self.assertEqual(maya.read_json(maya.STAMP)["src"], "abc")

    def test_refresh_rebuilds_a_changed_metal_engine(self):
        maya.select_build_backend("metal")
        maya.BUILD.mkdir(parents=True)
        maya.STAMP.write_text(json.dumps({"backend": "metal", "src": "old", "llama": str(self.root / "llama.cpp")}))
        (self.root / "llama.cpp/ggml").mkdir(parents=True)
        (self.root / "llama.cpp/ggml/CMakeLists.txt").touch()
        with patch.object(maya.S, "source_hash", return_value="new"), \
             patch.object(maya, "compile_engine_metal") as compile_:
            maya.refresh_engine({"backend": "metal", "exe": str(maya.EXE)})
        compile_.assert_called_once_with(self.root / "llama.cpp", "new", soft=True)

    def test_server_environment_for_metal(self):
        from serve.server import child_env
        cfg = {"backend": "metal", "gpu": [0], "env": {"MAYA_METAL_PP_CHUNK": "64"}}
        env = child_env(cfg)
        self.assertEqual(env["MAYA_METAL_PP_CHUNK"], "64")        # (maya-metal reads its settings by their MAYA_* names)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_a_failed_tokenizer_export_stops_the_setup(self):
        shard = self.root / "m-00001-of-00003.gguf"
        shard.touch()
        with patch.object(maya, "run", return_value=SimpleNamespace(returncode=1)), patch.object(maya, "step"), \
             self.assertRaises(SystemExit):
            maya.metal_tokenizer_step(SimpleNamespace(repack=False), self.root, [shard], "x")

    def test_tokenizer_is_exported_once(self):
        shard = self.root / "models" / "m-00001-of-00003.gguf"
        shard.parent.mkdir()
        shard.touch()
        with patch.object(maya, "run") as run, patch.object(maya, "ok"), patch.object(maya, "step"):
            def export(cmd, **kw):
                (self.root / "models/metal/tokenizer").mkdir(parents=True)
                (self.root / "models/metal/tokenizer/vocab.json").write_text("{}")
                return SimpleNamespace(returncode=0)
            run.side_effect = export
            out = maya.metal_tokenizer_step(SimpleNamespace(repack=False), shard.parent, [shard], "x")
            self.assertEqual(out, self.root / "models" / "metal")
            cmd = run.call_args.args[0]
            self.assertEqual((Path(cmd[1]).name, cmd[2:]), ("strata_tokenizer.py", ["--gguf", str(shard), "--out", str(out)]))
            maya.metal_tokenizer_step(SimpleNamespace(repack=False), shard.parent, [shard], "x")
            run.assert_called_once()                             # (the second step found it exported)


if __name__ == "__main__":
    unittest.main()
