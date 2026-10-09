"""Maya's HIP setup regressions; no GPU, network or model downloads required."""
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


class HipSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rocm = self.root / "rocm"
        (self.rocm / "llvm/bin").mkdir(parents=True)
        (self.rocm / "llvm/bin/clang++").touch()
        (self.rocm / "lib").mkdir()
        (self.rocm / "lib/libhipblas.so").touch()
        self.gpus = [
            {"index": 0, "arch": "gfx1100", "vendor": "amd", "name": "RX 7900 XT", "vram_gb": 20},
            {"index": 1, "arch": "gfx1201", "vendor": "amd", "name": "R9700", "vram_gb": 32},
            {"index": 2, "arch": "gfx1036", "vendor": "amd", "name": "iGPU", "vram_gb": 1},
        ]
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "BUILD", self.root / "build"),
            patch.object(maya, "EXE", self.root / "build/strata"),
            patch.object(maya, "STAMP", self.root / "build/MAYA-BUILD.json"),
            patch.object(maya, "WIN", False),
            patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}),
            patch.object(maya.S, "amd_gpus", return_value=self.gpus),
            patch.object(maya.S, "gpus", side_effect=AssertionError("HIP must not query NVIDIA")),
            patch.object(maya.S, "is_wsl", return_value=False),
            patch.object(maya.S, "cpu_info", return_value=("test CPU", True, True)),
            patch.object(maya, "mem_gb", return_value=(192, 128)),
            patch.object(maya, "tool_version", return_value=(7, 2)),
            patch.object(maya.shutil, "which", return_value="/usr/bin/c++"),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(backend="hip", gpu=0, gpus=None, no_vision=False, env=[],
                                 port=8099, host=None, api_key=None, gguf_dir=None)

    def test_selects_larger_supported_discrete_card(self):
        self.a.gpu = None
        pc = maya.check_pc(self.a)
        self.assertEqual(pc["gpus"], [self.gpus[1]])
        self.assertEqual(maya.EXE, self.root / "build-hip/strata")

    def test_rejects_unsupported_card_and_bad_gpu_lists(self):
        self.a.gpu = 2
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a)
        self.a.gpu = None
        for bad in ("0,2", "0,1,2", "1,1", "x"):
            self.a.gpus = bad
            with self.assertRaises(SystemExit):
                maya.check_pc(self.a)

    def test_two_gpu_split_config(self):
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        for arch in ("gfx1100", "gfx1201"):
            (tables / f"{arch}-glm-hipblaslt-100202.txt").write_text(f"STRATA_HIPBLASLT_TUNING_V1 {arch} 100202\n")
        self.a.gpu = None
        self.a.gpus = "0,1"
        pc = maya.check_pc(self.a)
        self.assertEqual([g["index"] for g in pc["gpus"]], [1, 0])   # the larger card first
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "lib")]},
                              self.root / "pack", "test", 8192, self.root / "data", None)
        cfg = json.loads(p.read_text())
        self.assertEqual(cfg["gpu"], [1, 0])
        self.assertNotIn("STRATA_GLM_SPLIT", cfg["env"])
        self.assertEqual(cfg["env"]["STRATA_HIPBLASLT_TUNING"],
                         f"{tables / 'gfx1201-glm-hipblaslt-100202.txt'}:{tables / 'gfx1100-glm-hipblaslt-100202.txt'}")

    def test_config_selects_hip_and_preserves_user_tuning(self):
        pc = maya.check_pc(self.a)
        self.a.env = ["STRATA_GLM_RESERVE_MB=4096"]
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "lib")]},
                              self.root / "pack", "test", 8192, self.root / "data", None)
        cfg = json.loads(p.read_text())
        self.assertEqual(cfg["backend"], "hip")
        self.assertEqual(cfg["gpu"], [0])
        self.assertEqual(cfg["env"]["STRATA_GLM_SPLIT"], "0")
        self.assertEqual(cfg["env"]["STRATA_GLM_RESERVE_MB"], "4096")
        self.assertEqual(cfg["exe"], str(self.root / "build-hip/strata"))
        self.assertNotIn("vision", cfg)

    def test_config_prompt_defaults_and_tuning_table(self):
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        (tables / "gfx1100-glm-hipblaslt-100202.txt").write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1100 100202\n")
        (tables / "gfx1201-glm-hipblaslt-100202.txt").write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1201 100202\n")
        pc = maya.check_pc(self.a)
        self.a.env = ["STRATA_GLM_PREFILL_SUB=512"]
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "lib")]},
                              self.root / "pack", "test", 8192, self.root / "data", None)
        env = json.loads(p.read_text())["env"]
        self.assertNotIn("STRATA_GLM_PREFILL_CHUNK", env)
        self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "512")   # the user's setting wins
        self.assertEqual(env["STRATA_GLM_PREFILL_MB"], "4096")
        self.assertEqual(env["STRATA_GLM_RESERVE_MB"], "3072")
        self.assertEqual(env["STRATA_GLM_RAM_HEADROOM_GB"], "16")
        self.assertEqual(env["STRATA_HIPBLASLT_TUNING"], str(tables / "gfx1100-glm-hipblaslt-100202.txt"))

    def test_gpus_with_an_apu_is_rejected(self):
        self.gpus.append({"index": 3, "arch": "gfx1151", "vendor": "amd", "name": "Radeon 8060S", "vram_gb": 4})
        self.a.gpu = None
        self.a.gpus = "1,3"
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a)
        self.a.gpus = None
        self.a.gpu = 3
        self.assertEqual(maya.check_pc(self.a)["gpus"], [self.gpus[3]])

    def test_strix_halo_defaults_use_system_ram_not_vram(self):
        self.gpus.append({"index": 3, "arch": "gfx1151", "vendor": "amd",
                          "name": "AMD Radeon (gfx1151)", "vram_gb": 0.5})
        self.a.gpu = 3
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        table = tables / "gfx1151-glm-hipblaslt-100202.txt"
        table.write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1151 100202\n")
        pc = maya.check_pc(self.a)
        self.assertEqual(pc["gpus"], [self.gpus[3]])
        self.assertTrue(maya.hip_unified_memory(pc["gpus"][0]))
        for ram, budget in ((64, "4096"), (96, "6144"), (128, "6144")):
            with self.subTest(ram=ram), patch.object(maya, "mem_gb", return_value=(ram, ram - 8)):
                p = maya.write_config(self.a, pc, {}, self.root / "pack", "test", 8192,
                                      self.root / "data", None)
                env = json.loads(p.read_text())["env"]
                self.assertEqual(env["STRATA_GLM_SPLIT"], "0")
                self.assertEqual(env["STRATA_GLM_RESERVE_MB"], "1024")
                self.assertEqual(env["STRATA_GLM_RAM_HEADROOM_GB"], "16")
                self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "1024")
                self.assertEqual(env["STRATA_GLM_PREFILL_MB"], budget)
                self.assertEqual(env["STRATA_HIPBLASLT_TUNING"], str(table))
                self.assertNotIn("STRATA_GLM_POOL_GB", env)
                self.assertNotIn("STRATA_GLM_RAM_GB", env)
                self.assertNotIn("STRATA_GLM_PREFILL_CHUNK", env)

    def test_strix_halo_explicit_settings_win(self):
        self.gpus.append({"index": 3, "arch": "gfx1151", "vendor": "amd",
                          "name": "Radeon 8060S", "vram_gb": 112})
        self.a.gpu = 3
        explicit = {"STRATA_GLM_RESERVE_MB": "2048", "STRATA_GLM_RAM_HEADROOM_GB": "8",
                    "STRATA_GLM_POOL_GB": "84", "STRATA_GLM_RAM_GB": "4",
                    "STRATA_GLM_PREFILL_SUB": "512", "STRATA_GLM_PREFILL_MB": "3072",
                    "STRATA_HIPBLASLT_TUNING": "/custom/table.txt"}
        self.a.env = [f"{k}={v}" for k, v in explicit.items()]
        p = maya.write_config(self.a, maya.check_pc(self.a), {}, self.root / "pack", "test", 8192,
                              self.root / "data", None)
        env = json.loads(p.read_text())["env"]
        for k, v in explicit.items():
            self.assertEqual(env[k], v)

    def test_discrete_r9700_defaults_unchanged(self):
        self.a.gpu = 1
        p = maya.write_config(self.a, maya.check_pc(self.a), {}, self.root / "pack", "test", 8192,
                              self.root / "data", None)
        self.assertEqual(json.loads(p.read_text())["env"], {
            "STRATA_GLM_SPLIT": "0", "STRATA_GLM_RESERVE_MB": "3072", "STRATA_GLM_RAM_HEADROOM_GB": "16",
            "STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "4096"})
        self.assertFalse(maya.hip_unified_memory(self.gpus[1]))

    def test_hip_build_enables_mmq_and_never_cuda(self):
        pc = maya.check_pc(self.a)
        maya.BUILD.mkdir()
        with patch.object(maya, "pick_cmake", return_value="cmake"), \
                patch.object(maya, "cmake_steps", return_value=None) as build:
            meta = maya.compile_engine_hip(pc, self.root / "llama", "source-sha")
        conf, _, env, _ = build.call_args.args
        self.assertIn("-DSTRATA_ENABLE_HIP=ON", conf)
        self.assertIn("-DSTRATA_ENABLE_CUDA=OFF", conf)
        self.assertIn("-DSTRATA_PREFILL_MMQ=ON", conf)
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201;gfx1151", conf)
        self.assertEqual(env["ROCM_PATH"], str(self.rocm))
        self.assertEqual(meta["backend"], "hip")

    def test_hip_skips_vision_without_downloading(self):
        with patch.object(maya.S, "download", side_effect=AssertionError("no vision downloads")):
            self.assertIsNone(maya.vision_step(self.a, {"backend": "hip"}, {}, self.root, self.root, "test"))

    def installed_config(self, backend="hip", gpu=None):
        tp = self.root / "tokenizer"
        tp.mkdir(exist_ok=True)
        (tp / "vocab.json").write_text('{"hello": 0}')
        (tp / "merges.txt").write_text("")
        (tp / "token_type.json").write_text("[1]")
        # A stale CUDA executable in a HIP config must not select the CUDA build.
        cfg = {"exe": str(self.root / "build/strata"), "args": ["--max-context", "16384"],
               "tokenizer": str(tp), "gpu": [0] if gpu is None else gpu,
               "env": {"STRATA_GLM_PREFILL_SUB": "512", "STRATA_GLM_RAM_GB": "60"},
               "lib_dirs": [str(self.rocm / "lib")]}
        if backend == "hip":
            cfg["backend"] = backend
        path = self.root / f"maya-test-{backend}.json"
        path.write_text(json.dumps(cfg))
        return path

    def run_bench(self, path):
        proc = MagicMock()
        proc.stdout.readline.side_effect = ["READY\n"] + ["DONE 256 0 1000 2000\n"] * 7
        tok = MagicMock()
        tok.encode.return_value = list(range(9000))
        # Exercise the real server's engine_args/child_env without optional tokenizer/template packages.
        modules = {"strata_tokenizer": SimpleNamespace(Tokenizer=SimpleNamespace(from_dir=MagicMock(return_value=tok))),
                   "serve.frontend": MagicMock()}
        with patch("urllib.request.urlopen", side_effect=OSError("no server")), \
                patch.dict(sys.modules, modules), \
                patch.object(maya.S, "out", side_effect=AssertionError("bench must not run GPU queries")), \
                patch.object(maya.subprocess, "Popen", return_value=proc) as popen, \
                patch.dict(os.environ, {}, clear=True):
            self.assertEqual(maya.bench(path, "test"), 0)
        proc.wait.assert_called_once_with(timeout=120)
        self.assertIn("QUIT\n", [call.args[0] for call in proc.stdin.write.call_args_list])
        text = (self.root / "maya-bench.txt").read_text()
        self.assertIn("128.0 tokens/s over 3 answers", text)
        self.assertIn("2048 tokens/s, 2048 tokens", text)
        self.assertIn("8192 tokens/s, 8192 tokens", text)
        return popen.call_args, text

    def test_bench_uses_hip_build_and_config_env(self):
        call, text = self.run_bench(self.installed_config(gpu=[1, 0]))
        self.assertEqual(maya.BUILD, self.root / "build-hip")
        self.assertEqual(maya.STAMP, self.root / "build-hip/MAYA-BUILD.json")
        self.assertEqual(call.args[0][:2], [str(self.root / "build-hip/strata"), "--serve"])
        self.assertEqual(call.args[0][-2:], ["--layer-split", "auto"])
        env = call.kwargs["env"]
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "1,0")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("CUDA_DEVICE_ORDER", env)
        self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "512")
        self.assertEqual(env["STRATA_GLM_RAM_GB"], "60")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(self.rocm / "lib"))
        self.assertIn("GPUs: R9700 32 GB, RX 7900 XT 20 GB", text)
        self.assertIn("GPUs [1, 0]", text)

    def test_bench_cuda_command_and_env_unchanged(self):
        path = self.installed_config(backend="cuda", gpu=[1, 0])
        cfg = json.loads(path.read_text())
        cfg["exe"] = str(self.root / "custom-cuda/strata")
        path.write_text(json.dumps(cfg))
        with patch.object(maya.S, "gpus", return_value=[{"name": "V100", "vram_gb": 16}]), \
                patch.object(maya.S, "amd_gpus", side_effect=AssertionError("CUDA must not query AMD")):
            call, text = self.run_bench(path)
        self.assertEqual(call.args[0][:2], [cfg["exe"], "--serve"])
        self.assertEqual(maya.BUILD, self.root / "build")
        self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1,0")
        self.assertEqual(call.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("HIP_VISIBLE_DEVICES", call.kwargs["env"])
        self.assertIn("GPUs: V100 16 GB", text)

    def test_bench_single_hip_gpu(self):
        call, text = self.run_bench(self.installed_config(gpu=0))
        self.assertEqual(call.kwargs["env"]["HIP_VISIBLE_DEVICES"], "0")
        self.assertNotIn("--layer-split", call.args[0])
        self.assertIn("GPUs: RX 7900 XT 20 GB;", text)

    def run_report(self, outputs=None, tools=(), backend=None, cfg_path=None):
        outputs = outputs or {}

        def out(cmd):
            self.assertNotEqual(Path(cmd[0]).name, "nvidia-smi", "HIP report must not query NVIDIA")
            return outputs.get(Path(cmd[0]).name, "")

        with patch.object(maya.S, "out", side_effect=out) as query, \
                patch.object(maya.S, "page_file_gb", return_value=None), \
                patch.object(maya.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}" if name in tools else None):
            self.assertEqual(maya.report("test", backend, cfg_path), 0)
        return (self.root / "maya-report.txt").read_text(), query

    def test_report_rocm_smi_json_and_engine_speed_lines(self):
        cfg_path = self.installed_config()
        log = cfg_path.with_suffix(".log")
        speed = ["glm fast: CUDA0 expert pool 12 GB", "glm prefill: CUDA0 chunks of 1024 tokens",
                 "glm prefill: HIP0 chunks of 1024 tokens", "glm prefill: 2048 tokens at 413 tok/s",
                 "glm stat: decode 50.0 ms/tok | prompt 400 tok/s",
                 "glm stat: decode 0.1 ms/tok vram hit 0.00% | prompt 413 tok/s"]
        log.write_text("\n".join(speed) + "\n")
        (self.root / "build-hip").mkdir()
        (self.root / "build-hip/MAYA-BUILD.json").write_text(json.dumps({
            "backend": "hip", "archs": ["gfx1100", "gfx1201"], "rocm": str(self.rocm), "date": "test-date"}))
        (self.rocm / ".info").mkdir()
        (self.rocm / ".info/version").write_text("7.2.1\n")
        # SMI names differ from KFD: keep both sources without assuming their device order matches.
        smi = {"card0": {"Card series": "AMD Radeon RX 7900 XT", "VRAM Total Memory (B)": str(20 * 2**30)},
               "card1": {"Card series": "AMD Radeon AI PRO R9700", "VRAM Total Memory (B)": str(32 * 2**30)},
               "system": {"Driver version": "6.14.0-37"}}
        text, query = self.run_report({"rocm-smi": json.dumps(smi)}, tools=("rocm-smi",))
        self.assertIn("GPU 0 (RX 7900 XT, 20 GB, gfx1100)", text)
        self.assertIn("GPU 1 (R9700, 32 GB, gfx1201)", text)
        self.assertIn("AMD Radeon RX 7900 XT", text)
        self.assertIn("AMD Radeon AI PRO R9700", text)
        self.assertIn(str(32 * 2**30), text)
        self.assertIn("6.14.0-37", text)
        self.assertIn(f"path {self.rocm}\nversion 7.2.1", text)
        self.assertIn('"backend": "hip"', text)
        self.assertIn("test-date", text)
        self.assertIn("the speed and memory lines (last 80 of 6)", text)
        for line in maya.speed_lines(speed):
            self.assertIn(line, text)
        query.assert_any_call(["/usr/bin/rocm-smi", "--showproductname", "--showmeminfo", "vram",
                               "--showdriverversion", "--json"])

    def test_report_kfd_fallback_without_smi_or_rocm_version(self):
        self.installed_config()
        text, query = self.run_report()
        self.assertIn("GPU 0 (RX 7900 XT, 20 GB, gfx1100)", text)
        self.assertIn("GPU 1 (R9700, 32 GB, gfx1201)", text)
        self.assertIn("amdgpu driver:", text)
        self.assertIn("kernel", text)
        self.assertIn("using KFD topology", text)
        self.assertIn(f"path {self.rocm}", text)
        self.assertIn("version unknown", text)
        self.assertFalse(any(Path(c.args[0][0]).name in ("rocm-smi", "amd-smi", "hipcc")
                             for c in query.call_args_list))

    def test_report_amd_smi_after_bad_rocm_smi_json(self):
        self.installed_config()
        smi = [{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XT", "target_graphics_version": "gfx1100"},
                "vram": {"size": "20480 MB"}, "driver": {"driver_version": "6.14.0", "rocm_version": "7.2.1"}}]
        text, query = self.run_report({"rocm-smi": "not JSON", "amd-smi": json.dumps(smi)},
                                      tools=("rocm-smi", "amd-smi"))
        self.assertIn("amd-smi (tool GPU indices)", text)
        self.assertIn("AMD Radeon RX 7900 XT", text)
        self.assertIn("20480 MB", text)
        self.assertIn("6.14.0", text)
        self.assertIn("7.2.1", text)
        query.assert_any_call(["/usr/bin/amd-smi", "static", "--json"])

    def test_report_kfd_fallback_for_empty_or_invalid_smi_json(self):
        self.installed_config()
        for response in ("", "warning: no access", "{}", "[]", "null", '"error"'):
            with self.subTest(response=response):
                text, _ = self.run_report({"rocm-smi": response, "amd-smi": response},
                                          tools=("rocm-smi", "amd-smi"))
                self.assertIn("GPU 1 (R9700, 32 GB, gfx1201)", text)
                self.assertIn("using KFD topology", text)

    def test_report_finds_smi_in_rocm_bin_and_hipcc_version(self):
        self.installed_config()
        (self.rocm / "bin").mkdir()
        (self.rocm / "bin/rocm-smi").touch()
        (self.rocm / "bin/hipcc").touch()
        text, query = self.run_report({"rocm-smi": '{"system": {"Driver version": "6.14.0"}}',
                                       "hipcc": "HIP version: 7.2.1"})
        self.assertIn("rocm-smi (tool GPU indices)", text)
        self.assertIn("version HIP version: 7.2.1", text)
        query.assert_any_call([str(self.rocm / "bin/hipcc"), "--version"])

    def test_report_hip_without_config_or_gpus(self):
        with patch.object(maya.S, "amd_gpus", return_value=[]):
            text, _ = self.run_report(backend="hip")
        self.assertIn("no AMD GPUs found in KFD topology", text)
        self.assertIn("## ROCm", text)
        self.assertIn("not compiled yet", text)

    def test_report_explicit_config_selects_hip_env_and_log(self):
        self.installed_config(backend="cuda")
        path = self.installed_config().rename(self.root / "chosen.json")
        cfg = json.loads(path.read_text())
        root = self.root / "chosen-rocm"
        (root / ".info").mkdir(parents=True)
        (root / ".info/version").write_text("7.2.2\n")
        cfg["env"]["ROCM_PATH"] = str(root)
        path.write_text(json.dumps(cfg))
        path.with_suffix(".log").write_text("glm stat: decode 50.0 ms/tok | prompt 413 tok/s\n")
        text, _ = self.run_report(cfg_path=path)
        self.assertIn(f"path {root}\nversion 7.2.2", text)
        self.assertIn("Setup chosen.json", text)
        self.assertIn("Engine log chosen.log: the speed and memory lines", text)
        self.assertNotIn("Setup maya-test-cuda.json", text)

    def test_report_cuda_query_and_stamp_unchanged(self):
        self.installed_config(backend="cuda")
        (self.root / "build").mkdir()
        stamp = {"archs": [70], "nvcc": "/usr/local/cuda/bin/nvcc", "host_compiler": "g++-12", "date": "test-date"}
        (self.root / "build/MAYA-BUILD.json").write_text(json.dumps(stamp))
        with patch.object(maya.S, "out", return_value="NVIDIA V100") as query, \
                patch.object(maya.S, "page_file_gb", return_value=None), \
                patch.object(maya.S, "amd_gpus", side_effect=AssertionError("CUDA must not query AMD")):
            self.assertEqual(maya.report("test"), 0)
        query.assert_any_call(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,"
                                             "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,"
                                             "temperature.gpu", "--format=csv"])
        text = (self.root / "maya-report.txt").read_text()
        self.assertIn(json.dumps(stamp), text)
        self.assertNotIn("## ROCm", text)

    def test_cli_backend_selects_hip_config_for_bench(self):
        hip = self.installed_config()
        cuda = self.installed_config(backend="cuda")
        with patch.object(maya, "configs", return_value=[cuda, hip]), \
                patch.object(sys, "argv", ["maya.py", "--backend", "hip", "--bench"]), \
                patch.object(maya, "bench", return_value=0) as bench:
            self.assertEqual(maya.main(), 0)
        bench.assert_called_once_with(hip, (maya.HERE / "VERSION").read_text().strip())

    def test_cli_explicit_config_for_bench_and_report(self):
        path = self.installed_config()
        for action in ("bench", "report"):
            with self.subTest(action=action), \
                    patch.object(sys, "argv", ["maya.py", "--backend", "hip", f"--{action}", "--config", str(path)]), \
                    patch.object(maya, action, return_value=0) as run:
                self.assertEqual(maya.main(), 0)
                self.assertIn(path, run.call_args.args)

    def test_cli_backend_never_benchmarks_cuda_config_as_hip(self):
        self.installed_config(backend="cuda")
        with patch.object(sys, "argv", ["maya.py", "--backend", "hip", "--bench"]), \
                patch.object(maya, "bench", side_effect=AssertionError("must not start the CUDA engine")):
            with self.assertRaises(SystemExit):
                maya.main()


class ConfigsTests(unittest.TestCase):
    def test_dashboard_settings_beside_a_config_are_not_a_config(self):
        with tempfile.TemporaryDirectory() as d, patch.object(maya, "ROOT", Path(d)):
            cfg = Path(d) / "maya-maya-l-hip.json"
            cfg.write_text(json.dumps({"backend": "hip", "exe": "strata.exe"}))
            side = Path(d) / "maya-maya-l-hip.shared-settings.json"
            side.write_text(json.dumps({"temperature": 0.7}))
            os.utime(cfg, (1, 1))   # the settings file is the newer one, as after a dashboard change
            self.assertEqual(maya.configs(), [cfg])


HIPINFO = """
--------------------------------------------------------------------------------
device#                           0
Name:                             AMD Radeon(TM) 8065S Graphics
pciBusID:                         195
totalGlobalMem:                   96.00 GB
gcnArchName:                      gfx1151
isIntegrated:                     1
--------------------------------------------------------------------------------
device#                           1
Name:                             AMD Radeon RX 7900 XTX
totalGlobalMem:                   23.98 GB
gcnArchName:                      gfx1100:sramecc-:xnack-
isIntegrated:                     0
"""


class WindowsDetectionTests(unittest.TestCase):
    def test_hipinfo_lists_gpus_in_hip_order(self):
        gpus = maya.S.parse_hipinfo(HIPINFO)
        self.assertEqual([(g["index"], g["arch"]) for g in gpus], [(0, "gfx1151"), (1, "gfx1100")])
        self.assertEqual(gpus[0]["name"], "AMD Radeon(TM) 8065S Graphics")
        self.assertAlmostEqual(gpus[0]["vram_gb"], 96.0)
        self.assertTrue(gpus[0]["integrated"])
        self.assertFalse(gpus[1]["integrated"])
        self.assertEqual(maya.S.parse_hipinfo("hipInfo: no devices\n"), [])

    def test_registry_adapters_by_device_id_then_name(self):
        one = json.dumps({"DriverDesc": "AMD Radeon(TM) 8060S Graphics",
                          "MatchingDeviceId": "pci\\ven_1002&dev_1586&rev_c1", "DriverVersion": "32.0.21",
                          "HardwareInformation.qwMemorySize": 96 * 2**30})
        g, = maya.S.parse_display_adapters(one)            # one adapter: an object, not a list
        self.assertEqual((g["index"], g["arch"], g["driver"]), (0, "gfx1151", "32.0.21"))
        self.assertAlmostEqual(g["vram_gb"], 96.0)
        rows = json.dumps([{"DriverDesc": "AMD Radeon(TM) 8065S Graphics", "MatchingDeviceId": "pci\\ven_1002&dev_9999"},
                           {"DriverDesc": "AMD Radeon RX 9070 XT", "MatchingDeviceId": "pci\\ven_1002&dev_7550"},
                           {"DriverDesc": "AMD Radeon(TM) Graphics", "MatchingDeviceId": "pci\\ven_1002&dev_164e"}])
        self.assertEqual([g["arch"] for g in maya.S.parse_display_adapters(rows)], ["gfx1151", "gfx1201", "unknown"])
        self.assertEqual(maya.S.parse_display_adapters(""), [])
        self.assertEqual(maya.S.parse_display_adapters("not json"), [])

    def test_windows_apu_names_are_unified_memory(self):
        for name in ("AMD Radeon(TM) 8060S Graphics", "AMD Radeon(TM) 8065S Graphics", "Radeon 8050S"):
            self.assertTrue(maya.hip_unified_memory({"arch": "unknown", "name": name}), name)
        self.assertFalse(maya.hip_unified_memory({"arch": "gfx1100", "name": "AMD Radeon RX 7900 XTX"}))


ROCM_DEVICES_MISSING, FIND_VCVARS = maya.rocm_devices_missing, maya.S.find_vcvars   # (setUp replaces them)


class WindowsHipSetupTests(unittest.TestCase):
    """START-MAYA.bat --backend hip on a Strix Halo / Gorgon Halo PC, with TheRock's ROCm wheels."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rocm = self.root / "_rocm_sdk_devel"        # TheRock's layout
        (self.rocm / "lib/llvm/bin").mkdir(parents=True)
        (self.rocm / "lib/llvm/bin/clang++.exe").touch()
        (self.rocm / "lib/llvm/bin/clang.exe").touch()
        (self.rocm / "lib/hipblas.lib").touch()
        (self.rocm / "bin").mkdir()
        (self.rocm / "bin/hipblas.dll").touch()
        self.apu = {"index": 0, "arch": "gfx1151", "vendor": "amd", "name": "AMD Radeon(TM) 8065S Graphics",
                    "vram_gb": 96, "driver": "windows"}
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "BUILD", self.root / "build"),
            patch.object(maya, "EXE", self.root / "build/strata.exe"),
            patch.object(maya, "STAMP", self.root / "build/MAYA-BUILD.json"),
            patch.object(maya, "WIN", True),
            patch.object(maya.S, "amd_gpus", return_value=[self.apu]),
            patch.object(maya.S, "gpus", side_effect=AssertionError("HIP must not query NVIDIA")),
            patch.object(maya.S, "find_vcvars", return_value=Path("C:/VS/vcvars64.bat")),
            patch.object(maya.S, "cpu_info", return_value=("AMD RYZEN AI MAX+ PRO 495", True, True)),
            patch.object(maya, "mem_gb", return_value=(32, 24)),
            patch.object(maya, "tool_version", return_value=(7, 14)),
            patch.object(maya, "rocm_devices_missing", return_value=[]),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(backend="hip", gpu=None, gpus=None, no_vision=False, env=[], check=False, yes=True,
                                 port=8099, host=None, api_key=None, gguf_dir=None)

    def test_visual_studio_is_checked_before_the_rocm_download(self):
        with patch.object(maya.S, "find_vcvars", return_value=None), \
                patch.object(maya, "windows_rocm", side_effect=AssertionError("no ROCm lookup without VS")), \
                patch.object(maya, "run", side_effect=AssertionError("no pip without VS")):
            with self.assertRaises(SystemExit):
                maya.check_pc(self.a)

    def test_visual_studio_2026_is_taken_for_hip_not_for_cuda(self):
        pf = self.root / "pf86"
        (pf / "Microsoft Visual Studio/Installer").mkdir(parents=True)
        (pf / "Microsoft Visual Studio/Installer/vswhere.exe").touch()
        vs = self.root / "VS2026"
        (vs / "VC/Auxiliary/Build").mkdir(parents=True)
        (vs / "VC/Auxiliary/Build/vcvars64.bat").touch()
        only_2026 = lambda cmd: "" if "-version" in cmd else str(vs)
        with patch.dict(os.environ, {"ProgramFiles(x86)": str(pf)}), patch.object(maya.S, "out", side_effect=only_2026):
            self.assertIsNone(FIND_VCVARS())                         # CUDA 12/13: 2019 or 2022 only
            self.assertEqual(FIND_VCVARS(cuda=False), vs / "VC/Auxiliary/Build/vcvars64.bat")

    def test_rocm_sdk_init_runs_before_any_path_query(self):
        sdk = str(self.root / "venv/Scripts/rocm-sdk.exe")
        calls = []

        def out(cmd):
            calls.append(("out", cmd[1:]))
            return {"--root": f"{self.rocm}\n", "--bin": f"{self.rocm / 'bin'}\n"}.get(cmd[-1], "")
        env = {k: v for k, v in os.environ.items() if k != "ROCM_PATH"}
        with patch.dict(os.environ, env, clear=True), patch.object(maya, "rocm_sdk", return_value=sdk), \
                patch.object(maya, "run", side_effect=lambda cmd, **kw: calls.append(("run", cmd[1:]))), \
                patch.object(maya.S, "out", side_effect=out):
            self.assertEqual(maya.windows_rocm(), (self.rocm, self.rocm / "bin"))
        self.assertEqual(calls[0], ("run", ["init"]))                # unpacks, seen and with no time limit
        self.assertEqual([c[1] for c in calls[1:]], [["path", "--root"], ["path", "--bin"]])

    def test_missing_device_wheel_is_offered(self):
        with patch.object(maya, "windows_rocm", return_value=(self.rocm, self.rocm / "bin")), \
                patch.object(maya, "rocm_devices_missing", side_effect=[["gfx1151"], []]), \
                patch.object(maya, "run") as pip, patch.object(sys, "base_prefix", "/somewhere/else"):
            pc = maya.check_pc(self.a)
        self.assertIn(f"rocm[libraries,devel,device-gfx1151]=={maya.THEROCK_VERSION}", pip.call_args.args[0])
        self.assertEqual(pc["gpus"], [self.apu])
        self.a.check = True                                       # --check installs nothing: it says so
        with patch.object(maya, "windows_rocm", return_value=(self.rocm, self.rocm / "bin")), \
                patch.object(maya, "rocm_devices_missing", return_value=["gfx1151"]), \
                patch.object(maya, "run", side_effect=AssertionError("--check installs nothing")), \
                patch.object(maya, "warn") as warned:
            maya.check_pc(self.a)
        self.assertTrue(any("no library kernels for gfx1151" in c.args[0] for c in warned.call_args_list))

    def test_device_wheels_are_looked_up_only_in_mayas_venv(self):
        sdk = str(self.root / "venv/Scripts/rocm-sdk.exe")
        Path(sdk).parent.mkdir(parents=True)
        Path(sdk).touch()
        from importlib import metadata

        def version(name):
            if name == "rocm-sdk-device-gfx1151":
                raise metadata.PackageNotFoundError(name)
            return maya.THEROCK_VERSION
        env = {k: v for k, v in os.environ.items() if k not in ("ROCM_PATH", "ROCM_VENV")}
        real = ROCM_DEVICES_MISSING
        with patch.dict(os.environ, env, clear=True), patch.object(maya, "venv_tool", return_value=sdk), \
                patch("importlib.metadata.version", side_effect=version):
            self.assertEqual(real(["gfx1151", "gfx1100", "gfx1151"]), ["gfx1151"])
            with patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}):
                self.assertEqual(real(["gfx1151"]), [])             # another ROCm: not this Python's to check
            with patch.dict(os.environ, {"ROCM_VENV": str(self.root / "other")}), \
                    patch.object(maya.shutil, "which", return_value=None):
                (self.root / "other/Scripts").mkdir(parents=True)
                (self.root / "other/Scripts/rocm-sdk.exe").touch()
                self.assertEqual(real(["gfx1151"]), [])

    def test_hipinfo_that_sees_no_gpu_warns_before_the_download(self):
        with patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}), \
                patch.object(maya, "hip_info", return_value=str(self.rocm / "bin/hipInfo.exe")), \
                patch.object(maya.S, "amd_gpus", return_value=[dict(self.apu, source="registry")]), \
                patch.object(maya, "warn") as warned:
            maya.check_pc(self.a)
        self.assertTrue(any("hipInfo sees no AMD GPU" in c.args[0] for c in warned.call_args_list))

    def test_check_finds_therock_and_the_apu(self):
        with patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}):
            pc = maya.check_pc(self.a)
        self.assertEqual(pc["gpus"], [self.apu])
        self.assertEqual(pc["rocm"], str(self.rocm.resolve()))
        self.assertEqual(pc["rocm_bin"], str(self.rocm.resolve() / "bin"))
        self.assertEqual(maya.EXE, self.root / "build-hip/strata.exe")

    def test_check_without_rocm_says_the_pip_command(self):
        with patch.object(maya, "windows_rocm", return_value=(None, None)), \
                patch.object(maya, "run", side_effect=AssertionError("--check installs nothing")), \
                patch.object(maya, "fail", side_effect=SystemExit) as stop:
            self.a.check = True
            with self.assertRaises(SystemExit):
                maya.check_pc(self.a)
        hint = stop.call_args.args[1]
        self.assertIn(maya.THEROCK_INDEX, hint)
        self.assertIn(f"rocm[libraries,devel,device-gfx1151]=={maya.THEROCK_VERSION}", hint)

    def test_setup_installs_rocm_into_the_venv_then_checks_again(self):
        found = iter([(None, None), (self.rocm, self.rocm / "bin"), (self.rocm, self.rocm / "bin")])
        with patch.object(maya, "windows_rocm", side_effect=lambda: next(found)), \
                patch.object(maya, "run") as pip, patch.object(sys, "base_prefix", "/somewhere/else"):
            pc = maya.check_pc(self.a)
        cmd = pip.call_args.args[0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn(maya.THEROCK_INDEX, cmd)
        self.assertEqual(pc["rocm"], str(self.rocm))

    def test_check_needs_visual_studio(self):
        with patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}), \
                patch.object(maya.S, "find_vcvars", return_value=None):
            with self.assertRaises(SystemExit):
                maya.check_pc(self.a)

    def test_windows_build_uses_therock_clang_and_ninja(self):
        with patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}):
            pc = maya.check_pc(self.a)
        maya.BUILD.mkdir()
        for dll in ("amdhip64_7.dll", "amd_comgr0702.dll", "rocm_kpack.dll"):
            (self.rocm / "bin" / dll).touch()
        with patch.object(maya, "pick_cmake", return_value="cmake"), \
                patch.object(maya, "venv_tool", return_value="C:/maya/.venv/Scripts/ninja.exe"), \
                patch.object(maya, "cmake_steps", return_value=None) as build:
            meta = maya.compile_engine_hip(pc, self.root / "llama", "source-sha")
        conf, _, env, bat = build.call_args.args
        self.assertEqual(build.call_args.kwargs["vcvars"], Path("C:/VS/vcvars64.bat"))
        # the HIP runtime it was built with sits next to strata.exe (else System32's, the driver's, loads first);
        # the libraries stay in lib_dirs
        self.assertEqual(sorted(p.name for p in maya.EXE.parent.glob("*.dll")),
                         ["amd_comgr0702.dll", "amdhip64_7.dll", "rocm_kpack.dll"])
        root = self.rocm.resolve().as_posix()
        self.assertEqual(conf[1:3], ["-G", "Ninja"])
        for want in (f"-DCMAKE_CXX_COMPILER={root}/lib/llvm/bin/clang++.exe",
                     f"-DCMAKE_C_COMPILER={root}/lib/llvm/bin/clang.exe",
                     f"-DCMAKE_HIP_COMPILER={root}/lib/llvm/bin/clang++.exe",
                     f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                     "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_PREFILL_MMQ=ON",
                     "-DSTRATA_PORTABLE=ON", "-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201;gfx1151"):
            self.assertIn(want, conf)
        if " " not in root:
            self.assertIn(f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={root}/lib/llvm/amdgcn/bitcode",
                          conf)
        self.assertEqual(env["HIP_DEVICE_LIB_PATH"], str(self.rocm.resolve() / "lib/llvm/amdgcn/bitcode"))
        self.assertEqual(env["HIP_PATH"], str(self.rocm.resolve()))
        self.assertTrue(bat.endswith(".bat"))
        self.assertEqual(meta["lib_dirs"], [str(self.rocm.resolve() / "bin")])

    def hip_sdk(self):
        """AMD's HIP SDK for Windows' layout: clang, hipInfo and the DLLs in bin, the device libraries in amdgcn."""
        sdk = self.root / "Program Files/AMD/ROCm/7.2"
        for f in ("bin/clang++.exe", "bin/clang.exe", "bin/hipcc.exe", "bin/hipblas.dll", "lib/hipblas.lib"):
            (sdk / f).parent.mkdir(parents=True, exist_ok=True)
            (sdk / f).touch()
        return sdk

    def test_hip_sdk_found_by_hip_path_and_program_files(self):
        sdk = self.hip_sdk()
        old = self.root / "Program Files/AMD/ROCm/6.4/bin"
        old.mkdir(parents=True)
        (old / "clang++.exe").touch()
        clean = {k: v for k, v in os.environ.items() if k not in ("ROCM_PATH", "ROCM_VENV", "HIP_PATH")}
        with patch.object(maya, "rocm_sdk", return_value=None):
            with patch.dict(os.environ, {**clean, "HIP_PATH": str(sdk) + os.sep}, clear=True):
                self.assertEqual(maya.windows_rocm(), (sdk.resolve(), sdk.resolve() / "bin"))
            with patch.dict(os.environ, {**clean, "ProgramFiles": str(self.root / "Program Files")}, clear=True):
                self.assertEqual(maya.windows_rocm(), (sdk.resolve(), sdk.resolve() / "bin"))   # the newest
            with patch.dict(os.environ, {**clean, "ProgramFiles": str(self.root / "none")}, clear=True):
                self.assertEqual(maya.windows_rocm(), (None, None))
        self.assertEqual(maya.hip_clang(sdk), sdk / "bin/clang++.exe")

    def test_hip_sdk_build_in_a_folder_with_a_space(self):
        sdk = self.hip_sdk()
        with patch.dict(os.environ, {"ROCM_PATH": str(sdk)}):
            pc = maya.check_pc(self.a)
        maya.BUILD.mkdir()
        with patch.object(maya, "pick_cmake", return_value="cmake"), \
                patch.object(maya, "venv_tool", return_value="C:/maya/.venv/Scripts/ninja.exe"), \
                patch.object(maya, "cmake_steps", return_value=None) as build:
            maya.compile_engine_hip(pc, self.root / "llama", "source-sha")
        conf, _, env, _ = build.call_args.args
        root = sdk.resolve()
        self.assertIn(f"-DCMAKE_HIP_COMPILER={(root / 'bin/clang++.exe').as_posix()}", conf)
        self.assertIn(f"-DCMAKE_C_COMPILER={(root / 'bin/clang.exe').as_posix()}", conf)
        self.assertFalse([x for x in conf if x.startswith("-DCMAKE_HIP_FLAGS=")])   # "Program Files": the env instead
        self.assertEqual(env["HIP_PATH"], str(root))
        self.assertEqual(env["HIP_DEVICE_LIB_PATH"], str(root / "amdgcn/bitcode"))

    def test_windows_apu_config(self):
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        (tables / "gfx1151-glm-hipblaslt-100202.txt").write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1151 100202\n")
        with patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}):
            pc = maya.check_pc(self.a)
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "bin")]}, self.root / "pack", "Maya-L", 32768,
                              self.root / "data", None)
        cfg = json.loads(p.read_text())
        self.assertEqual(p.name, "maya-maya-l-hip.json")
        self.assertEqual((cfg["backend"], cfg["gpu"], cfg["exe"]), ("hip", [0], str(self.root / "build-hip/strata.exe")))
        self.assertEqual(cfg["lib_dirs"], [str(self.rocm / "bin")])
        env = cfg["env"]
        self.assertEqual(env["STRATA_GLM_RESERVE_MB"], "3072")    # the desktop runs on the APU's GPU too
        self.assertEqual(env["STRATA_GLM_PREFILL_MB"], "6144")    # 32 GB Windows + 96 GB the GPU's
        self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "1024")
        self.assertEqual(env["STRATA_GLM_SPLIT"], "0")
        self.assertNotIn("STRATA_GLM_POOL_GB", env)


if __name__ == "__main__":
    unittest.main()
