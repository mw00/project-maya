"""serve/mixed_gpus.py: the two-GPU mode's config switch (on by itself where found), its arguments, and its fallback to one GPU."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mixed_gpus as MG  # noqa: E402

CFG = {"exe": "/x/build-hip/maya-ds4", "backend": "hip", "args": ["--pack", "/p"]}


class Base:
    """A stand-in for StrataEngine: records the command it was started with."""

    def __init__(self, exe, args, cwd=None, log=None, env=None):
        self.spawn = (exe, list(args), cwd, log, env)
        self.started = self.spawn

    def alive(self):
        return True


class TestMixedGpus(unittest.TestCase):
    def test_auto_where_found(self):
        real = (MG.nvidia_listed, MG.windows, MG.Path.is_file)
        try:
            MG.Path.is_file = lambda self: True
            MG.windows = lambda: False
            MG.nvidia_listed = lambda: False   # (no NVIDIA card: as before)
            self.assertIsNone(MG.config_of(dict(CFG)))
            MG.nvidia_listed = lambda: True    # (a card and a CUDA build: on by itself where AUTO)
            auto, MG.AUTO = MG.AUTO, False
            self.assertIsNone(MG.config_of(dict(CFG)))   # (AUTO False: only said)
            MG.AUTO = True
            self.assertEqual(MG.config_of(dict(CFG))["layers"], "auto")
            self.assertIsNone(MG.config_of(dict(CFG, mixed_gpus=False)))   # (switched off)
            self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=6))["layers"], 6)
            self.assertIsNone(MG.config_of(dict(CFG, backend="cuda")))
            MG.windows = lambda: True           # (Windows: as before)
            self.assertIsNone(MG.config_of(dict(CFG)))
        finally:
            MG.nvidia_listed, MG.windows, MG.Path.is_file = real
            MG.AUTO = auto
            MG.AUTO = False

    def test_off_when_not_found(self):
        real = MG.nvidia_listed
        try:
            MG.nvidia_listed = lambda: False
            self.assertIsNone(MG.config_of(dict(CFG)))
        finally:
            MG.nvidia_listed = real
        self.assertIsNone(MG.config_of(dict(CFG, mixed_gpus=False)))

    def test_on_and_layers(self):
        self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=True))["layers"], "auto")
        self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=8))["layers"], 8)
        self.assertTrue(MG.config_of(dict(CFG, mixed_gpus=True))["exe"].endswith(os.path.join("build-cuda", "maya-ds4")))

    def test_mode(self):
        self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=True))["mode"], MG.MODE)
        self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=True, mixed_mode="helper"))["mode"], "helper")
        self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=True, mixed_mode="split"))["mode"], "split")
        self.assertEqual(MG.config_of(dict(CFG, mixed_gpus=True, mixed_mode="x"))["mode"], MG.MODE)

    def test_only_hip_deepseek(self):
        self.assertIsNone(MG.config_of(dict(CFG, mixed_gpus=True, backend="cuda")))
        self.assertIsNone(MG.config_of(dict(CFG, mixed_gpus=True, exe="/x/build/maya")))

    def test_flags(self):
        a = ["--pack", "/p", "--ablate", "d.gguf", "--ablate-default", "off", "--threads", "off"]
        self.assertEqual(MG.set_flag(MG.strip_flags(a, MG.ABLATE_FLAGS), "--threads", "on"),
                         ["--pack", "/p", "--threads", "on"])
        self.assertEqual(MG.set_flag(["--prefill-chunk", "2048"], "--prefill-chunk", "4096", keep=True),
                         ["--prefill-chunk", "2048"])

    def test_no_cuda_build_runs_one_gpu(self):
        cls = MG.engine_class(Base, lambda pid: None)
        e = cls("/x/build-hip/maya-ds4", ["--pack", "/p"], mixed={"layers": "auto", "exe": "/nope/maya-ds4", "gpu": 0})
        self.assertEqual(e.started[0], "/x/build-hip/maya-ds4")   # the AMD GPU alone, its own command
        self.assertFalse(e.mixed_on)
        self.assertIn("not on Windows" if os.name == "nt" else "no CUDA build", e.mixed_off)

    def test_parallel_runs_one_gpu(self):
        cls = MG.engine_class(Base, lambda pid: None)
        e = cls("/x/build-hip/maya-ds4", ["--pack", "/p", "--parallel", "2"],
                mixed={"layers": "auto", "exe": sys.executable, "gpu": 0})
        self.assertFalse(e.mixed_on)
        self.assertIn("not on Windows" if os.name == "nt" else "--parallel", e.mixed_off)


if __name__ == "__main__":
    unittest.main()
