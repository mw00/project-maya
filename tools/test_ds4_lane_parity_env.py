"""Compile the actual lane-parity host functions without CUDA or a model.

Each check starts a new process: the environment is cached on first use.
Run with: python -m unittest tools.test_ds4_lane_parity_env -v
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PARITY_NAMES = ("MAYA_DS4_LANE_PARITY", "STRATA_DS4_LANE_PARITY", "DS4_LANE_PARITY")


class LaneParityEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("needs a C++ compiler")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        directory = Path(cls.tmp.name)
        source = (ROOT / "src/ds4/ds4_gguf.cu").read_text(encoding="utf-8")
        # Compile the production definitions, including their caches, override and
        # HIP guard. No copied implementation that could pass while CUDA regresses.
        start = source.index("bool iq_q8k() {")
        end = source.index("\nvoid iq_act_quant(", start)
        harness = directory / "parity.cpp"
        harness.write_text(
            '#include <atomic>\n#include <cstdio>\n#include <cstdlib>\n'
            '#include "strata/platform/env_alias.hpp"\n'
            + source[start:end] + '\n'
            'int main(int argc, char** argv) {\n'
            '    if (argc > 1) strata::platform::env_aliases();\n'
            '    if (argc > 2) g_lane_par.store(std::atoi(argv[2]));\n'
            '    std::printf("%d\\n", lane_parity_on() ? 1 : 0);\n'
            '    return 0;\n}\n', encoding="utf-8")
        cls.binaries = {}
        for backend, flags in (("cuda-host", []), ("hip-host", ["-DSTRATA_USE_HIP=1"])):
            binary = directory / backend
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unused-parameter", *flags, "-I", str(ROOT / "include"),
                            str(harness), "-o", str(binary)], check=True, capture_output=True, text=True)
            cls.binaries[backend] = binary

    def check(self, settings, expected, *, aliases=False, override=None, backend="cuda-host"):
        env = {k: v for k, v in os.environ.items() if k not in (*PARITY_NAMES, "DS4_IQ_Q8K")}
        env.update(settings)
        args = [str(self.binaries[backend])]
        if aliases:
            args.append("aliases")
        if override is not None:
            self.assertTrue(aliases)
            args.append(str(override))
        result = subprocess.run(args, env=env, check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.strip(), str(expected), (settings, aliases, override, backend))

    def test_direct_lookup(self):
        self.check({}, 0)
        for name in PARITY_NAMES:
            for value, expected in (("1", 1), ("0", 0), ("", 0), ("invalid", 0), ("-1", 1)):
                with self.subTest(name=name, value=value):
                    self.check({name: value}, expected)

    def test_precedence_with_and_without_startup_aliases(self):
        maya, strata, short = PARITY_NAMES
        cases = [({maya: "0", strata: "1", short: "1"}, 0),
                 ({maya: "1", strata: "0", short: "0"}, 1),
                 ({maya: "", strata: "1", short: "1"}, 0),
                 ({strata: "0", short: "1"}, 0),
                 ({strata: "1", short: "0"}, 1)]
        for settings, expected in cases:
            for aliases in (False, True):
                with self.subTest(settings=settings, aliases=aliases):
                    self.check(settings, expected, aliases=aliases)
        self.check({maya: "1"}, 1, aliases=True)

    def test_q8k_gate(self):
        for name in PARITY_NAMES:
            self.check({name: "1", "DS4_IQ_Q8K": "0"}, 0)
        self.check({PARITY_NAMES[0]: "1", "DS4_IQ_Q8K": "1"}, 1)

    def test_override(self):
        self.check({}, 1, aliases=True, override=1)
        self.check({PARITY_NAMES[0]: "1"}, 0, aliases=True, override=0)
        self.check({PARITY_NAMES[0]: "1"}, 1, aliases=True, override=-1)
        self.check({"DS4_IQ_Q8K": "0"}, 0, aliases=True, override=1)

    def test_hip_stays_disabled(self):
        for name in PARITY_NAMES:
            self.check({name: "1"}, 0, backend="hip-host")
        self.check({PARITY_NAMES[0]: "1"}, 0, aliases=True, override=1, backend="hip-host")


if __name__ == "__main__":
    unittest.main()
