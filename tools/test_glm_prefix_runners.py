"""Runner failures after successful compilation must not make CI pass."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parent


def load_runner(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner: {name}")
    module = importlib.util.module_from_spec(spec)
    # The slot runner reuses its sibling legacy fixture, also under tools discover.
    with mock.patch.object(sys, "path", [str(TOOLS), *sys.path]):
        spec.loader.exec_module(module)
    return module


class PrefixRunnerFailures(unittest.TestCase):
    def check_execution_failure(self, name: str, error: Exception) -> None:
        runner = load_runner(name)
        compiled = subprocess.CompletedProcess(["fixture-compiler"], 0, "", "")
        with tempfile.TemporaryDirectory(prefix="maya-prefix-runner-test-") as temp:
            receipt = Path(temp) / "receipt.json"
            argv = [name, "--receipt", str(receipt)]
            if name == "test_glm_prefix_slots":
                argv += ["--cases", "completion-off"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runner.shutil, "which", return_value="fixture-compiler"), \
                    mock.patch.object(runner.subprocess, "run", side_effect=[compiled, error]) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = runner.main()
            recorded = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 2, "must reach execution after compilation")
            self.assertEqual(recorded["compile"]["exit_code"], 0)
            self.assertEqual(recorded["status"], "error")
            self.assertIn(type(error).__name__, recorded["exception"])
            self.assertNotEqual(result, 0, "execution errors must fail CI")

    def test_helper_timeout_after_compilation_fails(self) -> None:
        self.check_execution_failure("test_glm_prefix_cache_host", subprocess.TimeoutExpired("fixture", 30))

    def test_helper_runtime_error_after_compilation_fails(self) -> None:
        self.check_execution_failure("test_glm_prefix_cache_host", RuntimeError("fixture execution failed"))

    def test_slots_timeout_after_compilation_fails(self) -> None:
        self.check_execution_failure("test_glm_prefix_slots", subprocess.TimeoutExpired("fixture", 15))

    def test_slots_runtime_error_after_compilation_fails(self) -> None:
        self.check_execution_failure("test_glm_prefix_slots", RuntimeError("fixture execution failed"))


if __name__ == "__main__":
    unittest.main()
