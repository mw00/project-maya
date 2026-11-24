#!/usr/bin/env python3
"""The setup's compile steps print ONE line each: the compilers' output goes to a log in the install folder, whose
last lines are shown when a step stops (setup.BuildLog, maya.cmake_steps).  MAYA_BUILD_OUTPUT=full: as before."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya  # noqa: E402
import setup as S  # noqa: E402


def py(code: str) -> list:
    return [sys.executable, "-c", code]


NINJA = ("import sys\n"
         "print('-- Configuring done')\n"
         "for i in range(1, 41):\n"
         "    print(f'[{i}/40] Building CXX object llama/tools/mtmd/CMakeFiles/mtmd.dir/models/whisper-enc.cpp.o')\n"
         "    print('warning: unused variable', file=sys.stderr)\n")


class BuildLogLines(unittest.TestCase):
    def run_steps(self, cmds, env=None):
        with tempfile.TemporaryDirectory() as t, patch.object(maya, "ROOT", Path(t)), patch.object(maya, "WIN", False), \
                patch.dict(os.environ, env or {}), redirect_stdout(io.StringIO()) as out:
            os.environ.pop("MAYA_BUILD_OUTPUT", None) if env is None else None
            stopped = maya.cmake_steps(cmds[0], cmds[1], None, "", what="the picture reader", log="build-vision.log")
            log = (Path(t) / "build-vision.log")
            return stopped, out.getvalue(), log.read_text(encoding="utf-8") if log.exists() else None, str(log)

    def test_one_line_and_everything_in_the_log(self):
        stopped, shown, log, path = self.run_steps([py("print('-- The CXX compiler identification is GNU')"), py(NINJA)])
        self.assertIsNone(stopped)
        lines = [x for x in shown.splitlines() if x.strip()]
        self.assertEqual(len(lines), 1, shown)
        self.assertIn("the picture reader: 40 files", lines[0])
        self.assertIn(path, lines[0])
        self.assertNotIn("whisper", shown)
        self.assertNotIn("warning", shown)
        self.assertEqual(sum(x.startswith("[") for x in log.splitlines()), 40)
        self.assertEqual(log.splitlines().count("warning: unused variable"), 40)
        self.assertIn("compiler identification", log)

    def test_a_stop_shows_the_last_lines_and_the_log(self):
        bad = py("import sys\nfor i in range(100): print(f'line {i}')\nprint('error: no such kernel'); sys.exit(1)")
        stopped, shown, log, path = self.run_steps([py("print('configured')"), bad])
        self.assertEqual(stopped, "compiling")
        self.assertIn("error: no such kernel", shown)
        self.assertIn("line 99", shown)
        self.assertNotIn("line 10\n", shown)                       # (the last ~30 lines, not all of them)
        self.assertLessEqual(len(shown.splitlines()), S.BuildLog.TAIL + 4)
        self.assertIn(path, shown)
        self.assertEqual(log.splitlines().count("error: no such kernel"), 2)   # (the build is tried twice, as before)

    def test_configure_stop(self):
        stopped, shown, _, _ = self.run_steps([py("import sys; print('CMake Error: no CUDA'); sys.exit(1)"), py("")])
        self.assertEqual(stopped, "configuring")
        self.assertIn("CMake Error: no CUDA", shown)

    def test_full_output_as_before(self):
        stopped, shown, log, _ = self.run_steps([py("print('configured')"), py("print('[1/1] Building x.o')")],
                                                env={"MAYA_BUILD_OUTPUT": "full"})
        self.assertIsNone(stopped)
        self.assertIsNone(log)
        self.assertIn("  > ", shown)                               # (the commands, echoed as before)


class Progress(unittest.TestCase):
    def test_counts_and_percent(self):
        with tempfile.TemporaryDirectory() as t, patch.object(S, "progress"), patch.object(S, "say"):
            b = S.BuildLog(Path(t) / "x.log", "the engine")
            b.line("[ 12/208] Building CUDA object a.cu.o")
            self.assertEqual((b.done, b.total), (12, 208))
            self.assertIn("the engine: 12 of 208 files (6%)", b.text())
            b.file.close()
            m = S.BuildLog(Path(t) / "y.log", "the engine")
            m.line("[ 37%] Building CXX object b.o")
            self.assertIn("the engine: 37%", m.text())
            m.file.close()


class OnTheSetupScreen(unittest.TestCase):
    """With the setup's screen (setup.UI, a Bridge) a compile step's lines go to the BuildLog, not to the screen's
    output or its summary: the screen gets progress events."""

    def test_the_bridge_hands_the_lines_to_the_log(self):
        import threading
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from setup_bridge import Bridge
        with tempfile.TemporaryDirectory() as t:
            ui = Bridge(Path(t) / "setup.log")
            ui.thread = threading.current_thread()
            with patch.object(S, "UI", ui):
                steps = S.BuildLog(Path(t) / "build-engine.log", "the engine")
                rc = steps.run(py("print('[1/2] Building a.o'); print('[2/2] Building whisper-enc.cpp.o')"))
                steps.close(None)
            ui.log.close()
            kinds = []
            while not ui.events.empty():
                kinds.append(ui.events.get()[0])
            self.assertEqual(rc, 0)
            self.assertEqual((steps.done, steps.total), (2, 2))
            self.assertNotIn("out", kinds)
            self.assertIn("progress", kinds)
            self.assertIn("whisper-enc", (Path(t) / "build-engine.log").read_text(encoding="utf-8"))
            self.assertNotIn("whisper-enc", (Path(t) / "setup.log").read_text(encoding="utf-8"))
            self.assertEqual([k for _, k, _ in ui.entries if k == "out"], [])


if __name__ == "__main__":
    unittest.main()
