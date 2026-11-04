"""The Windows build's .bat runs Visual Studio's vcvars64.bat with PATH's entries unquoted (issue #251: a quoted
"C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common" in PATH stopped vcvars64.bat with '\\NVIDIA was
unexpected at this time.' before CMake ran).  No GPU or toolkit needed; the last test runs vcvars64.bat itself where
Visual Studio is installed (Windows)."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya

PHYSX = r"C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common"


class PathUnquoted(unittest.TestCase):
    def test_quotes_taken_out(self):
        self.assertEqual(maya.path_unquoted(rf'C:\a;"{PHYSX}";C:\b'), rf"C:\a;{PHYSX};C:\b")
        self.assertEqual(maya.path_unquoted(rf'"{PHYSX}"'), PHYSX)
        self.assertEqual(maya.path_unquoted(r'C:\a;C:\"half quoted"\b'), r"C:\a;C:\half quoted\b")

    def test_unquoted_path_unchanged(self):
        p = r"C:\Windows\system32;C:\Program Files (x86)\x;;C:\y"
        self.assertEqual(maya.path_unquoted(p), p)

    def test_an_entry_quoted_for_its_semicolon_left_out(self):
        self.assertEqual(maya.path_unquoted(r'C:\a;"C:\x;y";C:\b'), r"C:\a;C:\b")

    def test_vs_env(self):
        env = maya.vs_env({"Path": rf'"{PHYSX}";C:\b', "OTHER": '"kept"'})
        self.assertEqual(env, {"Path": rf"{PHYSX};C:\b", "OTHER": '"kept"'})
        src = {"PATH": r"C:\a"}
        self.assertEqual(maya.vs_env(src), src)


class CmakeStepsEnv(unittest.TestCase):
    """cmake_steps on Windows hands its .bat the environment without PATH's quotes (the rest as given)."""

    def test_bat_gets_unquoted_path(self):
        seen = {}

        def fake_run(cmd, env=None, check=True, echo=True, **kw):
            seen["cmd"], seen["env"] = cmd, env
            return subprocess.CompletedProcess(cmd, 0)
        with tempfile.TemporaryDirectory() as t, \
                patch.object(maya, "WIN", True), patch.object(maya, "ROOT", Path(t)), \
                patch.object(maya.S, "find_vcvars", return_value=Path(r"C:\VS\vcvars64.bat")), \
                patch.object(maya, "run", side_effect=fake_run), patch.object(maya, "say"):
            env = {"PATH": rf'"{PHYSX}";C:\cuda\bin', "CUDA_VISIBLE_DEVICES": "1,0"}
            self.assertIsNone(maya.cmake_steps(["cmake", "-S", "x"], ["cmake", "--build", "b"], env, "build-maya.bat"))
            bat = (Path(t) / "build-maya.bat").read_text(encoding="utf-8")
        self.assertIn(r'call "C:\VS\vcvars64.bat"', bat)
        self.assertEqual(seen["cmd"][:2], ["cmd", "/c"])
        self.assertEqual(seen["env"]["PATH"], rf"{PHYSX};C:\cuda\bin")
        self.assertEqual(seen["env"]["CUDA_VISIBLE_DEVICES"], "1,0")
        self.assertEqual(env["PATH"], rf'"{PHYSX}";C:\cuda\bin')      # (the caller's own dict is not changed)


@unittest.skipUnless(os.name == "nt", "Windows only")
class VcvarsWithQuotedPath(unittest.TestCase):
    """The real vcvars64.bat: a quoted PATH entry with a parenthesis stops it; the unquoted environment runs it."""

    def test_vcvars_runs(self):
        vcvars = maya.S.find_vcvars()
        if vcvars is None:
            self.skipTest("no Visual Studio C++ Build Tools")
        env = dict(os.environ)
        key = next(k for k in env if k.upper() == "PATH")
        env[key] = f'"{PHYSX}";' + env[key]
        with tempfile.TemporaryDirectory() as t:
            bat = Path(t) / "probe.bat"
            bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\nwhere cl >nul || exit /b 7\r\n', encoding="utf-8")
            # (with `env` itself Visual Studio 2022's vcvars64.bat stops: '\NVIDIA was unexpected at this time.')
            r = subprocess.run(["cmd", "/c", str(bat)], env=maya.vs_env(env), capture_output=True, text=True,
                               timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
