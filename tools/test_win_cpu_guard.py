"""Windows' ggml-cpu SIMD level does not depend on CMake test programs that Smart App Control can block: the guard
(cmake/win_cpu_guard.cmake) takes AVX2/FMA/AVX-512 from the CPU itself, maya.py passes what it read and says in one
line when a configure log still shows ggml-cpu below this CPU.  No compiler or GPU needed; the CPUID test runs on
Windows x86 only."""
from pathlib import Path
import platform
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import maya
import setup as S

AVX_ONLY = "-- Adding CPU backend variant ggml-cpu: /arch:AVX GGML_AVX\n"
AVX2 = "-- Adding CPU backend variant ggml-cpu: /arch:AVX2 GGML_AVX2;GGML_FMA;GGML_F16C\n"
AVX512 = "-- Adding CPU backend variant ggml-cpu: /arch:AVX512 GGML_AVX512\n"
BD = Path("build")


class GuardNote(unittest.TestCase):
    def test_avx_only_on_an_avx2_cpu_is_said(self):
        note = maya.cpu_guard_note(AVX_ONLY, ["avx", "avx2", "fma", "f16c"], BD)
        self.assertIn("without AVX2", note)
        self.assertIn("Smart App Control", note)
        self.assertIn(str(BD / "CMakeCache.txt"), note)

    def test_matching_build_is_quiet(self):
        self.assertIsNone(maya.cpu_guard_note(AVX2, ["avx", "avx2", "fma", "f16c"], BD))
        self.assertIsNone(maya.cpu_guard_note(AVX512, ["avx", "avx2", "fma", "f16c", "avx512"], BD))

    def test_cpu_without_avx2_is_quiet(self):
        self.assertIsNone(maya.cpu_guard_note(AVX_ONLY, ["avx"], BD))
        self.assertIsNone(maya.cpu_guard_note(AVX_ONLY, ["avx", "avx2"], BD))       # (no FMA: ggml takes AVX)

    def test_avx512_cpu_built_avx2_is_said(self):
        self.assertIn("without AVX512", maya.cpu_guard_note(AVX2, ["avx", "avx2", "fma", "avx512"], BD))

    def test_last_configure_counts(self):
        self.assertIsNone(maya.cpu_guard_note(AVX_ONLY + AVX2, ["avx", "avx2", "fma"], BD))

    def test_correction_is_said(self):
        log = "-- Maya CPU guard: corrected FMA: an earlier configure's check program did not run\n" + AVX2
        self.assertIn("now built with AVX2", maya.cpu_guard_note(log, ["avx", "avx2", "fma"], BD))

    def test_no_variant_line_is_quiet(self):
        self.assertIsNone(maya.cpu_guard_note("-- Configuring done\n", ["avx", "avx2", "fma"], BD))


class Features(unittest.TestCase):
    def test_script_prints_a_list(self):
        out = subprocess.run([sys.executable, "-I", str(ROOT / "cmake" / "win_cpu_features.py")],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        got = [f for f in out.stdout.strip().split(",") if f]
        self.assertLessEqual(set(got), {"avx", "avx2", "fma", "f16c", "avx512"})
        if sys.platform != "win32":
            self.assertEqual(got, [])

    @unittest.skipUnless(sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64"),
                         "Windows x86 only")
    def test_agrees_with_cpu_info(self):
        feats = S.win_simd_features()
        self.assertIn("avx", feats)
        self.assertEqual("avx2" in feats, S._cpuid_avx2())
        if "avx512" in feats:
            self.assertIn("avx2", feats)

    def test_not_windows_is_empty(self):
        if sys.platform == "win32":
            self.skipTest("not Windows only")
        self.assertEqual(S.win_simd_features(), [])


class CMakeWiring(unittest.TestCase):
    def test_guard_only_for_msvc_x86_native(self):
        txt = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        i = txt.index("include(cmake/win_cpu_guard.cmake)")
        self.assertIn("if(MSVC AND MAYA_CPU_X86)", txt[i - 300:i])
        self.assertLess(txt.index("set(GGML_NATIVE ON CACHE BOOL"), i)
        self.assertLess(i, txt.index("maya_win_cpu_guard_check()"))

    def test_guard_writes_findsimd_cache_names(self):
        g = (ROOT / "cmake" / "win_cpu_guard.cmake").read_text(encoding="utf-8")
        self.assertIn('set(${_t}_FOUND TRUE CACHE BOOL "${_t} support" FORCE)', g)
        for t in ("AVX", "AVX2 FMA", "AVX512"):
            self.assertIn(f"list(APPEND _types {t})", g)


class Injection(unittest.TestCase):
    """cmake_steps on Windows: the CPU's features go into the configure line, and an AVX-only log is said."""

    def steps(self, feats, log_text):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        logp = tmp / "build-engine.log"

        class Log:
            def __init__(self, path, what):
                self.path = logp
                self.file = open(logp, "w", encoding="utf-8")

            def run(self, cmd, env=None):
                self.file.write(log_text)
                return 0

            def line(self, t):
                pass

            def close(self, stopped):
                self.file.close()

        warned = []
        with patch.object(maya, "WIN", True), patch.object(maya, "ROOT", tmp), \
                patch.object(S, "ARM", False), patch.object(S, "win_simd_features", return_value=feats), \
                patch.object(S, "full_build_output", return_value=False), \
                patch.object(S, "find_vcvars", return_value="vcvars64.bat"), \
                patch.object(S, "vcvars_args", return_value=("", None)), \
                patch.object(maya, "vs_env", side_effect=lambda e: e), patch.object(S, "BuildLog", Log), \
                patch.object(maya, "warn", side_effect=warned.append):
            stopped = maya.cmake_steps(["cmake", "-S", "src", "-B", str(tmp / "b")], ["cmake", "--build", "b"],
                                       {}, "x.bat")
        return stopped, (tmp / "x.bat").read_text(encoding="utf-8"), warned

    def test_features_in_the_configure_line(self):
        stopped, bat, warned = self.steps(["avx", "avx2", "fma", "f16c"], AVX2)
        self.assertIsNone(stopped)
        self.assertIn("-DMAYA_WIN_CPU_FEATURES=avx,avx2,fma,f16c", bat)
        self.assertEqual(warned, [])

    def test_avx_only_log_is_said_once(self):
        _, _, warned = self.steps(["avx", "avx2", "fma", "f16c"], AVX_ONLY)
        self.assertEqual(len(warned), 1)
        self.assertIn("without AVX2", warned[0])

    def test_unreadable_cpu_passes_nothing(self):
        _, bat, warned = self.steps([], AVX_ONLY)
        self.assertNotIn("MAYA_WIN_CPU_FEATURES", bat)
        self.assertEqual(warned, [])


if __name__ == "__main__":
    unittest.main()
