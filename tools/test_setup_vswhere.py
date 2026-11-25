"""Visual Studio discovery with an Installer copy of vswhere or a standalone one on PATH. No VS/CUDA required."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup as S


class VswhereDiscovery(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.program_files = self.root / "Program Files (x86)"
        self.default = self.program_files / "Microsoft Visual Studio/Installer/vswhere.exe"
        self.standalone = self.root / "Standalone tools/vswhere.exe"
        self.standalone.parent.mkdir(parents=True)
        self.standalone.touch()
        self.vs2022 = self.root / "Custom install/2022/BuildTools"
        self.vs2026 = self.root / "Custom install/2026/BuildTools"
        for install in (self.vs2022, self.vs2026):
            vcvars = install / "VC/Auxiliary/Build/vcvars64.bat"
            vcvars.parent.mkdir(parents=True)
            vcvars.touch()

    def find(self, *, installs=None, locator=True, release="12.8", **kwargs):
        installs = {"[16.0,18.0)": self.vs2022} if installs is None else installs

        def output(cmd):
            if cmd[1:] == ["--version"]:
                return f"Cuda compilation tools, release {release}, V{release}.0" if release else ""
            version = cmd[cmd.index("-version") + 1]
            return str(installs[version]) + "\n" if version in installs else ""

        with patch.dict(os.environ, {"ProgramFiles(x86)": str(self.program_files)}), \
                patch.object(S.shutil, "which", return_value=str(self.standalone) if locator else None) as which, \
                patch.object(S, "out", side_effect=output) as out:
            result = S.find_vcvars(**kwargs)
        return result, which, out

    def test_missing_installer_copy_uses_path(self):
        result, which, out = self.find()
        self.assertEqual(result, self.vs2022 / "VC/Auxiliary/Build/vcvars64.bat")
        which.assert_called_once_with("vswhere.exe")
        self.assertEqual(out.call_args.args[0], [
            str(self.standalone), "-latest", "-prerelease", "-products", "*", "-version", "[16.0,18.0)",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])

    def test_installer_copy_stays_first(self):
        self.default.parent.mkdir(parents=True)
        self.default.touch()
        result, which, out = self.find()
        self.assertEqual(result, self.vs2022 / "VC/Auxiliary/Build/vcvars64.bat")
        which.assert_not_called()
        self.assertEqual(out.call_args.args[0][0], str(self.default))

    def test_no_vswhere_returns_none(self):
        result, which, out = self.find(locator=False)
        self.assertIsNone(result)
        which.assert_called_once_with("vswhere.exe")
        out.assert_not_called()

    def test_installer_directory_is_not_an_executable(self):
        self.default.mkdir(parents=True)
        result, _, out = self.find()
        self.assertEqual(result, self.vs2022 / "VC/Auxiliary/Build/vcvars64.bat")
        self.assertEqual(out.call_args.args[0][0], str(self.standalone))

    def test_path_lookup_preserves_cuda_version_limits(self):
        for release in (None, "12.8", "13.0", "13.1", "13.2"):
            with self.subTest(release=release):
                result, _, out = self.find(installs={"[18.0,)": self.vs2026}, release=release,
                                           nvcc="nvcc.exe")
                expected = self.vs2026 / "VC/Auxiliary/Build/vcvars64.bat" if release == "13.2" else None
                self.assertEqual(result, expected)
                versions = [call.args[0][call.args[0].index("-version") + 1]
                            for call in out.call_args_list if "-version" in call.args[0]]
                self.assertEqual(versions, ["[16.0,18.0)", "[18.0,)"] if release == "13.2" else ["[16.0,18.0)"])

    def test_unknown_cuda_keeps_2026_excluded(self):
        result, _, out = self.find(installs={"[18.0,)": self.vs2026})
        self.assertIsNone(result)
        self.assertEqual(out.call_count, 1)

    def test_older_vs_stays_preferred_with_new_cuda(self):
        result, _, out = self.find(installs={"[16.0,18.0)": self.vs2022, "[18.0,)": self.vs2026},
                                   nvcc="nvcc.exe", release="13.2")
        self.assertEqual(result, self.vs2022 / "VC/Auxiliary/Build/vcvars64.bat")
        self.assertEqual(len([call for call in out.call_args_list if "-version" in call.args[0]]), 1)

    def test_hip_can_find_vs2026_on_path(self):
        result, _, out = self.find(installs={"[18.0,)": self.vs2026}, cuda=False)
        self.assertEqual(result, self.vs2026 / "VC/Auxiliary/Build/vcvars64.bat")
        self.assertFalse(any("--version" in call.args[0] for call in out.call_args_list))

    def test_no_compatible_cpp_install_returns_none(self):
        result, _, _ = self.find(installs={})
        self.assertIsNone(result)

    def test_missing_vcvars_is_not_accepted(self):
        (self.vs2022 / "VC/Auxiliary/Build/vcvars64.bat").unlink()
        result, _, _ = self.find()
        self.assertIsNone(result)


class ToolsetChoice(unittest.TestCase):
    """#992: CUDA 13.3 with MSVC 14.52 (VS 2026, Build Tools 18.12.0) crashes nvcc's cudafe++; an older toolset is asked
    for where one is installed.  A fake Visual Studio folder and a fake `nvcc --version`; nothing is compiled."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.install = Path(temp.name) / "18/BuildTools"
        self.vcvars = self.install / "VC/Auxiliary/Build/vcvars64.bat"
        self.vcvars.parent.mkdir(parents=True)
        self.vcvars.touch()

    def toolsets(self, *names):
        for n in names:
            (self.install / "VC/Tools/MSVC" / n).mkdir(parents=True)

    def pick(self, release):
        with patch.object(S, "out", return_value=f"Cuda compilation tools, release {release}, V{release}.0"):
            return S.vcvars_args(self.vcvars, "nvcc.exe")

    def test_lists_the_installed_toolsets(self):
        self.toolsets("14.52.36231", "14.44.35207", "14.50.35717")
        (self.install / "VC/Tools/MSVC/stray.txt").touch()
        self.assertEqual(S.msvc_toolsets(self.vcvars), [(14, 44), (14, 50), (14, 52)])
        self.assertEqual(S.msvc_toolsets(self.install / "nothing/Build/vcvars64.bat"), [])

    def test_newest_good_toolset_where_the_newest_is_the_bad_pair(self):
        self.toolsets("14.52.36231", "14.44.35207", "14.50.35717")
        args, note = self.pick("13.3")
        self.assertEqual(args, " -vcvars_ver=14.50")
        self.assertIn("MSVC 14.52", note)
        self.assertIn("14.50", note)
        (self.install / "VC/Tools/MSVC/14.50.35717").rmdir()
        self.assertEqual(self.pick("13.3")[0], " -vcvars_ver=14.44")
        self.assertEqual(self.pick("13.4")[0], " -vcvars_ver=14.44")      # (newer CUDA: the same pair holds)

    def test_only_the_bad_toolset_installed_says_the_fix(self):
        self.toolsets("14.52.36231")
        args, note = self.pick("13.3")
        self.assertEqual(args, "")                          # (the default stays: nothing else to ask for)
        self.assertIn("14.44", note)
        self.assertIn("MSVC v143", note)
        self.assertIn("no older toolset", note)

    def test_everything_else_is_left_alone(self):
        self.toolsets("14.52.36231", "14.44.35207")
        for release in ("12.8", "13.1", "13.2"):            # (an older CUDA: no change)
            self.assertEqual(self.pick(release), ("", None), release)
        with patch.object(S, "out", return_value=""):       # (no readable nvcc)
            self.assertEqual(S.vcvars_args(self.vcvars, "nvcc.exe"), ("", None))
        self.assertEqual(S.vcvars_args(self.vcvars, None), ("", None))
        (self.install / "VC/Tools/MSVC/14.52.36231").rmdir()   # (the newest is not the bad one: nothing to avoid)
        self.assertEqual(self.pick("13.3"), ("", None))


if __name__ == "__main__":
    unittest.main()
