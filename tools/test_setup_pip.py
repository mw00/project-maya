"""setup.py's package installs (#753): uv only on Linux ppc64le (IBM POWER9); every other OS / CPU installs with pip
exactly as before, even when uv is on PATH."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import setup as S  # noqa: E402

PKGS = ["numpy", "jinja2"]
PIP = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *PKGS]   # (v1.1.52's)


def uv_on_path(name):
    return "/usr/local/bin/uv" if name == "uv" else None


class UvOnlyOnPpc64le(unittest.TestCase):
    def test_pip_everywhere_else_even_with_uv_on_path(self):
        for machine, system in [("x86_64", "linux"), ("AMD64", "win32"), ("x86_64", "win32"), ("aarch64", "linux"),
                                ("arm64", "darwin"), ("ARM64", "win32"), ("ppc64", "linux"), ("ppc64le", "win32"),
                                ("ppc64le", "darwin"), ("riscv64", "linux"), ("", "linux")]:
            with self.subTest(machine=machine, system=system):
                self.assertEqual(S.pip_command(PKGS, machine, system, which=uv_on_path), PIP)

    def test_uv_on_linux_ppc64le(self):
        self.assertEqual(S.pip_command(PKGS, "ppc64le", "linux", which=uv_on_path),
                         ["/usr/local/bin/uv", "pip", "install", "--python", sys.executable, "--quiet", *PKGS])

    def test_pip_on_ppc64le_without_uv(self):
        self.assertEqual(S.pip_command(PKGS, "ppc64le", "linux", which=lambda name: None), PIP)

    def test_this_pc_installs_with_pip_with_uv_on_path(self):
        """pip_install itself, on the machine the tests run on (not ppc64le): uv on PATH changes nothing."""
        if S.platform.machine().lower() == "ppc64le" and sys.platform.startswith("linux"):
            self.skipTest("this PC is Linux ppc64le: uv is meant to be used here")
        ran = []
        with tempfile.TemporaryDirectory() as d, patch.object(S.sys, "prefix", d), \
                patch.object(S.shutil, "which", uv_on_path), patch.object(S, "run", lambda cmd, **kw: ran.append(cmd)), \
                patch.object(S, "say", lambda *a, **k: None), patch.object(S, "ok", lambda *a, **k: None):
            S.pip_install(PKGS, "test packages")
        self.assertEqual(ran, [PIP])


if __name__ == "__main__":
    unittest.main()
