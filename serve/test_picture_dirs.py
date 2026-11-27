"""#1180: the picture folders in the temp directory (maya-vision-*, maya-<model>-pictures-*) go with the server, and the
ones an earlier run left behind (kill -9) are cleaned at the next start - only ours, only old, only when their server is
gone."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as SV  # noqa: E402

DAY = 86400.0


def age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


class PictureDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make(self, name: str, old: bool, pid=None) -> Path:
        d = self.root / name
        d.mkdir()
        (d / "pic.rgb").write_bytes(b"x" * 10)
        if pid is not None:
            (d / SV.PICTURE_DIR_PID).write_text(str(pid))
        if old:
            age(d, 2 * DAY)
        return d

    def test_the_stale_folders_of_ours_go_and_nothing_else(self):
        stale = [self.make(f"{p}abc", True) for p in SV.PICTURE_DIR_PREFIXES]
        young = self.make("maya-vision-young", False)
        foreign = self.make("other-pictures-old", True)
        plain = self.root / "maya-vision-file"
        plain.write_text("a file, not a folder")
        age(plain, 2 * DAY)
        self.assertEqual(SV.clean_stale_picture_dirs(temp=str(self.root)), len(stale))
        for d in stale:
            self.assertFalse(d.exists(), d)
        for d in (young, foreign, plain):
            self.assertTrue(d.exists(), d)

    def test_a_folder_of_a_running_server_stays(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("psutil")
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(p.wait)
        self.addCleanup(p.kill)
        alive = self.make("maya-vision-alive", True, pid=p.pid)
        dead = self.make("maya-vision-dead", True, pid=2 ** 22 + 12345)
        SV.clean_stale_picture_dirs(temp=str(self.root))
        self.assertTrue(alive.exists())
        self.assertFalse(dead.exists())

    def test_close_removes_the_encoders_folder(self):
        v = SV.Vision({"exe": "none", "mmproj": "none"}, start=False)
        self.assertTrue(v.dir.is_dir())
        self.assertEqual((v.dir / SV.PICTURE_DIR_PID).read_text(), str(os.getpid()))
        (v.dir / "x.bin").write_bytes(b"1")
        v.close()
        self.assertFalse(v.dir.exists())

    def test_a_server_that_ends_leaves_no_folder(self):
        code = ("import sys; sys.path.insert(0, %r); from serve import server as SV; "
                "print(SV.picture_dir('maya-ds4-pictures-'), flush=True)" % str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertFalse(Path(out.stdout.strip()).exists())


if __name__ == "__main__":
    unittest.main()
