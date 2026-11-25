"""tools/test_direct_read.py - Self-contained unit tests for O_DIRECT aligned/unaligned reads
and automatic buffered fallback when O_DIRECT is refused (e.g. tmpfs / EINVAL), without needing
a GPU or a 60 GB model file.
"""
from __future__ import annotations

import errno
import mmap
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DIRECT_ALIGN = 4096


def direct_span(off: int, length: int) -> tuple[int, int]:
    """Return (aligned_offset, aligned_length) covering [off, off + length)."""
    a0 = off & ~(DIRECT_ALIGN - 1)
    a1 = (off + length + DIRECT_ALIGN - 1) & ~(DIRECT_ALIGN - 1)
    return a0, a1 - a0


class DirectReaderPy:
    """Reference mirror of ling::sys::DirectGgufReader (O_DIRECT with automatic buffered fallback)."""

    def __init__(self, path: Path, want_direct: bool = True):
        self.path = Path(path)
        self.fd_buf = os.open(self.path, os.O_RDONLY)
        self.fd_dir = -1
        self.direct_refused = False
        if want_direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd_dir = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            except OSError:
                self.fd_dir = -1
                self.direct_refused = True

    @property
    def direct_active(self) -> bool:
        return self.fd_dir >= 0 and not self.direct_refused

    def close(self) -> None:
        if self.fd_dir >= 0:
            os.close(self.fd_dir)
            self.fd_dir = -1
        if self.fd_buf >= 0:
            os.close(self.fd_buf)
            self.fd_buf = -1

    def read_span(self, off: int, length: int) -> bytes:
        if length == 0:
            return b""
        if self.direct_active:
            a0, span = direct_span(off, length)
            try:
                with mmap.mmap(-1, span) as buf:
                    got = os.preadv(self.fd_dir, [buf], a0)
                    if got >= (off - a0) + length:
                        start = off - a0
                        return buf[start : start + length]
            except OSError as exc:
                if exc.errno != errno.EFAULT:
                    self.direct_refused = True
        return os.pread(self.fd_buf, length, off)


HAVE_PREAD = hasattr(os, "pread")
HAVE_PREADV = hasattr(os, "preadv")


class TestDirectRead(unittest.TestCase):
    def setUp(self):
        self.payload = bytes((i * 131 + 17) & 0xFF for i in range(1 << 20))  # 1 MiB

    def test_direct_span_alignment(self):
        self.assertEqual(direct_span(0, 4096), (0, 4096))
        self.assertEqual(direct_span(13, 5000), (0, 8192))
        self.assertEqual(direct_span(4095, 131073), (0, 135168))
        self.assertEqual(direct_span(8192, 1), (8192, 4096))

    @unittest.skipUnless(HAVE_PREAD and HAVE_PREADV, "needs os.pread and os.preadv (not on Windows)")
    def test_direct_and_buffered_exact_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "sample.bin"
            f.write_bytes(self.payload)
            for want_direct in (True, False):
                rdr = DirectReaderPy(f, want_direct=want_direct)
                try:
                    if not want_direct:
                        self.assertFalse(rdr.direct_active)
                    for off, n in ((0, 4096), (13, 5000), (4095, 131073), (500003, 300000), (len(self.payload) - 19, 19)):
                        self.assertEqual(rdr.read_span(off, n), self.payload[off : off + n])
                finally:
                    rdr.close()

    @unittest.skipUnless(HAVE_PREAD, "needs os.pread (not on Windows)")
    def test_einval_refusal_falls_back_to_buffered(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "sample.bin"
            f.write_bytes(self.payload)
            real_open = os.open

            def fake_open(path, flags, *a, **kw):
                if hasattr(os, "O_DIRECT") and (flags & os.O_DIRECT):
                    raise OSError(errno.EINVAL, "Invalid argument")
                return real_open(path, flags, *a, **kw)

            with mock.patch("os.open", side_effect=fake_open):
                rdr = DirectReaderPy(f, want_direct=True)
                try:
                    self.assertFalse(rdr.direct_active)
                    self.assertEqual(rdr.read_span(12345, 77777), self.payload[12345 : 12345 + 77777])
                finally:
                    rdr.close()

    @unittest.skipUnless(HAVE_PREAD and HAVE_PREADV, "needs os.pread and os.preadv (not on Windows)")
    def test_preadv_refusal_falls_back_to_buffered(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "sample.bin"
            f.write_bytes(self.payload)
            real_open = os.open

            def fake_open(path, flags, *a, **kw):
                if hasattr(os, "O_DIRECT") and (flags & os.O_DIRECT):
                    return real_open(path, flags & ~os.O_DIRECT, *a, **kw)
                return real_open(path, flags, *a, **kw)

            with mock.patch("os.open", side_effect=fake_open), mock.patch(
                "os.preadv", side_effect=OSError(errno.EINVAL, "Invalid argument")
            ) as mock_preadv:
                rdr = DirectReaderPy(f, want_direct=True)
                try:
                    if hasattr(os, "O_DIRECT"):
                        self.assertTrue(rdr.direct_active)
                    self.assertEqual(rdr.read_span(4095, 65537), self.payload[4095 : 4095 + 65537])
                    self.assertFalse(rdr.direct_active)
                    calls_after_first = mock_preadv.call_count
                    self.assertEqual(rdr.read_span(500003, 300000), self.payload[500003 : 500003 + 300000])
                    self.assertEqual(mock_preadv.call_count, calls_after_first)
                finally:
                    rdr.close()

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/dev/shm").is_dir(), "Linux /dev/shm tmpfs check")
    def test_tmpfs_odirect_or_fallback(self):
        f = Path("/dev/shm") / f"maya_ling_odirect_ut_{os.getpid()}.bin"
        spans = ((0, 4096), (13, 5000), (4095, 65537), (500003, 300000), (len(self.payload) - 19, 19))
        try:
            f.write_bytes(self.payload)
            # Linux >= 6.8 supports O_DIRECT on tmpfs while older kernels refuse it with EINVAL;
            # accept either mode and verify exact byte-for-byte reads in both direct and buffered modes.
            rdr_dir = DirectReaderPy(f, want_direct=True)
            try:
                self.assertIsInstance(rdr_dir.direct_active, bool)
                for off, n in spans:
                    self.assertEqual(rdr_dir.read_span(off, n), self.payload[off : off + n])
            finally:
                rdr_dir.close()

            rdr_buf = DirectReaderPy(f, want_direct=False)
            try:
                self.assertFalse(rdr_buf.direct_active)
                for off, n in spans:
                    self.assertEqual(rdr_buf.read_span(off, n), self.payload[off : off + n])
            finally:
                rdr_buf.close()
        finally:
            if f.exists():
                f.unlink()

    def test_ling_sys_direct_reader_contract_present(self):
        src = (ROOT / "src" / "ling" / "ling_sys.hpp").read_text(encoding="utf-8")
        self.assertIn("class DirectGgufReader", src)
        self.assertIn("open_direct", src)
        self.assertIn("direct_is_refused", src)
        self.assertIn("pread_full", src)


if __name__ == "__main__":
    unittest.main()
