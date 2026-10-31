"""The server tee and an inherited engine stderr must share an append-only log (no GPU)."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from serve import server


class SharedRunLog(unittest.TestCase):
    def test_utf8_line_buffering_and_reopen_preserve_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run-\u00e9.log"
            with server.open_run_log(path, buffering=1) as log:
                self.assertFalse(os.get_inheritable(log.fileno()))
                log.write("server caf\u00e9\n")
                self.assertEqual(path.read_text(encoding="utf-8"), "server caf\u00e9\n")
            with server.open_run_log(path) as log:
                log.write("next start\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "server caf\u00e9\nnext start\n")

    def test_missing_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                server.open_run_log(Path(tmp) / "missing" / "run.log")

    @unittest.skipUnless(os.name == "nt", "Windows native handle semantics")
    def test_tee_native_handle_cannot_overwrite_an_existing_prefix(self):
        import ctypes
        import msvcrt
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.WriteFile.argtypes = (wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
        k32.WriteFile.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys, "stdout", io.StringIO()):
            path = Path(tmp) / "run.log"
            server.tee_server_log(str(path))
            with sys.stdout.handle as log:
                log.write("server prefix\n")
                log.flush()
                # The tee needs the same append-only access as the engine. Bypass
                # its CRT append emulation and write through an explicitly stale offset.
                os.lseek(log.fileno(), 0, os.SEEK_SET)
                data = b"native tail\n"
                written = wintypes.DWORD()
                self.assertTrue(k32.WriteFile(msvcrt.get_osfhandle(log.fileno()), data, len(data),
                                              ctypes.byref(written), None))
                self.assertEqual(written.value, len(data))
            self.assertEqual(path.read_text(encoding="utf-8"), "server prefix\nnative tail\n")

    @unittest.skipUnless(os.name == "nt", "Windows native handle ownership")
    def test_descriptor_failure_closes_native_handle(self):
        import ctypes
        from ctypes import wintypes

        handles = []

        def fail(handle, flags):
            handles.append(handle)
            raise OSError("descriptor creation failed")

        with tempfile.TemporaryDirectory() as tmp, patch("msvcrt.open_osfhandle", side_effect=fail):
            with self.assertRaisesRegex(OSError, "descriptor creation failed"):
                server.open_run_log(Path(tmp) / "run.log")
        self.assertEqual(len(handles), 1)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetHandleInformation.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.GetHandleInformation.restype = wintypes.BOOL
        flags = wintypes.DWORD()
        self.assertFalse(k32.GetHandleInformation(handles[0], ctypes.byref(flags)))
        self.assertEqual(ctypes.get_last_error(), 6)  # ERROR_INVALID_HANDLE

    def test_engine_stderr_preserves_server_lines(self):
        # Use a real child and the real StrataEngine/tee log opens. On Windows, Popen
        # inherits the OS handle, not the parent's CRT O_APPEND descriptor flag.
        child = """
import os, sys
os.write(2, b'engine start\\n')
print('READY 4096', flush=True)
for line in sys.stdin:
    if line == 'QUIT\\n':
        break
    os.write(2, line.encode('utf-8'))
    print('ACK', flush=True)
"""
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run-\u00e9.log"
                expected = ["previous run"] if existing else []
                if existing:
                    path.write_bytes(b"previous run\n")
                processes, timers = [], []
                popen = subprocess.Popen

                def start_child(args, **kwargs):
                    proc = popen([sys.executable, "-u", "-c", child], **kwargs)
                    processes.append(proc)
                    # The constructor waits for READY; a broken child must not hang CI.
                    timer = threading.Timer(10, proc.kill)
                    timer.daemon = True
                    timer.start()
                    timers.append(timer)
                    return proc

                engine = tee = None
                terminal = io.StringIO()
                try:
                    with patch.object(sys, "stdout", terminal):
                        server.tee_server_log(str(path))
                        tee = sys.stdout
                        with patch.object(server.subprocess, "Popen", side_effect=start_child), \
                                patch.object(server, "narrate_start"):
                            engine = server.StrataEngine("unused-engine", [], log=str(path))
                        expected.append("engine start")
                        for i in range(3):
                            # Flush the independent parent handle, then ask the child to
                            # write through its inherited handle at its old file position.
                            print(f"server {i}", flush=True)
                            engine.proc.stdin.write(f"engine {i}\n")
                            engine.proc.stdin.flush()
                            self.assertEqual(engine.lines.get(timeout=5), "ACK\n")
                            expected += [f"server {i}", f"engine {i}"]
                        self.assertEqual(path.read_text(encoding="utf-8").splitlines(), expected)
                        self.assertIn("server 2\n", terminal.getvalue())
                finally:
                    for proc in processes:
                        if proc.poll() is None:
                            proc.kill()
                        proc.wait(timeout=5)
                        proc.stdin.close()
                        proc.stdout.close()
                    for timer in timers:
                        timer.cancel()
                    if engine is not None:
                        engine.log.close()
                    if tee is not None:
                        tee.handle.close()


if __name__ == "__main__":
    unittest.main()
