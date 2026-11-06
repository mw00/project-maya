"""CPU-only checkpoint lifecycle and parallel admission tests."""
import threading
import time
import unittest
import json
import io
import sys
import tempfile
from pathlib import Path
import urllib.request
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

from serve import deep_sleep as d
from serve import server as server_module
from serve.server import Service, StrataEngine, FairLock, MockEngine, ByteTokenizer, ChatTemplate, ROOT, serve


class Driver:
    def __init__(self):
        self.calls = []
        self.states = {101: d.RUNNING, 102: d.RUNNING}
        self.fail = None
    def state(self, pid): return self.states.get(pid)
    def lock(self, pid, timeout): self.calls.append(('lock', pid))
    def checkpoint(self, pid):
        self.calls.append(('checkpoint', pid))
        if self.fail == ('checkpoint', pid): raise d.CheckpointError('failed')
        self.states[pid] = d.CHECKPOINTED
    def restore(self, pid):
        self.calls.append(('restore', pid))
        self.states[pid] = d.LOCKED
    def unlock(self, pid):
        self.calls.append(('unlock', pid))
        self.states[pid] = d.RUNNING


class CheckpointTests(unittest.TestCase):
    def test_config_is_opt_in_and_validated(self):
        self.assertEqual(d.mode_from_config({}), "off")
        self.assertEqual(d.mode_from_config({"deep_sleep": False}), "off")
        for mode in (True, "auto", "checkpoint"):
            self.assertEqual(d.mode_from_config({"deep_sleep": mode}), "checkpoint")
        with self.assertRaises(ValueError):
            d.mode_from_config({"deep_sleep": "native"})

    def test_sleep_and_wake_are_idempotent(self):
        svc = self.service()
        svc.engine.sleeping = True
        svc.engine.info["asleep"] = {"method": "checkpoint"}
        self.assertEqual(svc.sleep(), {"status": "asleep", "method": "checkpoint"})
        svc.engine.sleeping = False
        self.assertEqual(svc.wake(), {"status": "not asleep"})

    def test_ending_sleeping_engine_kills_tracked_processes(self):
        svc = self.service()
        svc.engine.sleeping, svc.engine.sleep_via = True, "checkpoint"
        svc.engine.sleep_pids = [101, 102]
        svc.engine.sleep_launcher = 103
        svc.engine.proc = unittest.mock.Mock()
        with patch.object(d, "_kill_processes") as kill:
            self.assertTrue(d.end(svc.engine))
        self.assertEqual(kill.call_args_list, [unittest.mock.call([101, 102]), unittest.mock.call([103])])
        svc.engine.proc.kill.assert_not_called()
        self.assertFalse(svc.engine.sleeping)

    def test_checkpoint_rolls_back_all_locked_processes(self):
        drv = Driver(); drv.fail = ('checkpoint', 102)
        with patch.object(d, "_process_start", return_value=42):
            pids = [d.ProcessId(101), d.ProcessId(102)]
            with self.assertRaises(d.CheckpointError):
                d._checkpoint(drv, pids)
        self.assertEqual(drv.calls, [('lock', 101), ('lock', 102), ('checkpoint', 101),
                                     ('checkpoint', 102), ('restore', 101), ('unlock', 101), ('unlock', 102)])
        self.assertEqual(list(drv.states.values()), [d.RUNNING, d.RUNNING])

    def service(self):
        engine = object.__new__(StrataEngine)
        engine.sleeping = False
        engine.info = {}
        engine.alive = lambda: True
        svc = object.__new__(Service)
        svc.engine, svc.vision = engine, None
        svc.fifo = FairLock(4)
        svc.status_lock = threading.Lock()
        svc._engine_start = threading.Lock()
        svc.status, svc.active, svc.reload = {'busy': False, 'queued': 0}, 0, None
        return svc

    def test_four_parallel_slots_exclude_sleep_and_queue(self):
        svc = self.service()
        svc.fifo.acquire(units=1)
        with patch.object(d, 'sleep') as sleep:
            self.assertEqual(svc.sleep()['status'], 'busy')
            sleep.assert_not_called()
        svc.fifo.release(units=1)
        svc.status['queued'] = 1
        self.assertEqual(svc.sleep()['status'], 'busy')

    def test_context_resize_and_restart_keep_opt_in_sleep_mode(self):
        engine = object.__new__(StrataEngine)
        engine.spawn = ("engine", ["--max-context", "1024"], None, None, None)
        engine.deep_sleep = "checkpoint"
        engine.set_context(2048)
        self.assertEqual(engine.spawn, ("engine", ["--max-context", "2048"], None, None, None))
        self.assertEqual(engine.deep_sleep, "checkpoint")
        engine.proc = unittest.mock.Mock(pid=123)
        engine._routes_lock = threading.Lock()
        engine._routes = {}
        engine._rids = iter(range(10))
        engine.info = {}
        engine.max_context = 1024
        with patch("serve.server.wait_gpu_release"), patch.object(StrataEngine, "__init__", return_value=None) as init:
            engine.restart()
        init.assert_called_once_with(*engine.spawn, sleep_mode="checkpoint")

    def test_reused_pid_never_touches_driver_or_receives_signal(self):
        drv = Driver()
        with patch.object(d, "_process_start", return_value=42):
            pids = [d.ProcessId(101), d.ProcessId(102)]
        # The second PID now belongs to a different process. Validate the entire
        # set before operating on the first process, too.
        with patch.object(d, "_process_start", side_effect=lambda pid: 42 if int(pid) == 101 else 99):
            for operation in (d._checkpoint, d._restore):
                with self.assertRaises(d._Stuck):
                    operation(drv, pids)
            with patch.object(d.os, "kill") as kill, patch.object(d.os, "pidfd_open", create=True) as pidfd:
                d._kill_processes([pids[1]])
                kill.assert_not_called()
                pidfd.assert_not_called()
        self.assertEqual(drv.calls, [])

    def test_kill_falls_back_when_pidfd_api_is_unavailable(self):
        with patch.object(d, "_process_start", return_value=42):
            pid = d.ProcessId(101)
            fallback_os = SimpleNamespace(kill=unittest.mock.Mock())
            with patch.object(d, "os", fallback_os):
                d._kill_processes([pid])
            fallback_os.kill.assert_called_once_with(pid, d.signal.SIGKILL)

    def test_external_vision_rejected_before_checkpoint(self):
        svc = self.service(); svc.vision = object()
        with patch.object(d, 'sleep') as sleep:
            self.assertEqual(svc.sleep()['status'], 'unsupported')
            sleep.assert_not_called()

    def test_concurrent_requests_wake_once(self):
        svc = self.service(); svc.engine.sleeping = True
        calls = []
        def wake(engine):
            calls.append(1); time.sleep(.05); engine.sleeping = False
        with patch.object(d, 'wake', side_effect=wake):
            threads = [threading.Thread(target=svc.ensure_engine) for _ in range(4)]
            for t in threads: t.start()
            for t in threads: t.join()
        self.assertEqual(len(calls), 1)

    def test_disabled_sleep_close_uses_the_original_quit_protocol(self):
        engine = object.__new__(StrataEngine)
        engine.deep_sleep = "off"
        engine.can_stop = True
        engine.proc = unittest.mock.Mock()
        engine.proc.poll.return_value = 0
        with patch.object(d, "end") as checkpoint_end:
            engine.close()
        checkpoint_end.assert_not_called()
        self.assertEqual(engine.proc.stdin.write.call_args_list,
                         [unittest.mock.call("STOP\n"), unittest.mock.call("QUIT\n")])
        engine.proc.wait.assert_called_once()
        engine.proc.kill.assert_not_called()

    def test_main_shutdown_only_takes_the_sleep_lock_when_enabled(self):
        for mode in ("off", "checkpoint"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.json"
                config.write_text(json.dumps({"deep_sleep": mode}))
                events = []
                engine = MockEngine(ByteTokenizer(), "ok", max_context=1024)
                engine.close = lambda: events.append("close")
                def fake_serve(svc, **kwargs):
                    svc.fifo = unittest.mock.MagicMock()
                    svc.fifo.__enter__.side_effect = lambda: events.append("lock")
                    svc.fifo.__exit__.side_effect = lambda *args: events.append("unlock")
                    return SimpleNamespace(shutdown=lambda: events.append("http shutdown"))
                args = ["server", "--engine", "mock", "--host", "127.0.0.1", "--config", str(config)]
                with patch.object(sys, "argv", args), patch.object(sys, "stdout", io.StringIO()), \
                     patch.object(server_module, "MockEngine", return_value=engine), \
                     patch.object(server_module, "serve", side_effect=fake_serve), \
                     patch.object(server_module, "stop_signals", side_effect=lambda stop: stop.set()):
                    self.assertEqual(server_module.main(), 0)
                expected = ["http shutdown", "close"] if mode == "off" else ["http shutdown", "lock", "close", "unlock"]
                self.assertEqual(events, expected)


class HttpTests(unittest.TestCase):
    def test_sleep_wake_health_and_busy_http(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1024), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        server = serve(svc, port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        def post(path):
            request = urllib.request.Request(base + path, data=b"{}", method="POST")
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                health = json.load(response)
            self.assertTrue(health["loaded"])
            self.assertFalse(health["asleep"])
            self.assertEqual(post("/sleep")[0], 501)
            self.assertEqual(post("/wake"), (200, {"status": "not asleep"}))
            with patch.object(svc, "sleep", return_value={"status": "busy"}):
                self.assertEqual(post("/sleep?wait=0")[0], 409)
        finally:
            server.shutdown(); server.server_close()


if __name__ == '__main__': unittest.main()
