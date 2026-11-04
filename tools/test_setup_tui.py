"""tools/test_setup_tui.py - the setup on its own screen (tools/setup_tui.py, setup_bridge.py): the commands' output as
it comes (curl's rewritten meter apart), the questions answered with keys, a failure read on the screen, a stop that
ends the running command, and the plain text setup unchanged without the screen.  No GPU, no network; the screen's
tests need Textual (./maya.sh --setup installs it from third_party/wheels) and are skipped without it."""
from pathlib import Path
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import setup as S  # noqa: E402
from setup_bridge import Bridge  # noqa: E402
try:
    import setup_look as L
    import setup_tui as T
except ImportError:
    L = T = None
needs_textual = unittest.skipIf(T is None, "Textual is not installed (./maya.sh --setup installs it)")


def events(bridge: Bridge) -> list:
    out = []
    while not bridge.events.empty():
        out.append(bridge.events.get())
    return out


class Streaming(unittest.TestCase):
    """Bridge.run: whole lines go to the log and the summary; a line curl rewrites with \\r only to the screen."""

    def run_code(self, code):
        with tempfile.TemporaryDirectory() as d:
            b = Bridge(Path(d) / "setup.log")
            b.thread = threading.current_thread()       # (the caller is the setup's thread here)
            r = b.run([sys.executable, "-c", code])
            b.log.close()
            return r, [e for e in events(b) if e[0] in ("out", "status")], b

    # (the child writes bytes: a text-mode stdout on Windows makes every \n a \r\n, so 'a\r\n' arrived as 'a\r\r\n')
    def test_lines_rewritten_lines_and_the_exit_code(self):
        r, ev, b = self.run_code("import sys; sys.stdout.buffer.write(b'a\\r\\nb\\n 1 x\\r 50 y\\rlast'); sys.exit(3)")
        self.assertEqual(r.returncode, 3)
        self.assertEqual(ev, [("out", "a"), ("out", "b"), ("status", " 1 x"), ("status", " 50 y"), ("out", "last")])
        self.assertEqual([t for _, k, t in b.entries if k == "out"], ["a", "b", "last"])

    def test_a_cr_lf_split_between_two_reads_is_one_line_end(self):
        _, ev, _ = self.run_code("import sys, time; sys.stdout.buffer.write(b'a\\r'); sys.stdout.buffer.flush(); "
                                 "time.sleep(0.3); sys.stdout.buffer.write(b'\\nb\\n')")
        self.assertEqual(ev, [("out", "a"), ("out", "b")])

    def test_stdin_is_closed(self):                 # nothing the setup runs can wait for a key under the screen
        _, ev, _ = self.run_code("import sys; print(repr(sys.stdin.read()))")
        self.assertEqual(ev, [("out", "''")])


class Bridging(unittest.TestCase):
    def test_after_the_screen_other_threads_print_and_the_setup_thread_stops(self):
        with tempfile.TemporaryDirectory() as d:
            b = Bridge(Path(d) / "setup.log")
            b.thread = threading.Thread(target=lambda: None)
            self.assertTrue(b.say("on the screen"))
            b.closed = True
            self.assertFalse(b.say("printed"))         # setup.say prints it
            b.thread, b.closed = threading.current_thread(), False
            b.stop()
            with self.assertRaises(KeyboardInterrupt):
                b.ok("the setup's thread ends at its next line")
            b.log.close()

    def test_a_stop_ends_what_else_the_setup_started(self):  # (the engine the tuning loads, outside Bridge.run)
        if importlib.util.find_spec("psutil") is None:
            self.skipTest("psutil comes with the setup's step 3")
        engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with tempfile.TemporaryDirectory() as d:
            b = Bridge(Path(d) / "setup.log")
            b.stop()
            b.log.close()
        self.assertIsNotNone(engine.wait(10))

    def test_summary_keeps_results_and_the_last_steps_messages_when_it_ended_there(self):
        with tempfile.TemporaryDirectory() as d:
            b = Bridge(Path(d) / "setup.log")
            b.thread = threading.current_thread()
            b.step(1, "checking this PC")
            b.say("  NVIDIA GPUs:")
            b.ok("using GPU 0")
            b.step(5, "the model")
            b.say("  The exact commands ...")
            b._output("curl output", False)
            b.warn("on a spinning disk")
            b.log.close()
            log = (Path(d) / "setup.log").read_text(encoding="utf-8")
        self.assertEqual(b.summary(whole=False), ["", "=== Step 1: checking this PC ===", "  [ok] using GPU 0", "",
                                                  "=== Step 5: the model ===", "  [!]  on a spinning disk"])
        whole = b.summary(whole=True)
        self.assertIn("  The exact commands ...", whole)
        self.assertIn("curl output", whole)
        self.assertNotIn("  NVIDIA GPUs:", whole)       # (an earlier step's messages stay in the log)
        self.assertIn("  NVIDIA GPUs:", log)


class PlainSetup(unittest.TestCase):
    """Without the screen (setup.UI None, or a screen that does not take it) the setup prints as before."""

    def tearDown(self):
        S.UI = None

    def test_choose_prints_the_numbered_list_without_a_screen(self):
        lines = []
        with patch.object(maya, "say", side_effect=lambda *a: lines.append(" ".join(map(str, a)))):
            pick = maya.choose("Context?", ["Context length ..."], [("8K tokens", None, None),
                                                                   ("32K tokens", "recommended", "about")], 1, True)
        self.assertEqual(pick, 1)
        self.assertEqual(lines, ["", "  Context length ...", "  1) 8K tokens", "  2) 32K tokens   (recommended)"])

    def test_choose_asks_the_screen_when_there_is_one(self):
        S.UI = SimpleNamespace(choose=lambda *a: 0)
        with patch.object(maya, "say", side_effect=AssertionError("nothing is printed")):
            self.assertEqual(maya.choose("Context?", [], [("8K", None, None), ("32K", None, None)], 1, False), 0)

    def test_a_closed_screen_falls_back_to_print(self):
        S.UI = SimpleNamespace(say=lambda msg: False, ok=lambda msg: False)
        with patch("sys.stdout", new=io.StringIO()) as out:
            S.ok("printed")
        self.assertEqual(out.getvalue(), "  [ok] printed\n")

    def test_no_screen_for_plain_yes_or_no_terminal(self):
        with patch.object(maya.subprocess, "run", side_effect=AssertionError("no pip")):
            self.assertFalse(maya.setup_screen(SimpleNamespace(plain=True, yes=False)))
            self.assertFalse(maya.setup_screen(SimpleNamespace(plain=False, yes=True)))
            with patch("sys.stdin", new=io.StringIO()):
                self.assertFalse(maya.setup_screen(SimpleNamespace(plain=False, yes=False)))

    def test_the_wheels_hold_textual_and_what_it_needs(self):
        names = {p.name.split("-")[0].lower() for p in maya.TUI_WHEELS.glob("*.whl")}
        self.assertLessEqual({"textual", "rich", "pygments", "markdown_it_py", "mdurl", "platformdirs",
                              "typing_extensions"}, names)
        self.assertTrue(all(p.name.endswith("-py3-none-any.whl") for p in maya.TUI_WHEELS.glob("*.whl")))


@needs_textual
class Screen(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "maya-setup.log"

    def tearDown(self):
        S.UI = None
        self.dir.cleanup()

    async def drive(self, body, *answers, check=None):
        """Runs `body` on the screen; each of `answers` (keys) answers the next question.  -> (end, bridge, app)"""
        bridge = Bridge(self.log)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "Project Maya - setup", "./maya.sh", self.log)
        async with app.run_test(size=(110, 32)) as pilot:
            async def until(cond):
                t0 = time.time()
                while not cond():
                    self.assertLess(time.time() - t0, 20, "the screen did not get there")
                    await pilot.pause(0.02)
            for keys in answers:
                await until(lambda: app.pending is not None)
                if check:
                    check(app)
                await pilot.press(*keys)
            await until(lambda: app.return_value is not None)
        bridge.log.close()
        return app.return_value, bridge, app

    async def test_menus_and_a_yes_no_question(self):
        def body():
            S.step(2, "your choices")
            ctx = maya.choose("Context?", ["Context length ..."],
                              [("8K tokens", None, None), ("32K tokens", "recommended", None),
                               ("64K tokens", None, None)], 1, False)
            model = maya.choose("Model?", [], [("Maya-S", None, "about S"), ("Maya-L", None, "about L")], 0, False)
            return ctx, model, S.ask("  Download them now?", ["y", "n"], "n", False)
        end, bridge, _ = await self.drive(body, ["down", "enter"], ["2"], ["y"])
        self.assertEqual(end, ("done", (2, 1, "y")))
        self.assertEqual([t for _, k, t in bridge.entries if k == "answer"],
                         ["  Context? 3) 64K tokens", "  Model? 2) Maya-L", "  Download them now? y"])

    async def test_enter_takes_the_default_and_a_download_defaults_to_no(self):
        end, _, _ = await self.drive(lambda: S.ask("Download about 95 GB now?", ["y", "n"], "n", False), ["enter"])
        self.assertEqual(end, ("done", "n"))

    async def test_a_failure_is_read_on_the_screen_then_ends_the_setup(self):
        def body():
            S.step(4, "the engine")
            S.fail("the engine build stopped while compiling", "common causes: ...")
            raise AssertionError("not reached")
        end, bridge, app = await self.drive(body, ["enter"])
        self.assertEqual(end, ("fail", ("the engine build stopped while compiling", "common causes: ...")))
        self.assertEqual(app.state[4], "fail")
        self.assertIn("[X]  the engine build stopped while compiling", self.log.read_text(encoding="utf-8"))

    async def test_a_compiles_progress_from_ninjas_lines(self):
        code = ("import time\nfor i in range(1, 4): print(f'[{i}/4] Building CXX object x{i}.o', flush=True)\n"
                "time.sleep(1)")
        bridge = Bridge(self.log)
        S.UI = bridge
        app = T.SetupApp(bridge, lambda: S.run([sys.executable, "-c", code]), "1.0", "./maya.sh", self.log)
        async with app.run_test(size=(110, 32)) as pilot:
            t0 = time.time()
            while app.bar is None or app.bar[0] < 0.75:
                self.assertLess(time.time() - t0, 20)
                await pilot.pause(0.02)
            self.assertEqual(app.bar, (0.75, "3 of 4"))
            while app.return_value is None:
                await pilot.pause(0.02)
        self.assertIsNone(app.bar)                      # (the command ended)
        bridge.log.close()

    async def test_tab_shows_an_earlier_steps_output_and_esc_the_live_one(self):
        def body():
            S.step(1, "checking this PC")
            S.say("  the first step's line")
            S.step(2, "your choices")
            return S.ask("Download them now?", ["y", "n"], "n", False)

        def text(app):
            return "\n".join(line.text for line in app.query_one("#output").lines)

        async def look(app, pilot):
            await pilot.press("shift+tab")
            await pilot.pause(0.1)
            self.assertEqual(app.view, 1)
            self.assertIn("the first step's line", text(app))
            self.assertNotIn("Your choices", text(app))
            await pilot.press("escape")
            await pilot.pause(0.1)
            self.assertIsNone(app.view)
            self.assertIn("Your choices", text(app))
        bridge = Bridge(self.log)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "1.0", "./maya.sh", self.log)
        async with app.run_test(size=(110, 40)) as pilot:
            while app.pending is None:
                await pilot.pause(0.02)
            await pilot.pause(0.1)
            await look(app, pilot)
            await pilot.press("y")
            while app.return_value is None:
                await pilot.pause(0.02)
        self.assertEqual(app.return_value, ("done", "y"))
        bridge.log.close()

    @unittest.skipIf(os.name == "nt", "the process group check is POSIX")
    async def test_ctrl_c_twice_stops_the_setup_and_the_running_command(self):
        went_on = []
        code = ("import os, subprocess, sys, time\n"
                "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(f'pids {os.getpid()} {c.pid}', flush=True)\n"
                "time.sleep(60)\n")

        def body():
            S.run([sys.executable, "-c", code])
            went_on.append(True)
        bridge = Bridge(self.log)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "Project Maya - setup", "./maya.sh", self.log)
        async with app.run_test(size=(110, 32)) as pilot:
            while not any(t.startswith("pids ") for _, _, t in bridge.entries):
                await pilot.pause(0.05)
            await pilot.press("ctrl+c")
            self.assertIsInstance(app.screen, T.StopScreen)
            await pilot.press("ctrl+c")
        self.assertEqual(app.return_value, ("stopped", None))
        bridge.thread.join(5)
        self.assertFalse(bridge.thread.is_alive())
        self.assertEqual(went_on, [])
        for pid in map(int, next(t for _, _, t in bridge.entries if t.startswith("pids ")).split()[1:]):
            for _ in range(50):                         # (the child is reaped by init a moment later)
                try:
                    os.kill(pid, 0)
                    alive = open(f"/proc/{pid}/stat").read().split()[2] != "Z" if os.path.exists("/proc") else True
                except ProcessLookupError:
                    alive = False
                if not alive:
                    break
                time.sleep(0.05)
            self.assertFalse(alive, pid)
        bridge.log.close()

    INFO = {"model": "glm-5.3-flash", "quant": "Maya-L", "context": "32768", "dashboard": "http://127.0.0.1:9/",
            "api": "http://127.0.0.1:9/v1", "log": "/x/maya-maya-l.log"}   # (port 9: nothing answers - not a real Maya)

    @unittest.skipIf(os.name == "nt", "SIGINT to a process group is POSIX")
    async def test_maya_runs_on_the_screen_after_the_setup_and_ctrl_c_stops_it_as_without_it(self):
        marker = Path(self.dir.name) / "closed"
        server = ("import sys, time\ntry:\n    print('loading the model (the first start takes a minute or two) ...')\n"
                  "    print('ready: http://127.0.0.1:8080/v1  (OpenAI: /v1/chat/completions)')\n"
                  "    print('[strata] done: 512 tokens in 12 s (42.1 tok/s)')\n"
                  "    time.sleep(60)\nexcept KeyboardInterrupt:\n"   # (the server closes the engine on Ctrl+C)
                  f"    open({str(marker)!r}, 'w').write('closed')\n")

        def body():
            S.step(8, "the configuration and the start script")
            S.ok("config: maya-maya-l.json")
            return S.UI.serve([sys.executable, "-c", server], dict(os.environ, PYTHONUNBUFFERED="1"), self.INFO)
        bridge = Bridge(self.log)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "1.0", "./maya.sh", self.log)
        async with app.run_test(size=(110, 40)) as pilot:
            t0 = time.time()
            while not (app.serving and app.serving["speed"]):
                self.assertLess(time.time() - t0, 20)
                await pilot.pause(0.05)
            self.assertEqual((app.serving["state"], app.state[T.SERVE], app.state[8]), ("ready", "ready", "done"))
            self.assertEqual(app.serving["speed"], "512 tokens at 42.1 tok/s")
            self.assertIn("ready: http://127.0.0.1:8080/v1  (OpenAI: /v1/chat/completions)", app.served)   # (its Log)
            await pilot.press("ctrl+c")
            self.assertTrue(app.screen.serving)          # "Stop Maya?"
            await pilot.press("ctrl+c")
        self.assertEqual(app.return_value, ("stopped", None))
        bridge.finish(10)
        self.assertTrue(marker.exists())                 # SIGINT, as Ctrl+C without the screen: not killed
        bridge.log.close()
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("[ok] config: maya-maya-l.json", log)
        self.assertNotIn("ready:", log)                  # (the server's output stays on the screen: days of it)
        self.assertFalse(any("ready:" in text for _, _, text in bridge.entries))

    def test_a_ready_line_is_found_wherever_it_sits(self):
        for line in ("ready: http://127.0.0.1:8080/v1", "\x1b[1;32mready: http://127.0.0.1:8080/v1\x1b[0m",
                     "loading 99%\rready: http://127.0.0.1:8080/v1", "  ready: http://127.0.0.1:8080/v1"):
            self.assertTrue(T.is_ready(line), repr(line))
        self.assertFalse(T.is_ready("not ready: the model is loading"))

    async def test_the_screen_is_ready_when_the_server_answers_though_the_ready_line_was_missed(self):
        import http.server

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        info = dict(self.INFO, api=f"http://127.0.0.1:{httpd.server_port}/v1")
        server = "import time\nprint('loading the model ...', flush=True)\ntime.sleep(30)"   # (no "ready:" line at all)

        def body():
            S.step(8, "the configuration and the start script")
            return S.UI.serve([sys.executable, "-c", server], dict(os.environ, PYTHONUNBUFFERED="1"), info)
        bridge = Bridge(self.log)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "1.0", "./maya.sh", self.log)
        try:
            async with app.run_test(size=(110, 40)) as pilot:
                t0 = time.time()
                while not (app.serving and app.serving["state"] == "ready"):
                    self.assertLess(time.time() - t0, 20, "stuck at Loading the model")
                    await pilot.pause(0.05)
                self.assertEqual(app.state[T.SERVE], "ready")
                bridge.stop()
        finally:
            httpd.shutdown()
        bridge.finish(10)
        bridge.log.close()

    async def test_a_server_that_ends_is_a_failure_and_an_update_starts_the_new_version(self):
        for code, keys, want in (("print('boom'); sys.exit(3)", ["enter"], "fail"), ("sys.exit(75)", [], "done")):
            with self.subTest(code=code), \
                    patch.object(maya, "set_up", return_value=Path("maya-test.json")), \
                    patch.object(maya, "server_command", return_value=([sys.executable, "-c", "import sys; " + code],
                                                                      None, self.INFO)):
                end, _, app = await self.drive(lambda: maya.set_up_and_serve(SimpleNamespace(no_start=False), {}),
                                               *([keys] if keys else []))
                self.assertEqual(end[0], want)
                if want == "fail":
                    self.assertEqual(end[1][0], "Maya stopped: its server ended with exit code 3")
                    self.assertIn("boom", app.served)
                else:
                    self.assertEqual(end[1], (Path("maya-test.json"), maya.UPDATE_EXIT))

    def test_curl_meter_and_what_runs(self):
        self.assertEqual(T.curl_meter(" 12 31.2G   12 3900M    0     0  85.1M      0  0:06:15  0:00:45  0:05:30 88.2M"),
                         (0.12, "3900MB of 31.2GB, 88.2MB/s, 0:05:30 left"))
        self.assertIsNone(T.curl_meter("  % Total    % Received % Xferd  Average Speed   Time    Time     Time  "
                                       "Current"))
        self.assertEqual(T.describe(["curl", "-L", "-o", "/m/x.gguf", "https://hf"]), "Downloading x.gguf")
        self.assertEqual(T.describe(["/v/bin/python", "-m", "pip", "install", "numpy"]), "Installing Python packages")
        self.assertEqual(T.describe(["/v/bin/cmake", "--build", "build"]), "Compiling")

    def test_the_logo(self):                            # font8x8 in quadrant characters, its rows as wide as each other
        rows = L.big("maya.")
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({len(r) for r in rows}), 1)
        self.assertTrue(set("".join(rows)) <= set(L.QUADRANTS))
        self.assertEqual(L.mix("#000000", "#ffffff", 0.5), "#808080")


if __name__ == "__main__":
    unittest.main()
