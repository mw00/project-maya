"""tools/test_setup_dash.py - Maya's dashboard on its screen (setup_dash.py): the Monitor's numbers from a GET /metrics
like the server's, the screen's switch from the setup to the dashboard and its log, and ./maya.sh's start on the screen
(the real engine's server, no setup steps).  No GPU, no network; needs Textual (skipped without it)."""
from pathlib import Path
import io
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import setup as S  # noqa: E402
from setup_bridge import Bridge  # noqa: E402
try:
    from rich.console import Console
    from rich.text import Text
    import setup_dash as D
    import setup_tui as T
except ImportError:
    D = T = None

NOW = time.time()
GPU = {"util": 41, "mem_used": 14 * 2**30, "mem_total": 16 * 2**30, "temp": 61, "power": 88.0, "power_limit": 180.0,
       "pcie_gen": 4, "pcie_gen_max": 5, "pcie_width": 8, "pcie_rx_mb": 3100.0}
METRICS = {   # GET /metrics as the server sends it (serve/server.py, metrics) - two GPUs, an answer being written
    "engine": {"model": "glm-5.3-flash", "max_context": 32768, "experts": 19456, "expert_slots": 5248, "ram_gb": 92.8},
    "live": {"state": "generating", "queued": 0, "phase": "answering", "prompt_tokens": 2048, "generated": 312,
             "max_tokens": 600, "elapsed_s": 11.0, "tok_s": 28.5, "prefill_tok_s_mean": 410.0},
    "tiers": {"now": {"tok_s": 28.4, "ms_tok": 35.1, "vram_hit": 0.912, "ram_fetch": 0.62, "disk": 0.08,
                      "promo": 0.21, "vram_used": 5210, "ram_used": 11850, "vram_gb": 40.2, "ram_gb": 91.4,
                      "time": NOW},
              "history": {"ram_fetch": [0.5, 0.7, 0.6], "disk": [0, 0.1, 0], "promo": [0.2, 0.3, 0.2]}},
    "requests": [{"time": NOW - 60, "duration_s": 17.2, "finish": "stop", "api": "openai", "prefill_tok_s": 382.9,
                  "prompt_tokens": 1533, "reused": 0, "output_tokens": 234, "decode_tok_s": 18.5, "hit_rate": 0.92}],
    "totals": {"since": NOW - 600, "requests": 1, "prompt_tokens": 1533, "reused": 0, "output_tokens": 234,
               "prompt_ms": 4003.6, "decode_ms": 12586.5},
    "hardware": {"gpu_util": 41, "gpu_mem_used": 28 * 2**30, "gpu_mem_total": 32 * 2**30, "gpu_temp": 61,
                 "gpu_power": 176.0, "gpu_power_limit": 360.0, "gpu_pcie_gen_max": 5, "gpu_pcie_width": 8,
                 "gpu_pcie_rx_mb": 6200.0, "cpu": 34.0, "ram_used": 140 * 2**30, "ram_total": 251 * 2**30,
                 "disk_read_mb": 12.0, "disk_write_mb": 0.4,
                 "gpus": [dict(GPU, index=0), dict(GPU, index=1, temp=88)]},
    "hardware_static": {"gpu_name": "NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 3090", "cores": 88,
                        "threads": 88},
    "history": {"tok_s": [0, 12.0, 27.0, 28.5], "gpu_util": [10, 40, 41], "cpu": [30, 34], "ram_used": [1, 2]},
    "reload": None, "time": NOW}


def text(renderable, width: int = 120) -> str:
    console = Console(file=io.StringIO(), width=width, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


@unittest.skipIf(T is None, "Textual is not installed (./maya.sh --setup installs it)")
class Boxes(unittest.TestCase):
    """Each box from GET /metrics, as the web Monitor computes it (serve/web/app.js)."""

    def setUp(self):
        METRICS["tiers"]["now"]["time"] = time.time()   # (a fresh reading: older than 30 s is "the last answer's")

    def test_throughput(self):
        out = text(D.speed(METRICS, 56))
        for want in ("tok/s", "writing now", "prefill 410", "Answering", "312 tokens", "35.1 ms", "91.2%",
                     "2.4k / 32K", "Requests"):
            self.assertIn(want, out)
        self.assertEqual(len(D.big("28.5")), 4)          # (the speed in font8x8, as the logo)

    def test_where_the_experts_live(self):
        out = text(D.tiers(METRICS, 56))
        for want in ("VRAM", "5,210", "40.2 GB", "Pinned RAM", "11,850", "SSD only", "2,396", "From RAM", "0.62"):
            self.assertIn(want, out)
        self.assertIn("live · 19,456 routed experts", D.tiers_sub(METRICS).plain)
        self.assertIn("while it writes", text(D.tiers({"engine": {}}, 56)))    # (the mock engine: no tiers)

    def test_the_tiers_bar_fits_when_the_tiers_overlap(self):   # (Maya-L on 9 GPUs: 10,813 + 1,571 of 12,096)
        now = {**METRICS["tiers"]["now"], "vram_used": 10813, "ram_used": 1571}
        m = {**METRICS, "engine": {**METRICS["engine"], "experts": 12096}, "tiers": {**METRICS["tiers"], "now": now}}
        for width in (103, 61, 42):
            first, second = text(D.tiers(m, width), width).split("\n")[:2]
            self.assertEqual(len(first.rstrip()), width)
            self.assertTrue(second.startswith("■"), second)        # (the legend next: nothing of the bar wrapped)

    def test_the_tiers_advice_fits_on_one_line(self):          # (a word of it on a line of its own: no more)
        m = {**METRICS, "tiers": {**METRICS["tiers"], "now": {**METRICS["tiers"]["now"], "vram_hit": 0.33,
                                                                 "ram_fetch": 19.8}}}
        for width, said in ((120, 0), (92, 1), (42, 2), (30, 2)):
            out = text(D.tiers(m, width), width + 2).rstrip("\n").split("\n")
            tip = D.TIPS["ram"][said]
            self.assertTrue(out[-1].strip() == tip if len(tip) <= width else out[-1].strip().endswith("…"), out[-1])
            self.assertIn(out[-1].strip()[:20], tip)

    def test_gpus_hardware_requests_and_banners(self):
        gpus = text(D.gpus(METRICS))
        for want in ("RTX 5060 Ti", "RTX 3090", "41%", "14.0 / 16 GB", "88 / 180 W", "Gen5 x8", "3,100 MB/s"):
            self.assertIn(want, gpus)
        hw = text(D.hardware(METRICS, 116))
        for want in ("GPU load", "41%", "28.0 / 32 GB", "5,248 experts cached", "CPU", "88 cores", "RAM",
                     "92.8 GB pinned", "6.1 GB/s"):
            self.assertIn(want, hw)
        reqs = text(D.requests(METRICS, 116))
        for want in ("Done", "OpenAI", "1,533", "234", "18.5", "92.0%", "17.2 s", "Since"):
            self.assertIn(want, reqs)
        self.assertIn("GPU 1 is at 88 °C", D.banners(METRICS, 0).plain)
        self.assertIn("does not answer", D.banners(METRICS, 3).plain)

    def test_an_empty_server_is_no_error(self):            # (before the first request, or a server just started)
        empty = {"engine": {}, "live": {"state": "idle"}, "hardware": {}, "history": {}}
        for box in (D.speed(empty, 40), D.tiers(empty, 40), D.gpus(empty), D.hardware(empty, 80),
                    D.requests(empty, 80)):
            text(box)

    def test_one_gpu_and_a_request_without_its_prefill(self):  # (a one-card server: gpu_* only; older requests)
        hw = {k: v for k, v in METRICS["hardware"].items() if k != "gpus"}
        m = dict(METRICS, hardware=hw, hardware_static={"gpu_name": "NVIDIA GeForce RTX 3090"},
                 requests=[dict(METRICS["requests"][0], prefill_tok_s=None, prompt_ms=4000.0)])
        self.assertIn("RTX 3090", text(D.gpus(m)))
        self.assertIn("28.0 / 32 GB", text(D.gpus(m)))
        self.assertIn("383", text(D.requests(m, 116)))          # 1,533 tokens read in 4 s

    def test_a_boxes_shape_is_what_decides_its_layout(self):   # (put: laid out again only when it changes)
        a, b = D.speed(METRICS, 50), D.speed(dict(METRICS, live=dict(METRICS["live"], tok_s=31.0)), 50)
        self.assertEqual(D.shape(a), D.shape(b))                # a new reading of the same box: the same shape
        self.assertIsNotNone(D.shape(a))
        self.assertIsNotNone(D.shape(D.gpus(METRICS)))
        self.assertNotEqual(D.shape(D.gpus(METRICS)), D.shape(D.gpus(dict(METRICS, hardware=dict(METRICS["hardware"], gpus=[GPU])))))
        self.assertEqual(D.shape(Text("a\nb")), 1)
        self.assertIsNone(D.shape("not a renderable we know"))


    def test_the_engines_facts_come_as_text(self):           # (its INFO line: "ram_gb": "107.8", as the server sends)
        m = dict(METRICS, engine=dict(METRICS["engine"], ram_gb="107.8", vram_gb="12.3"))
        self.assertIn("107.8 GB pinned", text(D.hardware(m, 116)))
        self.assertIn("107.8 GB pinned for …", text(D.hardware(m, 96)))         # (a card too narrow: cut, marked)


@unittest.skipIf(T is None, "Textual is not installed (./maya.sh --setup installs it)")
class Screen(unittest.IsolatedAsyncioTestCase):
    async def test_maya_running_shows_the_dashboard_and_tab_its_log(self):
        server = "import time\nprint('ready: http://127.0.0.1:9/v1')\nprint('[strata] a log line')\ntime.sleep(30)"
        info = {"model": "glm-5.3-flash", "quant": "Maya-L", "context": "32768", "dashboard": "http://127.0.0.1:9/",
                "api": "http://127.0.0.1:9/v1", "log": None, "key": ""}
        with tempfile.TemporaryDirectory() as d:
            bridge = Bridge(Path(d) / "setup.log")
            S.UI = bridge
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            app = T.SetupApp(bridge, lambda: S.UI.serve([sys.executable, "-c", server], env, info), "1.0", "./maya.sh",
                             Path(d) / "setup.log")
            try:
                async with app.run_test(size=(120, 40)) as pilot:
                    t0 = time.time()
                    while app.state[T.SERVE] != "ready":
                        self.assertLess(time.time() - t0, 20, "the stand-in server did not get ready")
                        await pilot.pause(0.05)
                    dash = app.query_one(T.Dashboard)
                    dash.metrics = METRICS                  # (as its poll would read it from the server)
                    await pilot.pause(1.3)
                    self.assertTrue(dash.display)
                    self.assertFalse(app.query_one("#page").display)
                    self.assertIs(dash.drawn, METRICS)
                    await pilot.press("tab")                # the server's log
                    await pilot.pause(0.2)
                    self.assertEqual(app.view, T.LOG)
                    self.assertTrue(app.query_one("#page").display)
                    self.assertIn("[strata] a log line", "\n".join(x.text for x in app.query_one("#output").lines))
                    await pilot.press("escape")             # back to the dashboard
                    await pilot.pause(0.1)
                    self.assertIsNone(app.view)
                    self.assertTrue(dash.display)
                    await pilot.press("ctrl+c", "ctrl+c")
                bridge.finish(10)
            finally:
                S.UI = None
                bridge.log.close()


@unittest.skipIf(T is None, "Textual is not installed (./maya.sh --setup installs it)")
class IdleScreen(unittest.IsolatedAsyncioTestCase):
    """Issue #685: a screen with Maya ready and no request does as little as it shows - the card's clock once a second
    (not the spinner's eight), the setup's events looked for less often, and a new reading of the dashboard's boxes
    drawn without laying the whole screen out again."""

    async def test_ready_and_quiet_slows_the_timers_and_lays_out_rarely(self):
        server = "import time\nprint('ready: http://127.0.0.1:9/v1')\ntime.sleep(60)"
        info = {"model": "glm-5.3-flash", "quant": "Maya-L", "context": "32768", "dashboard": "http://127.0.0.1:9/",
                "api": "http://127.0.0.1:9/v1", "log": None, "key": ""}
        with tempfile.TemporaryDirectory() as d:
            bridge = Bridge(Path(d) / "setup.log")
            S.UI = bridge
            app = T.SetupApp(bridge, lambda: S.UI.serve([sys.executable, "-c", server], dict(os.environ, PYTHONUNBUFFERED="1"),
                                                        info), "1.0", "./maya.sh", Path(d) / "setup.log")
            try:
                async with app.run_test(size=(120, 40)) as pilot:
                    t0 = time.time()
                    while app.state[T.SERVE] != "ready":
                        self.assertLess(time.time() - t0, 20, "the stand-in server did not get ready")
                        await pilot.pause(0.05)
                    self.assertFalse(app.animating())
                    await pilot.pause(1.6)
                    self.assertEqual(app.tick_every, T.IDLE_TICK_S)             # the clock: once a second
                    self.assertEqual(app.drain_every, T.IDLE_DRAIN_S)           # nothing comes in: four looks a second
                    # a model loading again moves the spinner: the timer is quick at once
                    app.serving["state"] = "Reloading the model"
                    self.assertTrue(app.animating())
                    app.set_tick(T.TICK_S)
                    self.assertEqual(app.tick_every, T.TICK_S)
                    app.serving["state"] = "ready"
                    # the dashboard's boxes: the same shape is only drawn, a new one (or every tenth reading) laid out
                    dash = app.query_one(T.Dashboard)
                    seen = []
                    real = T.Dashboard.put.__globals__["Static"].update
                    def spy(widget, content="", *, layout=True):
                        seen.append((widget.id, layout))
                        return real(widget, content, layout=layout)
                    with patch.object(T.Dashboard.put.__globals__["Static"], "update", spy):
                        dash.puts.clear()
                        dash.shapes.clear()
                        for _ in range(12):
                            dash.put("#dash-gpus", D.gpus(METRICS))
                        one = [lay for i, lay in seen if i == "dash-gpus"]
                        self.assertEqual(one[0], True)                          # the first: no shape known yet
                        self.assertEqual(one[1:9], [False] * 8)                 # the same shape: drawn only
                        self.assertEqual(one[9], True)                          # every tenth: laid out all the same
                        seen.clear()
                        dash.put("#dash-gpus", D.gpus(dict(METRICS, hardware=dict(METRICS["hardware"], gpus=[GPU]))))
                        self.assertEqual([lay for i, lay in seen], [True])      # fewer cards: a new shape
                    await pilot.press("ctrl+c", "ctrl+c")
                bridge.finish(10)
            finally:
                S.UI = None
                bridge.log.close()


@unittest.skipIf(T is None, "Textual is not installed (./maya.sh --setup installs it)")
class Loading(unittest.TestCase):
    """The model's start in the loading box: the engine log's warm-up of each GPU's experts as one share (by GB; the
    GPUs not started yet count as the mean), from where the log stood when the start began."""

    def watch(self, lines, gpus=(3, 0, 1), vram=None):
        log = Path(tempfile.mkdtemp()) / "maya-maya-l.log"
        log.write_text("glm fast: CUDA0 warming the expert tiers 100% (25.0 of 25.0 GB, 50 s)\n")   # an earlier start's
        w = D.LoadWatch(str(log), list(gpus), vram)
        with open(log, "a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
        w.read()
        return w

    def test_the_share_across_the_gpus(self):
        self.assertEqual(self.watch([]).fraction(), 0.0)          # (the earlier start's 100% does not count)
        w = self.watch(["glm fast: CUDA0 warming the expert tiers 50% (10.0 of 20.0 GB, 9 s)"])
        self.assertAlmostEqual(w.fraction(), 10 / 60)              # 20 GB known; the other two: 20 each
        w = self.watch(["glm fast: CUDA0 tiers warm: 2016 experts (20.0 GB) in 50 s (0.50 GB/s)",
                        "glm fast: CUDA1 warming the expert tiers 50% (5.0 of 10.0 GB, 4 s)"])
        self.assertAlmostEqual(w.fraction(), 25 / 45)              # 20 + 5 of 20 + 10 + the mean, 15
        out = text(D.loading(w, 60))
        for want in ("25.0 of about 45 GB of experts warm", "Warming the experts on GPU 0 (part 2 of 3)", "GPU 3",
                     "✓ 20.0 GB", "5.0 of 10.0 GB", "waiting"):
            self.assertIn(want, out)
        w.reset()                                                   # (a reload: from 0% again)
        self.assertEqual(w.fraction(), 0.0)

    def test_a_gpu_not_started_yet_counts_by_its_vram(self):  # (Maya-L on the 3090 + 8x 16 GB: 148.3 GB, never 225)
        gpus, vram = [3, 0, 1, 2, 4, 5, 6, 7, 8], {3: 24.0, **{g: 16.0 for g in (0, 1, 2, 4, 5, 6, 7, 8)}}
        w = self.watch(["glm fast: CUDA0 tiers warm: 2016 experts (25.0 GB) in 50 s (0.50 GB/s)"], gpus, vram)
        total, known = w.total()
        self.assertAlmostEqual(total, 25 + 8 * 16 * 25 / 24)       # the 3090's 25 GB per its 24 GB, on each 16 GB
        self.assertFalse(known)
        self.assertIn("25.0 of about 158 GB of experts warm", text(D.loading(w, 60)))
        sizes = (25.0, 13.4, 13.4, 16.8, 13.4, 16.8, 13.4, 16.8, 19.3)
        w = self.watch([f"glm fast: CUDA{k} tiers warm: 1152 experts ({gb} GB) in 30 s (0.45 GB/s)"
                        for k, gb in enumerate(sizes)], gpus, vram)
        self.assertAlmostEqual(w.total()[0], 148.3)
        self.assertTrue(w.total()[1])
        self.assertIn("148.3 of 148.3 GB of experts warm", text(D.loading(w, 60)))   # (all known: no "about")

    def test_before_the_warm_up_after_it_and_without_a_log(self):
        self.assertIn("Reading the model's weights", text(D.loading(self.watch([]), 60)))
        w = self.watch([f"glm fast: CUDA{k} tiers warm: 1152 experts (13.4 GB) in 30 s (0.45 GB/s)" for k in range(3)])
        self.assertEqual(w.fraction(), 1.0)
        self.assertIn("almost ready", text(D.loading(w, 60)))
        blind = D.LoadWatch(None, [0])
        self.assertIsNone(blind.fraction())
        self.assertIn("so far", text(D.loading(blind, 60)))


@unittest.skipIf(T is None, "Textual is not installed (./maya.sh --setup installs it)")
class Start(unittest.IsolatedAsyncioTestCase):
    """./maya.sh on an installed Maya: the real engine (server.py --engine maya) on the screen - its dashboard and
    log, without the setup's steps or its log; --plain (or no terminal): in the terminal as before."""

    async def test_a_start_runs_the_engine_on_the_screen(self):
        got, served = {}, []
        cmd = [sys.executable, "serve/server.py", "--engine", "maya", "--config", "maya-test.json"]
        ui = SimpleNamespace(serve=lambda c, env, info: served.append((c, info)) or 0)

        def run(body, version, log_path, me, steps=True):
            got.update(log=log_path, steps=steps)
            return body()
        with patch.object(maya, "setup_screen", return_value=True), patch("setup_tui.run", side_effect=run), \
                patch.object(maya, "server_command", return_value=(cmd, {}, {"dashboard": "http://127.0.0.1:9/"})), \
                patch.object(maya.S, "UI", ui), patch.object(maya, "run_server") as plain:
            self.assertEqual(maya.start(Path("maya-test.json"), SimpleNamespace(plain=False, yes=True)), 0)
        self.assertEqual(served[0][0], cmd)
        self.assertEqual(got, {"log": None, "steps": False})   # (the setup's log stays the setup's)
        plain.assert_not_called()

    def test_plain_starts_in_the_terminal(self):
        with patch.object(maya, "setup_screen", return_value=False), \
                patch.object(maya, "server_command", return_value=(["server"], {}, {})), \
                patch.object(maya, "run_server", return_value=0) as plain, patch("setup_tui.run") as screen:
            self.assertEqual(maya.start(Path("maya-test.json"), SimpleNamespace(plain=True, yes=False)), 0)
        plain.assert_called_once()
        screen.assert_not_called()
        self.assertFalse(maya.setup_screen(SimpleNamespace(plain=True, yes=False), asks=False))

    async def test_the_loading_box_until_ready_and_a_failure_on_the_log_page(self):
        server = "import sys, time\ntime.sleep(1.5)\nprint('ready: http://127.0.0.1:9/v1')\ntime.sleep(2)\nsys.exit(3)"
        info = {"model": "glm-5.3-flash", "dashboard": "http://127.0.0.1:9/", "api": "http://127.0.0.1:9/v1",
                "log": None, "key": "", "gpus": [0]}

        def body():
            rc = S.UI.serve([sys.executable, "-c", server], dict(os.environ, PYTHONUNBUFFERED="1"), info)
            S.fail(f"Maya stopped: its server ended with exit code {rc}", "the reason is in its output")
        bridge = Bridge(None)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "1.0", "./maya.sh", None, steps=False)
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                async def until(cond):
                    t0 = time.time()
                    while not cond():
                        self.assertLess(time.time() - t0, 20)
                        await pilot.pause(0.05)
                await until(lambda: app.serving is not None)
                await pilot.pause(1.1)
                self.assertTrue(app.query_one("#dash-load").display)       # loading: the box, not the numbers
                self.assertFalse(app.query_one("#dash-gpus").display)
                await until(lambda: app.state[T.SERVE] == "ready")
                await pilot.pause(1.1)
                self.assertFalse(app.query_one("#dash-load").display)
                self.assertTrue(app.query_one("#dash-gpus").display)
                await until(lambda: app.pending is not None)               # the server ended: the failure ...
                self.assertEqual(app.view, T.LOG)                          # ... on the log's page, in sight
                self.assertTrue(app.query_one("#page").display)
                self.assertIn("Maya stopped", str(app.query_one("#ask").border_title))   # (not the setup)
                await pilot.press("enter")
            self.assertEqual(app.return_value[0], "fail")
        finally:
            S.UI = None
            bridge.log.close()

    def test_a_failed_start_says_its_engine_s_last_line_and_not_setup_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "maya.log"
            log.write_text("strata generate: an earlier start's error\n")

            def serve(cmd, env, info):
                with open(log, "a") as f:
                    f.write("glm fast: CUDA8 tiers warm: 1440 experts (19.3 GB)\n"
                            "strata generate: pack: out of memory\n")
                return 1
            failed = []
            ui = SimpleNamespace(serve=serve, fail=lambda m, h: failed.append((m, h)), say=lambda m: True)
            with patch.object(maya, "server_command", return_value=([], {}, {"log": str(log)})), \
                    patch.object(S, "UI", ui), self.assertRaises(SystemExit):
                maya.serve_on_screen(Path("maya-test.json"), SimpleNamespace())
        self.assertIn("its engine's last line: strata generate: pack: out of memory", failed[0][1])
        self.assertNotIn("earlier", failed[0][1])
        app = SimpleNamespace(run=lambda: ("fail", failed[0]), serving={"state": "starting"}, served=[])
        out = io.StringIO()
        try:
            with patch.object(T, "SetupApp", return_value=app), patch("sys.stdout", out), self.assertRaises(SystemExit):
                T.run(lambda: None, "1.0", None, "./maya.sh", steps=False)
        finally:
            S.UI = None
        self.assertIn("pack: out of memory", out.getvalue())
        self.assertNotIn("Setup stopped", out.getvalue())          # (Maya's server stopped, not the setup)

    async def test_only_maya_s_tabs_and_starting_maya_before_it_runs(self):
        server = "import time\nprint('ready: http://127.0.0.1:9/v1')\ntime.sleep(30)"
        info = {"model": "glm-5.3-flash", "quant": "Maya-L", "context": "32768", "dashboard": "http://127.0.0.1:9/",
                "api": "http://127.0.0.1:9/v1", "log": None, "key": ""}

        def body():
            time.sleep(0.5)                                     # (what runs before Maya: an engine update's compile)
            return S.UI.serve([sys.executable, "-c", server], dict(os.environ, PYTHONUNBUFFERED="1"), info)
        bridge = Bridge(None)
        S.UI = bridge
        app = T.SetupApp(bridge, body, "1.0", "./maya.sh", None, steps=False)
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.2)
                self.assertIn("Starting Maya", app.query_one("#card").plain)
                t0 = time.time()
                while app.state[T.SERVE] != "ready":
                    self.assertLess(time.time() - t0, 20, "the stand-in server did not get ready")
                    await pilot.pause(0.05)
                app.draw_tabs()
                tabs = app.query_one("#tabs").content.plain
                self.assertIn("Dashboard", tabs)
                self.assertNotIn("This PC", tabs)
                self.assertTrue(app.query_one("#dash").display)
                await pilot.press("ctrl+c", "ctrl+c")
            bridge.finish(10)
        finally:
            S.UI = None
            bridge.log.close()


if __name__ == "__main__":
    unittest.main()
