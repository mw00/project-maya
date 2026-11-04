"""The setup on a screen of its own (./maya.sh's first run and --setup), laid out like binsider
(github.com/orhun/binsider) in the dashboard's colours (setup_look.py): the steps as tabs along the top, the logo, a
card with what runs now and its progress, the output in a box below it, a question in a box of its own, and the keys
in the bottom border.  Tab / Shift+Tab show an earlier step's output, Esc the live one.  Textual (MIT) draws it: it
comes with Maya in third_party/wheels, and maya.py puts it into .venv from there (setup_screen).

The setup is the same code as without the screen: it runs in a thread of its own, and setup.py's output functions
(say, step, ok, warn, fail, ask, run, progress) hand everything to a Bridge (setup_bridge.py) instead of printing it
(setup.UI).  When the screen closes the terminal gets the steps with their results - and when the setup ended early,
that step's messages and the end of its output - and maya-setup.log keeps all of it.
"""
from __future__ import annotations

import queue
import re
from collections import deque
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.widgets import OptionList, RichLog, Static
from textual.widgets.option_list import Option

from setup_bridge import TAIL, Bridge, curl_meter, describe
from setup_dash import Dashboard
from setup_look import (ACCENT, ACCENT_TEXT, DANGER, EDGE, FAINT, INK, LINE, MUTED, OK, OK_TEXT, SETUP_CSS, THEME,
                        TINT, Card, Output, StopScreen, clock, hints, serving_card, setup_card, splash, styled, title)

# maya.py's steps (setup.step) as tabs, the tuning after them, then Maya running (Bridge.serve): its dashboard, its log
STEPS = {1: "This PC", 2: "Choices", 3: "Packages", 4: "Engine", 5: "Model", 6: "Pack", 7: "Images", 8: "Config",
         9: "Tuning", 10: "Dashboard", 11: "Log"}
SERVE, LOG = 10, 11
ROOT = Path(__file__).resolve().parents[1]              # the Maya folder (top right)
READY = re.compile(r"(?:^|\s|\x1b\[[0-9;]*m)ready: http")      # the server's "ready:" line, wherever it sits in the chunk
PROBE_S = 3.0                                       # while it loads: ask /health this often (a missed "ready:" line)


def is_ready(line: str) -> bool:
    return bool(READY.search(line)) or "the model runs with" in line or "engine is running again" in line


def health_ok(api: str) -> bool:
    """Maya's server answers /health: it is up (api: http://host:port/v1)."""
    import urllib.request
    try:
        with urllib.request.urlopen(api.rstrip("/").removesuffix("/v1") + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:                                # noqa: BLE001 - not up yet
        return False


DONE = re.compile(r"\] done: (\d+) tokens in [\d.]+ s \(([\d.]+) tok/s\)")   # the server, after an answer
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
DRAIN_S, IDLE_DRAIN_S = 0.05, 0.25                        # the setup's events: looked for this often / when quiet
TICK_S, IDLE_TICK_S = 0.12, 1.0                        # the card's timer: a spinner moving / only seconds counting
NINJA = re.compile(r"\[(\d+)/(\d+)\] ")               # ninja's "[412/1033] Building CUDA object ..."


class SetupApp(App):
    CSS = SETUP_CSS
    BINDINGS = [Binding("ctrl+c", "stop", show=False, priority=True),
                Binding("ctrl+q", "stop", show=False, priority=True),
                Binding("tab", "browse(1)", show=False, priority=True),
                Binding("shift+tab", "browse(-1)", show=False, priority=True),
                Binding("escape", "live", show=False)]
    ENABLE_COMMAND_PALETTE = False
    TITLE = "Project Maya"

    def __init__(self, bridge: Bridge, body, version: str, me: str, log_path: Path | None, steps: bool = True):
        super().__init__()
        self.bridge, self.body, self.version, self.me, self.log_path = bridge, body, version, me, log_path
        self.steps = steps                              # False: Maya's start alone - no setup steps, no setup log
        self.state = {n: "wait" for n in STEPS}
        self.times: dict = {}                           # step -> [start, end or None]
        self.titles: dict = {}                          # step -> its title, as maya.py says it
        self.warned: set = set()
        self.cur = 0                                    # the step running now
        self.view = None                                # the step whose output the box shows (None: the live one)
        self.frame = 0
        self.doing, self.doing_since, self.said_doing, self.last = "", 0.0, False, ""
        self.bar, self.bar_since = None, 0.0            # (share done, what to show) while a command tells it
        self.pending = None                             # the question shown: (future, values, kind)
        self.abouts: list = []                          # its options' descriptions (under the highlighted one)
        self.shown = 0                                  # the Bridge's entries handled so far
        self.written, self.flow = 0, 0                  # lines in the output box, and the width they wrap at
        self.printed = ""                               # print() from other threads, until its line ends
        self.serving = None                             # Maya's server, once it runs here: what the card shows
        self.probed, self.probing = 0.0, None            # the last /health look while it loads, and its thread
        self.served = deque(maxlen=5000)                # its output (only here: it can run for days)

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), "maya-edge": EDGE, "maya-line": LINE, "maya-tint": TINT,
                "maya-muted": MUTED, "maya-accent-text": ACCENT_TEXT}

    # ---------------------------------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static(id="tabs")
            where = str(ROOT).replace(str(Path.home()), "~")
            yield Static(where if len(where) <= 32 else "…/" + ROOT.name, id="where")
        with Vertical(id="main"):
            with Vertical(id="page"):                   # the setup's page: logo, card, output, a question
                yield Static(splash(), id="splash")
                with Center():
                    yield Card(id="card")
                with Vertical(id="lower"):
                    yield Output(id="output", wrap=True, min_width=20, max_lines=5000)
                    with Vertical(id="ask"):
                        yield Static(id="intro")
                        yield OptionList(id="options")
                        yield Static(id="about")
            yield Dashboard(id="dash")                  # Maya's, once it runs

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "maya"
        self.query_one("#dash").display = False
        self.query_one("#top").border_title = Text.assemble(("│", EDGE), ("Project Maya", f"bold {INK}"),
                                                            (f" v{self.version}", MUTED), ("│", EDGE))
        out = self.query_one("#output")
        out.border_title = title("Output")
        out.border_subtitle = Text(self.log_path.name if self.log_path else "", f"italic {FAINT}")
        self.show_keys()
        self.quiet, self.drain_every = 0, DRAIN_S
        self.drain_timer = self.set_interval(DRAIN_S, self.drain)
        self.tick_every = TICK_S
        self.tick_timer = self.set_interval(TICK_S, self.tick)
        self.begin_capture_print(self)
        self.stop_on_signals()

    def stop_on_signals(self) -> None:
        """SIGTERM / SIGHUP (a service manager, `kill`, the terminal closed) stop the screen as a confirmed Ctrl+C does:
        the setup's thread ends, Maya's server and its engine end (POSIX; on Windows the server ends with this process:
        its --parent-pipe and job object).  A signal that is ignored (nohup) stays ignored."""
        import os
        import signal
        def stop():
            if not getattr(self, "signalled", False):   # (once: a second SIGINT would end the server at once)
                self.signalled = True
                self.stop_answered(True)
        if os.name != "nt":   # (Windows has no SIGHUP and no loop signal handlers; the setup's thread starts all the same)
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGTERM, signal.SIGHUP):
                    if signal.getsignal(sig) is not signal.SIG_IGN:
                        loop.add_signal_handler(sig, stop)
            except (ImportError, NotImplementedError, RuntimeError, ValueError):
                pass
        self.bridge.thread = threading.Thread(target=self.work, name="setup", daemon=True)
        self.bridge.thread.start()

    def on_resize(self, event: events.Resize) -> None:
        self.screen.set_class(event.size.width < 100, "narrow")
        self.screen.set_class(event.size.height < 32, "short")      # (the logo goes first,
        self.screen.set_class(event.size.height < 44, "tight")      # and while a question needs the room)
        self.call_after_refresh(self.draw_tabs)
        self.call_after_refresh(self.reflow)

    def reflow(self) -> None:
        """The output again at the box's new width (a line keeps the wrap it was written with)."""
        if self.written and self.query_one("#output").scrollable_content_region.width != self.flow:
            self.show_view(self.view)

    def on_print(self, event: events.Print) -> None:
        """print() from another thread - the engine's start narration while the tuning loads it - as output."""
        self.printed += event.text
        *lines, self.printed = self.printed.split("\n")
        self.bridge.entries.extend((self.bridge.step_no, "out", line) for line in lines)

    def work(self) -> None:
        """The setup's thread: maya.py's set_up, then the screen closes."""
        try:
            end = ("done", self.body())
        except KeyboardInterrupt:
            end = ("stopped", None)
        except SystemExit as e:
            end = ("fail", self.bridge.failure or (f"the setup ended (exit {e.code})", None))
        except BaseException as e:                     # noqa: BLE001 - a bug: raised again once the screen is gone
            end = ("error", e)
        self.bridge.events.put(("end", end))

    # ---------------------------------------------------------------------------------------- the setup's events
    def drain(self) -> None:
        worked = False
        for _ in range(500):
            if self.pending is not None:                # the setup waits for this answer
                break
            try:
                event = self.bridge.events.get_nowait()
            except queue.Empty:
                break
            worked = True
            self.flush()                                # (its lines come before what the event writes itself)
            getattr(self, "ev_" + event[0])(*event[1:])
            if event[0] != "live" or self.animating():  # (the timer slows itself again once nothing moves)
                self.set_tick(TICK_S)
        worked = self.flush() or worked
        # (twenty looks a second while something comes in; once it has been quiet for a second, four a second - a
        # screen that waits for requests need not wake 20 times a second to find nothing)
        self.quiet = 0 if worked else self.quiet + 1
        want = DRAIN_S if self.quiet < 20 else IDLE_DRAIN_S
        if want != self.drain_every:
            self.drain_every = want
            self.drain_timer.stop()
            self.drain_timer = self.set_interval(want, self.drain)

    def flush(self) -> bool:
        """The Bridge's new entries into the output box (when it shows the live output); whether there were any."""
        new = self.bridge.entries[self.shown:]
        self.shown += len(new)
        for n, kind, text in new:
            if self.view is None:
                self.write(kind, text)
        return bool(new)

    def write(self, kind: str, text: str) -> None:
        out = self.query_one("#output", RichLog)
        width = out.scrollable_content_region.width     # (wrapped at the box's width, not the console's 80)
        if kind == "step" and self.written:
            out.write(Text(""))
        out.write(styled(kind, text), width=width if width > 20 else None)
        self.written, self.flow = self.written + 1, width

    def ev_say(self, msg: str) -> None:
        line = msg.strip()
        if line.endswith("..."):                        # "Compiling the engine for sm_86 (...) ...": what runs now
            self.set_doing(line.rstrip(". "))
            self.said_doing = True
        elif line and not line.startswith(">"):
            self.last = line

    def ev_step(self, n: int, title: str) -> None:
        self.end_step()
        self.cur, self.state[n], self.times[n] = n, "run", [time.monotonic(), None]
        self.titles[n] = title[:1].upper() + title[1:]
        self.set_doing("")
        self.bar, self.last = None, ""
        self.draw_tabs()

    def end_step(self, state: str | None = None) -> None:
        if self.cur in self.times:
            self.times[self.cur][1] = time.monotonic()
            self.state[self.cur] = state or ("warn" if self.cur in self.warned else "done")
            self.draw_tabs()

    def ev_ok(self, msg: str) -> None:
        self.last = msg

    def ev_warn(self, msg: str) -> None:
        self.warned.add(self.cur)
        self.last = msg

    def ev_out(self, line: str) -> None:
        m = NINJA.match(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            took = time.monotonic() - self.bar_since if self.bar else 0.0
            left = f" · about {clock(took * (total - done) / done)} left" if took > 15 and done > total * 0.03 else ""
            self.set_bar(done / total, f"{done} of {total}{left}")

    def ev_status(self, line: str) -> None:
        if self.serving and is_ready(line):             # (a "ready:" that came with a carriage return: still ready)
            self.ev_live(line)
            return
        meter = curl_meter(line)
        if meter:
            self.set_bar(*meter)
        elif line.strip():
            self.last = line.strip()

    def ev_progress(self, msg, frac) -> None:
        if msg is None:
            self.bar = None
        elif frac is not None:
            self.set_bar(frac, msg)
        else:
            self.last = msg

    def ev_run(self, cmd: list) -> None:
        if not self.said_doing or Path(cmd[0]).stem.lower() == "curl":
            self.set_doing(describe(cmd))
        self.bar = None

    def ev_ran(self, rc: int) -> None:
        self.bar = None
        self.set_doing("")
        if self.cur == SERVE:
            self.serving["state"], self.state[SERVE] = f"Stopped (exit code {rc})", "fail"
            self.draw_tabs()

    def ev_serve(self, info: dict) -> None:
        """The setup is done: Maya's server runs on this screen now."""
        self.end_step()
        self.cur, self.state[SERVE], self.times[SERVE] = SERVE, "run", [time.monotonic(), None]
        self.state[LOG], self.times[LOG] = "done", [time.monotonic(), None]
        self.serving = {**info, "state": "Loading the model", "since": time.monotonic(), "speed": None}
        log = Path(info["log"]).name if info.get("log") else ""
        self.query_one("#output").border_subtitle = Text(f"engine log: {log}" if log else "", f"italic {FAINT}")
        self.query_one(Dashboard).follow(info["dashboard"] + "metrics", info.get("key") or "", info.get("log"),
                                         info.get("gpus") or [], info.get("vram"))
        self.show_view(None)

    def ev_live(self, line: str) -> None:
        self.served.append(line)
        if self.view == LOG:
            self.write("live", line)
        s, m = self.serving, DONE.search(line)
        if is_ready(line):
            s["state"], s["since"], self.state[SERVE] = "ready", time.monotonic(), "ready"
            self.draw_tabs()
        elif "reloading the model" in line or "the engine had stopped" in line:   # (a load again: from 0%)
            self.query_one(Dashboard).load.reset()
            s["state"], s["since"], self.state[SERVE] = ("Reloading the model" if "reloading" in line else
                                                         "Starting the engine again"), time.monotonic(), "run"
        elif m:
            s["speed"] = f"{m.group(1)} tokens at {m.group(2)} tok/s"

    def ev_up(self) -> None:
        """/health answered while the screen still said it loads: the line that says so was missed."""
        s = self.serving
        if s and s["state"] != "ready":
            s["state"], s["since"], self.state[SERVE] = "ready", time.monotonic(), "ready"
            self.draw_tabs()

    def probe(self) -> None:
        """While the screen says the model loads, /health is asked now and then (a thread; never blocks the screen)."""
        s, now = self.serving, time.monotonic()
        if not s or s["state"] == "ready" or not s.get("api") or now - self.probed < PROBE_S:
            return
        self.probed = now
        api = s["api"]
        if self.probing and self.probing.is_alive():
            return
        self.probing = threading.Thread(target=lambda: health_ok(api) and self.bridge.events.put(("up",)),
                                        daemon=True)
        self.probing.start()

    def ev_end(self, end: tuple) -> None:
        self.end_step("fail" if end[0] in ("fail", "error") else None)
        self.exit(end)

    def set_doing(self, what: str) -> None:
        self.doing, self.doing_since, self.said_doing = what, time.monotonic(), False

    def set_bar(self, frac: float, info: str) -> None:
        if self.bar is None:
            self.bar_since = time.monotonic()
        self.bar = (frac, info)

    # ---------------------------------------------------------------------------------------- questions
    def ev_ask(self, question: str, choices: list, default: str, fut: Future) -> None:
        if [c.lower() for c in choices] == ["y", "n"]:
            self.show_question(fut, "yn", question, [], [("Yes", None, None), ("No", None, None)], choices,
                               choices.index(default), [])
        else:
            self.show_question(fut, "ask", question, [], [(c, None, None) for c in choices], choices,
                               choices.index(default) if default in choices else 0, [])

    def ev_choose(self, question, intro, options, default, outro, fut: Future) -> None:
        self.show_question(fut, "choose", question, intro, options, list(range(len(options))), default, outro)

    def ev_fail(self, msg: str, hint, fut: Future) -> None:
        self.end_step("fail")
        self.set_doing("")
        self.query_one("#output", RichLog).write(Text.assemble(("✗ ", f"bold {DANGER}"), (msg, f"bold {DANGER}")))
        again = "" if self.serving is not None else f" - fix it and run {self.me} again: everything finished is kept"
        self.show_question(fut, "fail", msg, [hint] if hint else [], [(f"Exit{again}", None, None)], [None], 0, [])

    def show_question(self, fut, kind, question, intro, options, values, default, outro) -> None:
        if self.serving is not None and self.view is None:   # (the dashboard shows: the log's page has the box)
            self.show_view(LOG)
        self.pending = (fut, values, kind)
        self.abouts = [" ".join(x for x in (o[2], " ".join(outro)) if x) for o in options]
        box = self.query_one("#ask")
        named = kind != "fail" and len(question) <= 40  # a short question names the box, a long one opens it
        stopped = "Maya stopped" if self.serving is not None else "The setup stopped"   # (once Maya ran: its server)
        box.border_title = title(stopped, f"bold {DANGER}") if kind == "fail" else \
            title(question if named else "Question")
        body = Text.assemble((question, f"bold {DANGER if kind == 'fail' else INK}") if not named else "",
                             ("\n" if intro and not named else "") + " ".join(intro))
        self.query_one("#intro", Static).update(body)
        self.query_one("#intro").display = bool(body.plain)
        ol = self.query_one("#options", OptionList)
        ol.clear_options()
        ol.add_options([Option(Text.assemble(f"{i}) " if kind == "choose" and len(options) < 10 else "", label,
                                             (f"   {note}", f"italic {OK_TEXT}") if note else ""))
                        for i, (label, note, _) in enumerate(options, 1)])
        ol.highlighted = default
        self.query_one("#about", Static).update(self.abouts[default])
        box.set_class(kind == "fail", "fail")
        box.add_class("on")
        self.screen.add_class("asking")
        ol.focus()
        self.show_keys()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if self.pending is not None and event.option_list.id == "options" and event.option_index < len(self.abouts):
            self.query_one("#about", Static).update(self.abouts[event.option_index])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.pending is not None and event.option_list.id == "options":
            self.answer(event.option_index)

    def on_key(self, event) -> None:
        if self.pending is None or isinstance(self.screen, StopScreen):
            return
        kind, n = self.pending[2], len(self.pending[1])
        if kind == "yn" and event.key in ("y", "n"):
            event.stop()
            self.answer(0 if event.key == "y" else 1)
        elif kind == "choose" and event.key.isdigit() and 1 <= int(event.key) <= n:
            event.stop()
            self.answer(int(event.key) - 1)

    def answer(self, i: int) -> None:
        fut, values, _ = self.pending
        self.pending = None
        self.query_one("#ask").remove_class("on")
        self.screen.remove_class("asking")
        self.show_keys()
        fut.set_result(values[i])

    # ---------------------------------------------------------------------------------------- keys
    def action_stop(self) -> None:
        if isinstance(self.screen, StopScreen):         # Ctrl+C twice
            self.screen.dismiss(True)
        elif self.pending is not None and self.pending[2] == "fail":
            self.answer(0)
        else:
            self.push_screen(StopScreen(self.me, self.serving is not None), self.stop_answered)

    def stop_answered(self, stop: bool) -> None:
        if not stop:
            return
        self.bridge.stop()
        if self.pending is not None:
            fut, self.pending = self.pending[0], None
            fut.set_exception(KeyboardInterrupt())
        self.exit(("stopped", None))

    def action_browse(self, by: int) -> None:
        """Tab / Shift+Tab: the output of the next / the step before (back at the one running: the live output)."""
        seen = [n for n in STEPS if n in self.times]
        if isinstance(self.screen, StopScreen) or not seen:
            return
        at = seen.index(self.view if self.view is not None else self.cur)
        pick = seen[(at + by) % len(seen)]
        self.show_view(None if pick == self.cur else pick)

    def action_live(self) -> None:
        if self.view is not None:
            self.show_view(None)

    def show_view(self, view) -> None:
        """The output of a step, or Maya's log - or (None) the live page: the setup's, then Maya's dashboard."""
        self.view = view
        dash = self.serving is not None and view is None
        self.query_one("#dash").display, self.query_one("#page").display = dash, not dash
        if dash:
            self.query_one(Dashboard).focus()           # (↑↓ and PgUp / PgDn scroll it)
        self.call_after_refresh(self.replay)            # (once the output box has its width back)
        self.draw_tabs()
        self.show_keys()

    def replay(self) -> None:
        view, out = self.view, self.query_one("#output", RichLog)
        out.clear()
        self.written = 0
        for n, kind, text in ([(LOG, "live", x) for x in self.served] if view == LOG else
                              self.bridge.entries[:self.shown][-5000:]):
            if view is None or n == view:
                self.write(kind, text)
        out.border_title = title("Output" if view is None else "Log" if view == LOG else f"Step {view} · {STEPS[view]}")

    # ---------------------------------------------------------------------------------------- drawing
    def animating(self) -> bool:
        """Whether anything on the screen moves faster than the clock: the spinner (a step or the model loading, a
        question waiting, a stop) - not Maya waiting for requests, which only counts seconds."""
        s = self.serving
        return s is None or bool(self.pending) or s["state"] != "ready" or self.state.get(self.cur) == "run"

    def set_tick(self, every: float) -> None:
        """The card's and the tabs' timer at `every` seconds.  A screen that only counts seconds (Maya ready, no
        request) ticks once a second instead of eight times: each tick that changes a line lays the screen out again."""
        if every != self.tick_every:
            self.tick_every = every
            self.tick_timer.stop()
            self.tick_timer = self.set_interval(every, self.tick)

    def tick(self) -> None:
        self.set_tick(TICK_S if self.animating() else IDLE_TICK_S)
        self.frame += 1
        self.probe()
        if self.state.get(self.cur) == "run":
            self.draw_tabs()
        if self.serving is None or self.pending:
            self.query_one("#card", Card).show(setup_card(self, SPIN[self.frame % len(SPIN)]))
        else:                                           # (the log's page and the dashboard show the same card)
            text = serving_card(self.serving, SPIN[self.frame % len(SPIN)])
            self.query_one("#card", Card).show(text)
            self.query_one("#dash-card", Card).show(text)

    def draw_tabs(self) -> None:
        spin = SPIN[self.frame % len(SPIN)]
        tabs = []
        for n in STEPS:
            if n > 8 and n not in self.times or n < SERVE and not self.steps:   # (the tuning and Maya: once
                continue                                                        # they start; a start: Maya's only)
            state = self.state[n]
            style = {"wait": FAINT, "run": f"bold {INK}", "ready": f"bold {OK}", "fail": f"bold {DANGER}"}.get(
                state, ACCENT_TEXT) + (" bold underline" if n == self.view else "")   # (done, warned or not: alike)
            mark = (spin + " ", ACCENT) if state == "run" else ("● ", OK) if state == "ready" else ""
            tabs.append(Text.assemble(mark, (STEPS[n], style)))
        room = self.query_one("#tabs").size.width or 70
        for sep in (" │ ", "│"):                        # (an 80-column terminal: closer together)
            line = Text(sep, LINE).join(tabs)
            if line.cell_len <= room:
                break
        self.query_one("#tabs", Static).update(line)

    def show_keys(self) -> None:
        kind, keys = self.pending[2] if self.pending else None, []
        if kind == "fail":
            self.query_one("#main").border_subtitle = hints(("Enter", "Exit"))
            return
        if kind == "yn":
            keys = [("Enter", "Choose"), ("y n", "Answer")]
        elif kind:
            keys = [("Enter", "Choose"), ("↑↓", "Move")] + ([(f"1-{len(self.pending[1])}", "Pick")]
                                                            if kind == "choose" else [])
        live, maya = self.view is None, self.serving is not None
        keys += [("↑↓", "Scroll")] * (maya and live) + \
            [("Tab", "Log" if maya and live else "Steps" if self.steps else "Dashboard")] + \
            ([] if live or not self.steps else [("Esc", "Dashboard" if maya else "Live")]) + \
            [("Ctrl+C", "Stop Maya" if maya else "Stop")]
        self.query_one("#main").border_subtitle = hints(*keys)


def run(body, version: str, log_path: Path | None, me: str, steps: bool = True):
    """body() - maya.py's set_up (and Maya's start), or the start alone (steps False) - on the setup's screen; returns
    what it returns.  A fail() in it, read on the screen, is printed once the screen has closed and ends the setup as
    without it; a stop is a KeyboardInterrupt here.  log_path: the setup's whole output (None: none kept)."""
    import setup as S
    bridge = Bridge(log_path)
    S.UI = bridge
    app = SetupApp(bridge, body, version, me, log_path, steps)
    try:
        end = app.run()
    except BaseException:
        bridge.stop()                                   # the screen broke: the setup's thread ends too
        raise
    finally:
        bridge.closed = True
    if not isinstance(end, tuple):                      # the screen ended by itself (its reason is printed above)
        bridge.stop()
        S.fail("the setup's screen stopped", f"{me} --plain runs the setup without it")
    kind, value = end
    if kind != "stopped":                               # (after a stop the setup's thread may still be in a line)
        S.UI = None
    for line in bridge.summary(whole=(kind != "done" or value is None) and app.serving is None):
        print(line)
    if log_path:
        print(f"\n  The setup's whole output: {log_path}")
    bridge.log.close()
    if app.serving is not None and kind != "done":      # Maya ran here: the end of its server's output
        print("\n" + "\n".join(list(app.served)[-TAIL:]))
    if bridge.interrupted is not None:
        print("  Stopping ...", flush=True)
        bridge.finish()
    if kind == "done":
        return value
    if kind == "fail":                                  # (once Maya ran, the setup had not stopped: Maya's server did)
        S.fail(*value, end=S.STOPPED if app.serving is None else None)
    if kind == "error":
        raise value
    print(f"  Maya stopped: {me} starts it again." if app.serving else
          f"  Stopped: everything finished so far is kept, and {me} continues where it stopped.")
    raise KeyboardInterrupt
