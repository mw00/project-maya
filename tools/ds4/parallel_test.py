#!/usr/bin/env python3
"""tools/ds4/parallel_test.py - several conversations at once on maya-ds4 (`--parallel N`): the answers are the ones
each conversation gets alone, a STOP ends one alone, the kept conversations (MAYA_DS4_SLOT_KEEP) work per lane, a new
prompt is read while the others answer - and what it costs and what it gains, measured.

    python tools/ds4/parallel_test.py --pack /mnt/data/models/.../maya-ds4 [--exe build/maya-ds4] [--gpus 0,2,1]
                                      [--config maya-ds4.json] [--lanes 3] [--skip-long] [--measure] [--spec 0]

The engine gets the environment this test inherits plus, with `--config <maya-ds4.json>`, that config's DeepSeek
settings (`DS4_*` and `MAYA_DS4_*` names, under the names the engine reads).  Every run uses a scratch
MAYA_DS4_SLOT_DIR (the pack's own slots/ is never touched: the engine prunes what is not its own at start), and the
engine's stdin/stdout protocol - no ports, no server.

Checks (greedy, spec=0 so a conversation's state ends exactly at its last printed token, as slot_keep_test.py's):
  alone     the reference: four salted prompts, one at a time, `--parallel 1`, twice in two runs of one process
            (run-to-run determinism is measured, not assumed: where the two reads disagree the comparison below is
            against both)
  together  one process, `--parallel <lanes>`: the same prompts sent at once, all lanes busy - every answer must be
            the one it got alone, token for token (the check the GLM engine's docs give for --parallel)
  slots     `--parallel 2` with MAYA_DS4_SLOT_KEEP=1: A and B at once, then A's next turn - it reuses A's whole
            prefix from the lane (DONE's `reused`), so a conversation that stays in its lane is not read again
  stop      two requests at once: STOP <rid> one of them; its DONE says `cancel` with fewer tokens, the other's
            answer is whole (a STOP is a request's own)
  reading   `--parallel 2`: a ~100K-token prompt and, a moment later, a small one: the small one's answer must make
            progress while the long prompt is still being read (the long read pauses at its chunk boundaries, the
            small request is admitted and read in its own lane, and the other conversation decodes in the gaps)

The exactness checks (alone, together, slots, stop) run with MAYA_DS4_PREFILL_GAPS=0: a gap's decode can move experts
in the pool's lent tail, so a later chunk can find others resident than the uncut read.  The reading and measurement
runs use the default (adaptive gaps, shortest remaining work first) - a config's own MAYA_DS4_PREFILL_GAPS left out.

`together` is the run that confirms the engine on the hardware: a conversation's window is held in its own slots of the
mailbox (a layer's device publishes its routing there, the engine's service answers it) and the service takes the
published requests in the order the devices publish them - so the conversations it runs at once must each answer token
for token what it answers alone (run it with `--lanes 2` and `--lanes 4`, the GLM engine's docs' pair).  A wrong slot (a
layer computed with another layer's experts) or a request held behind another conversation's later one shows up here and
nowhere else.  Without a GPU, `tools/test_ds4_parallel.py`'s Mailbox drives the same protocol with host-only structs:
three conversations' windows of 40 layers issued ahead, published a stage at a time, taken as published.

Measurements (`--measure`, the table is printed for the PR): for N = 1 .. lanes, four 2K-token conversations (and with
`--long-len`, four of that many tokens) of `--max-new` tokens each, in turns of N at once (a request past the N states
is answered with ERR; a row with a failed request says so): wall time, tokens/s in all, tokens/s a
conversation, and the GPUs' mean utilisation while they answer (nvidia-smi, sampled every 250 ms).  One conversation's
own speed (N=1) against the GPUs' utilisation is the number that decides the win: where the cards are already busy
with one conversation, several cannot add much.

Exit 0: every check passed.  1: a check failed (the reason is printed).  2: the engine did not start or answer, or the
pack's tokenizer is missing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from slot_keep_test import Engine, config_env, text_tokens  # noqa: E402  (one engine process; salted real text)


class Multi(Engine):
    """`maya-ds4 --serve` with several conversations: a request's GEN carries its rid, its lines come back with it
    (T / PP / DONE / ERR <rid> ...), and a reader thread routes them so requests run at the same time."""

    def __init__(self, exe: str, args: list[str], env: dict):
        super().__init__(exe, args, env)
        self.multi = 0
        self.got: dict[int, list] = {}
        self.other: list[str] = []
        self.stat = ""
        self.rids = 0
        self.dead: str | None = None
        self.thread: threading.Thread | None = None
        self.cond = threading.Condition()

    def start(self, timeout: float) -> None:
        super().start(timeout)
        f = self.ready.split()
        self.multi = next((int(x[6:]) for x in f if x.startswith("multi=") and x[6:].isdigit()), 0)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                self._route(self._line(time.monotonic() + 86400.0))
        except Exception as e:   # noqa: BLE001 - (the engine ended: every wait says so, with the reason)
            with self.cond:
                self.dead = str(e)
                self.cond.notify_all()

    def _route(self, line: str) -> None:
        kind = line.split(None, 1)[0] if line.strip() else ""
        if not self.multi and kind in ("T", "PP", "DONE", "ERR"):
            # (--parallel 1 is the engine as before: its lines carry no rid - they are the one request's in flight)
            line = f"{kind} {self.rids} {line[len(kind):].lstrip()}"
        f = line.split(None, 2)
        if len(f) >= 2 and f[0] in ("T", "PP", "DONE", "ERR") and f[1].isdigit():
            with self.cond:
                self.got.setdefault(int(f[1]), []).append((f[0], line, time.monotonic()))
                self.cond.notify_all()
            return
        if line.startswith("STAT "):
            self.stat = line
        elif line.startswith("NOTE "):
            self.notes.append(line[5:].strip())
        elif line.startswith(("INFO ", "VRAM ", "READY ")):
            pass
        else:
            self.other.append(line)

    def send(self, prompt: list[int], max_new: int, spec: int = 0, rid: int | None = None) -> int:
        if rid is None:
            with self.cond:
                self.rids += 1
                rid = self.rids
                self.got.setdefault(rid, [])
        else:
            with self.cond:
                self.got.setdefault(rid, [])
        self.proc.stdin.write(f"GEN {max_new} rid={rid} spec={spec} {','.join(str(t) for t in prompt)}\n".encode())
        self.proc.stdin.flush()
        return rid

    def stop_request(self, rid: int) -> None:
        self.proc.stdin.write(f"STOP {rid}\n".encode())
        self.proc.stdin.flush()

    def stream(self, rid: int, timeout: float = 7200.0):
        """(kind, line, time) for this request, until its DONE or ERR; then a dict of what it was."""
        t0, end = time.monotonic(), time.monotonic() + timeout
        out = {"tokens": [], "times": [], "pp": [], "reused": None, "finish": None, "generated": 0, "error": None}
        seen = 0
        while True:
            with self.cond:
                lines = self.got.setdefault(rid, [])
                while seen >= len(lines) and self.dead is None:
                    left = end - time.monotonic()
                    if left <= 0:
                        raise RuntimeError(f"request {rid}: no line for {time.monotonic() - t0:.0f} s")
                    self.cond.wait(min(left, 1.0))
                if seen >= len(lines):
                    raise RuntimeError(f"request {rid}: the engine ended ({self.dead})")
                kind, line, t = lines[seen]
                seen += 1
            if kind == "T":
                out["tokens"].append(int(line.split()[2]))
                out["times"].append(t)
            elif kind == "PP":
                out["pp"].append((line, t))
            elif kind == "ERR":
                out["error"] = line.split(None, 2)[2].strip() if len(line.split()) > 2 else line
                return out
            else:
                f = line.split()
                out["generated"], out["finish"] = int(f[2]), f[6]   # (DONE <rid> <generated> <prompt> <ms> <ms> <finish>)
                out["reused"] = int(f[9])
                out["s"] = time.monotonic() - t0
                return out

    def run_at_once(self, prompts: list[list[int]], max_new: int, spec: int) -> list[dict]:
        rids = [self.send(p, max_new, spec) for p in prompts]
        return [self.stream(r) for r in rids]

    def run_in_order(self, prompts: list[list[int]], max_new: int, spec: int) -> list[dict]:
        return [self.stream(self.send(p, max_new, spec)) for p in prompts]


class Util:
    """The GPUs' utilisation while something runs (nvidia-smi, every 250 ms): the series and the mean."""

    def __init__(self, every: float = 0.25):
        self.every = every
        self.series: list[list[float]] = []
        self.stop = threading.Event()
        self.t: threading.Thread | None = None

    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                vals = [float(x) for x in out.split() if x.strip()]
                if vals:
                    self.series.append(vals)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(self.every)

    def __exit__(self, *a):
        self.stop.set()
        if self.t is not None:
            self.t.join(timeout=5)

    def mean(self) -> list[float]:
        if not self.series:
            return []
        n = min(len(s) for s in self.series)
        return [round(sum(s[i] for s in self.series) / len(self.series), 1) for i in range(n)]


def engine_args(a, parallel: int, ctx: int) -> list[str]:
    return [*(("--gpus", a.gpus) if a.gpus else ()), "--pack", a.pack, "--max-context", str(ctx), "--dspark",
            "on", "--threads", "on", "--parallel", str(parallel), "--vision", "off", "--expert-profile", "none",
            "--serve"]


def one_run(a, settings: dict, parallel: int, ctx: int, slots: str, keep: bool = False) -> Multi:
    env = dict(settings)
    env.update({"MAYA_DS4_SLOT_DIR": slots, "MAYA_DS4_SLOT_KEEP": "1" if keep else "0"})
    eng = Multi(a.exe, engine_args(a, parallel, ctx), env)
    t0 = time.monotonic()
    eng.start(timeout=3600.0)
    eng.load_s = time.monotonic() - t0
    return eng


def main() -> int:
    ap = argparse.ArgumentParser(description="several conversations at once on maya-ds4 (--parallel N)")
    ap.add_argument("--pack", required=True, help="the model's pack dir (maya-ds4)")
    ap.add_argument("--exe", default=str(Path(__file__).resolve().parents[2] / "build/maya-ds4"),
                    help="the engine (default: <repo>/build/maya-ds4)")
    ap.add_argument("--gpus", default="", help="the GPUs (default: every visible one, the engine's own split)")
    ap.add_argument("--config", default="", help="a model config (maya-ds4.json): its DS4_/MAYA_DS4_ settings")
    ap.add_argument("--lanes", type=int, default=3, help="the conversations at once to test and measure (1-8)")
    ap.add_argument("--ctx", type=int, default=131072, help="--max-context")
    ap.add_argument("--len", type=int, default=2000, help="the checks' prompts (tokens)")
    ap.add_argument("--long-len", type=int, default=100000, help="the measurement's long prompts (tokens)")
    ap.add_argument("--max-new", type=int, default=64, help="the measurement's answers (tokens)")
    ap.add_argument("--check-new", type=int, default=24, help="the checks' answers (tokens)")
    ap.add_argument("--spec", type=int, default=0, help="spec= (0: the state ends at the printed token, as the checks "
                                                        "need); the measurement uses the same")
    ap.add_argument("--skip-long", action="store_true", help="the interleaving check and the long measurement only "
                                                             "with --long-len, which a --ctx below it cannot run")
    ap.add_argument("--measure", action="store_true", help="run and print the measurement table")
    a = ap.parse_args()
    settings = config_env(a.config) if a.config else {}
    # exactness (alone vs together, slots, stop) is checked with the decode gaps off: a gap's decode can move
    # experts in the pool's lent tail, so a later chunk can find other experts resident than the uncut read.  The
    # reading check needs them on.
    settings_off = dict(settings)
    settings_off["MAYA_DS4_PREFILL_GAPS"] = "0"
    settings_gaps = {k: v for k, v in settings.items() if k != "MAYA_DS4_PREFILL_GAPS"}

    if not os.path.exists(a.exe):
        print(f"{a.exe}: no engine - build it first (cmake --build build --target maya-ds4)", flush=True)
        return 2
    try:
        from strata_tokenizer import Tokenizer   # (tools/strata_tokenizer.py's own: `regex`)
        tok = Tokenizer.from_dir(os.path.join(a.pack, "tokenizer"))
    except Exception as e:  # noqa: BLE001
        print(f"{os.path.join(a.pack, 'tokenizer')}: no tokenizer (the prompts are real text): {e}", flush=True)
        return 2
    lanes = max(1, min(8, a.lanes))
    probes = [text_tokens(tok, f"Conversation {c}.", a.len) for c in "ABCD"]
    tmp = tempfile.mkdtemp(prefix="ds4-parallel-")
    ok = True

    def check(what: str, cond: bool) -> None:
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
        ok = ok and cond

    def diff(x: list[int], y: list[int]) -> str:
        d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        return f"token {d}: {x[d:d + 4]} vs {y[d:d + 4]}"

    print(f"engine {a.exe}, lanes {lanes}, prompts {a.len} tokens, answers {a.check_new} tokens (spec={a.spec})",
          flush=True)

    # ---- alone: one at a time, --parallel 1, twice (the prompt path's own run-to-run agreement is measured)
    alone_runs: list[list[list[int]]] = []
    try:
        eng = one_run(a, settings_off, 1, a.ctx, os.path.join(tmp, "alone1"))
        for k in range(2):
            got = []
            for i, p in enumerate(probes):
                r = eng.stream(eng.send(p, a.check_new, a.spec))
                if r["error"]:
                    raise RuntimeError(f"alone {i}: {r['error']}")
                got.append(r["tokens"])
            alone_runs.append(got)
        eng.quit()
    except Exception as e:  # noqa: BLE001
        print(f"the alone run failed: {e}", flush=True)
        return 2
    same = all(x == y for x, y in zip(*alone_runs))
    print(f"  {'ok' if same else 'note'}    alone: two reads of the same prompts "
          f"{'agree' if same else 'differ - compared against both'} ({eng.load_s:.0f} s to load)", flush=True)
    ref = alone_runs[0]

    def matches(i: int, got: list[int]) -> bool:
        # (a shorter answer - the stop check's 12 - is the start of what it answers alone)
        return bool(got) and any(got == run[i][:len(got)] for run in alone_runs)

    # ---- together: the same prompts at once, --parallel <lanes>
    try:
        eng = one_run(a, settings_off, lanes, a.ctx, os.path.join(tmp, "together"))
        if eng.multi != lanes:
            raise RuntimeError(f"READY said multi={eng.multi}, not {lanes}")
        got = eng.run_at_once(probes[:lanes], a.check_new, a.spec)
        for i, r in enumerate(got):
            if r["error"]:
                raise RuntimeError(f"{i}: {r['error']}")
            check(f"together: conversation {i} answered what it answers alone ({len(r['tokens'])} tokens)",
                  matches(i, r["tokens"]))
            if not matches(i, r["tokens"]):
                print("        " + diff(r["tokens"], ref[i]), flush=True)
        # ---- stop: two at once, one stopped - the other's answer is whole
        rids = [eng.send(probes[0], 300, a.spec), eng.send(probes[1], 12, a.spec)]
        short = eng.stream(rids[1])
        time.sleep(0.3)
        eng.stop_request(rids[0])
        long_ = eng.stream(rids[0])
        check(f"stop: the stopped request ends `cancel` with fewer tokens ({long_['finish']}, {len(long_['tokens'])})",
              long_["finish"] == "cancel" and len(long_["tokens"]) < 300)
        check("stop: the other request's answer is whole", short["generated"] == 12 and len(short["tokens"]) == 12)
        check("stop: and is what it answers alone", matches(1, short["tokens"]))
        eng.quit()
    except Exception as e:  # noqa: BLE001
        print(f"the together run failed: {e}", flush=True)
        ok = False

    # ---- slots: with the conversations kept (MAYA_DS4_SLOT_KEEP=1), A's conversation leaves its lane when another
    # request takes it, and A's next turn takes it back whole (its prefix reused, not read again)
    try:
        eng = one_run(a, settings_off, min(2, lanes), a.ctx, os.path.join(tmp, "slots"), keep=True)
        r = eng.run_at_once([probes[2], probes[3]], a.check_new, a.spec)      # A and B, a lane each
        third = eng.stream(eng.send(probes[0], a.check_new, a.spec))          # C: takes a lane, A's is the older
        turn = probes[2] + r[0]["tokens"] + text_tokens(tok, "The next turn:", 8)
        nxt = eng.stream(eng.send(turn, 8, a.spec))
        check(f"slots: A's next turn reused {nxt['reused']} of its {len(turn)} tokens (A's lane had been taken)",
              nxt["error"] is None and nxt["reused"] == len(probes[2]) + a.check_new - 1)   # (all but the last
              # printed token, never fed: slot_keep_test.py's reuse_check)
        check("slots: and answered from there", nxt["error"] is None and nxt["generated"] == 8)
        eng.quit()
    except Exception as e:  # noqa: BLE001
        print(f"the slots run failed: {e}", flush=True)
        ok = False

    # ---- reading: a long prompt and a small one after it; the small one's answer must not wait for the whole read
    if not a.skip_long:
        try:
            eng = one_run(a, settings_gaps, 2, a.ctx, os.path.join(tmp, "reading"))
            big = text_tokens(tok, "A long one.", min(a.long_len, a.ctx - a.max_new - 1024))
            small = text_tokens(tok, "A short one.", 256)
            long_rid = eng.send(big, 4, a.spec)
            time.sleep(0.5)
            t_small = time.monotonic()
            small_rid = eng.send(small, 16, a.spec)
            got = eng.stream(small_rid)
            first_t = got["times"][0] if got["times"] else float("inf")
            # the long prompt's own PP lines tell when its read was done: the one whose tokens done reach its total
            # (none yet: the read is still going)
            with eng.cond:
                big_end = [t for k, l, t in eng.got.get(long_rid, [])
                           if k == "PP" and int(l.split()[2]) >= int(l.split()[3])]
            read_end = big_end[0] if big_end else float("inf")
            early = len([t for t in got["times"] if t < read_end])
            check(f"reading: {early} of the small answer's {len(got['times'])} tokens came before the long read's "
                  f"last piece", early > 0 and got["generated"] == 16)
            if a.measure and got["times"]:
                dt = got["times"][-1] - got["times"][0]
                print(f"  reading measurement: first token {first_t - t_small:.1f} s after the small prompt was sent; "
                      f"{len(got['times'])} tokens in {dt:.1f} s while the long read ran", flush=True)
            eng.stop_request(long_rid)
            eng.quit()
        except Exception as e:  # noqa: BLE001
            print(f"the reading run failed: {e}", flush=True)
            ok = False

    # ---- the measurement: the same conversations, N at once (the table is what the PR puts in)
    if a.measure:
        print("\nmeasurement (four conversations, one chain of prompts; spec=%d):" % a.spec, flush=True)
        print("  N  prompts   wall    tok/s  a conversation (slowest / mean) tok/s  GPUs (mean %)", flush=True)
        for length, label in ((a.len, f"{a.len}"), (a.long_len, f"{a.long_len}")):
            if length == a.long_len and a.skip_long:
                continue
            ps = [text_tokens(tok, f"Measure {c}.", length) for c in "ABCD"]
            for n in range(1, lanes + 1):
                eng = None
                try:
                    eng = one_run(a, settings_gaps, n, a.ctx, os.path.join(tmp, f"m{n}-{length}"))
                    with Util() as u:
                        t0 = time.monotonic()
                        # (the four in turns of N at once: the engine answers a request past its N states with ERR)
                        got = []
                        for i in range(0, len(ps), n):
                            got += eng.run_at_once(ps[i:i + n], a.max_new, a.spec)
                        wall = time.monotonic() - t0
                    bad = [r["error"] for r in got if r["error"]]
                    if bad:
                        raise RuntimeError(f"{len(bad)} of {len(got)} requests failed: {bad[0]}")
                    tok_n = sum(len(r["tokens"]) for r in got)
                    # (a conversation's own rate: its tokens over its first to last token, from the arrival times)
                    rates = []
                    for r in got:
                        if len(r["times"]) < 2:
                            continue
                        dt = r["times"][-1] - r["times"][0]
                        rates.append((len(r["times"]) - 1) / dt if dt > 0 else 0.0)
                    slowest = min(rates) if rates else 0.0
                    mean_per = sum(rates) / len(rates) if rates else 0.0
                    util = u.mean()
                    print(f"  {n}  {label:>7}  {wall:6.1f}s  {tok_n / max(1e-9, wall):6.2f}  "
                          f"{slowest:6.2f} / {mean_per:6.2f}                 {util}", flush=True)
                    eng.quit()
                except Exception as e:  # noqa: BLE001
                    print(f"  {n}  {label:>7}  measurement failed: {e}", flush=True)
                    ok = False
                    try:   # (the engine ends with the row: a failed one holds the GPUs otherwise)
                        eng.proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
        print("\n(the engines' own `ds4 multi:` lines are in each run's stderr, with the VRAM the states took)"
              "\n(a conversation's numbers: its tokens a second from its first token to its last - the slowest of the four and "
              "their mean; a conversation that answers while another's prompt is read counts that time)"
              "\n(GPUs: nvidia-smi's mean utilisation over the answer, one figure a GPU)", flush=True)

    print(("every check passed" if ok else "a check failed"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
