"""tools/maya_nightly.py - the unattended nightly check of this machine's models (cron / Task Scheduler runs it).

    python tools/maya_nightly.py --once                 # tonight's run, now
    python tools/maya_nightly.py --dry-run              # what it would do: no engine start, nothing written
    python tools/maya_nightly.py --once --only Inkling  # one model

For each model in maya-nightly.json (the machine's own list, not in git):
  1. quiet check and locks - the machine has to be free (GPU, CPU load, heat flag, the GPU locks and the model's own
     extra locks); a model whose machine stays busy for "wait_min" minutes is skipped with BUSY: nothing is disturbed;
  2. ./maya.sh --bench --compare <previous release tag> on this checkout (main): exactness first, then speed, with
     the prompts cut from the release's docs (--bench-text-ref: the same prompts every night);
  3. the saved answers (tools/maya_golden.py --check) from the same run's tokens - no second engine start;
  4. one line appended to maya-nightly-results.txt, and maya-nightly-status.txt rewritten: PASS, or what differs.
The night's time budget is printed before the first model (its last run's time, else an estimate) and a model that
no longer fits is skipped with SKIPPED, so a night never runs into the morning.

maya-nightly.json (docs/RELEASE-CHECKS.md has an example):
  {"base": "auto", "rounds": 2, "wait_min": 2, "slow_pct": 5, "budget_min": 20,
   "models": [{"name": "Inkling", "config": "maya-inkling-2-mini.json", "layout": "1x-v100-resident",
               "locks": ["~/hy4/locks/v100"]}]}
"base": "auto" is the newest release tag before this commit.  The config is the one the setup wrote.  "locks": lock
files of the box the bench does not take itself (it takes gpu<N> / radeon in ~/.maya/locks and ~/hy4/locks).
"smoke": true (per model): the short run instead - two short answers against the saved ones, no previous release's
build, no 2K / 8K prompts, no speed verdict.  For a model too large for its box to run the standard in a night's
budget (the full GLM-5.3 on one 32 GB card): it still has to load and answer (#1200).
"answers_vary": true (per model): this layout's greedy answers are not the same on every start (experts read from
the disk while it answers: which ones a GPU holds at a token follows the reads) - answers that differ from the saved
ones are then said in the line, not a DIFF; an engine that cannot answer is still an ERROR.

Exit code: 0 all PASS, 1 something differs or failed, 3 nothing ran (busy or over budget).
The pure parts (the plan, the budget, a model's verdict, the files' lines) need no GPU: tools/test_nightly.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maya_bench as MB  # noqa: E402
import maya_golden as MG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "maya-nightly.json"
RESULTS = "maya-nightly-results.txt"
STATUS = "maya-nightly-status.txt"
RUN_LOCK = "maya-nightly.lock"
DEFAULTS = {"base": "auto", "rounds": 2, "wait_min": 2.0, "slow_pct": 5.0, "budget_min": 20.0, "est_min": 6.0}
BAD = ("DIFF", "SLOW", "ERROR")          # a model's status words that make the night a FAIL


# ------------------------------------------------------------------------------------------------ pure parts
def load_plan(doc: dict, only=()) -> tuple:
    """maya-nightly.json -> (settings, models); a model needs a name and a config."""
    st = {k: doc.get(k, v) for k, v in DEFAULTS.items()}
    models = []
    for m in doc.get("models") or []:
        if not m.get("name") or not m.get("config"):
            raise ValueError("maya-nightly.json: every model needs a name and a config")
        if only and m["name"].lower() not in [x.lower() for x in only]:
            continue
        models.append(m)
    return st, models


def last_seconds(result_lines, machine: str, name: str):
    """A model's wall time the last time it ran here (from the results file), None if it never did."""
    for line in reversed(list(result_lines)):
        f = [x.strip() for x in line.split(" | ")]
        if len(f) >= 9 and f[1] == machine and f[2] == name:
            m = re.fullmatch(r"(\d+) s", f[-2])
            if m and int(m.group(1)) > 0 and not f[-1].startswith(("BUSY", "SKIPPED")):
                return int(m.group(1))
    return None


def budget(models, st: dict, result_lines, machine: str) -> tuple:
    """([(model, its estimated seconds, run it?)], the total of those that run): models in order while they fit."""
    out, used = [], 0.0
    for m in models:
        est = last_seconds(result_lines, machine, m["name"]) or float(m.get("est_min", st["est_min"])) * 60
        fits = used + est <= float(st["budget_min"]) * 60
        out.append((m, est, fits))
        used += est if fits else 0
    return out, used


def budget_lines(rows, total: float, st: dict) -> list:
    out = [f"time budget: {total / 60:.0f} of {float(st['budget_min']):.0f} min for {sum(r[2] for r in rows)} model(s)"]
    out += [f"  {m['name']}: about {est / 60:.1f} min" + ("" if fits else " - does not fit tonight: SKIPPED")
            for m, est, fits in rows]
    return out


COMPARE = re.compile(r"(decode|\d+K) [\d.]+ vs [\d.]+ ([+-][\d.]+)%")


def compare_parts(record: str) -> tuple:
    """A --bench compare line -> (the exactness verdict, {'decode': +1.2, '2K': -0.4 ...})."""
    m = re.search(r"\| exact: ([^|]+)", record)
    return (m.group(1).strip() if m else "?"), {k: float(v) for k, v in COMPARE.findall(record)}


def verdict(exact: str, speeds: dict, golden, slow_pct: float, vary: bool = False) -> tuple:
    """(status, what differs): PASS, or DIFF / SLOW with the reasons.  golden: maya_golden.compare()'s result, or
    None when nothing is saved yet (a note, not a failure).  vary: the model's answers are known to differ between
    starts on this layout ("answers_vary") - saved answers that differ are a note."""
    why, status = [], "PASS"
    if exact == "-":                             # (a smoke run: no other build - the saved answers alone)
        why.append("smoke: it loads and answers" + ("" if golden is None else ", against the saved answers"))
    elif exact.startswith("NO") or exact == "?":
        status = "DIFF"
        why.append("answers differ from the previous release" if exact != "?" else "no compare line")
    elif exact != "yes":
        why.append(f"exact: {exact}")
    if golden is None:
        why.append("no saved answers yet (maya_golden --capture on the release)")
    elif golden["status"] == "DIFF" and vary:
        why.append(f"saved answers: {golden['compared'] - len(golden['diffs'])} of {golden['compared']} equal - "
                   "this layout's answers vary between starts (not a failure)")
    elif golden["status"] != "PASS":
        status = "DIFF"
        why.append("saved answers: " + (golden["diffs"][0] if golden["diffs"] else "nothing compared") +
                   (f" (+{len(golden['diffs']) - 1} more)" if len(golden["diffs"]) > 1 else ""))
        why += golden["notes"][:1]
    slow = [f"{k} {v:+.1f}%" for k, v in speeds.items() if v <= -abs(slow_pct)]
    if slow:
        status = "SLOW" if status == "PASS" else status
        why.append("slower than the previous release: " + ", ".join(slow))
    return status, why


def result_line(date, machine, m, commit, base, exact, speeds, golden_text, secs, status, why) -> str:
    sp = " ".join(f"{k} {v:+.1f}%" for k, v in speeds.items()) or "n/a"
    return " | ".join([date, machine, m["name"], m.get("layout") or "-", f"{commit} vs {base}", f"exact: {exact}",
                       f"speed: {sp}", f"saved answers: {golden_text}", f"{secs:.0f} s",
                       status + (": " + "; ".join(why) if why else "")])


def status_text(date, machine, commit, base, rows, total_s: float) -> str:
    """maya-nightly-status.txt: one headline, then a line a model.  rows: [(name, layout, status, why, secs)]."""
    ran = [r for r in rows if r[2] not in ("BUSY", "SKIPPED")]
    head = "FAIL" if any(r[2] in BAD for r in rows) else "PASS" if ran and len(ran) == len(rows) else \
        "PARTIAL" if ran else "NOT RUN"
    out = [f"{head} | {date} | {machine} | {commit} vs {base} | {len(ran)} of {len(rows)} model(s) ran | "
           f"{total_s / 60:.1f} min"]
    out += [f"  {st:<7} {name} ({layout or '-'}) {secs:.0f} s" + (": " + "; ".join(why) if why else "")
            for name, layout, st, why, secs in rows]
    return "\n".join(out) + "\n"


def exit_code(rows) -> int:
    if any(r[2] in BAD for r in rows):
        return 1
    return 0 if any(r[2] == "PASS" for r in rows) else 3


# ------------------------------------------------------------------------------------------------ the machine
def git(root: Path, *a) -> str:
    return MB._out(["git", "-C", str(root), *a]).strip()


def base_ref(root: Path, want: str) -> str:
    """'auto': the newest release tag before this commit (a commit that is itself a release: the one before it)."""
    if want and want != "auto":
        return want
    tag = git(root, "describe", "--tags", "--abbrev=0", "--match", "v*", "HEAD")
    if tag and git(root, "rev-parse", tag + "^{commit}") == git(root, "rev-parse", "HEAD"):
        tag = git(root, "describe", "--tags", "--abbrev=0", "--match", "v*", "HEAD^") or tag
    if not tag:
        raise RuntimeError("no release tag found (git describe): give \"base\" in maya-nightly.json")
    return tag


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        return str(pid) in MB._out(["tasklist", "/FI", f"PID eq {pid}", "/NH"])
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def take_run_lock(root: Path) -> bool:
    """One nightly at a time in a checkout (a lock left by a run that died is taken over)."""
    p = root / RUN_LOCK
    try:
        old = int(p.read_text().split()[0])
        if old != os.getpid() and pid_alive(old):
            return False
    except (OSError, ValueError, IndexError):
        pass
    p.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M')}\n")
    return True


def extra_locks_held(paths) -> list:
    return [f"lock {p} is held: {(Path(p).expanduser().read_text(errors='replace').strip().splitlines() or [''])[0]}"
            for p in paths if Path(p).expanduser().exists()]


def config_gpus(cfg: dict) -> list:
    """The config's "gpu" (2, [0, 2] or "0,2") as a list, as the server reads it; [] when it names none."""
    g = cfg.get("gpu")
    return [int(x) for x in re.findall(r"\d+", ",".join(map(str, g)) if isinstance(g, (list, tuple)) else str(g))] \
        if g not in (None, "") else []


def extra_locks(cfg: dict, m: dict) -> list:
    """The model's "locks" this run takes itself: lock files the bench does not know (a box's own names, such as
    ~/hy4/locks/v100).  The gpu<N> files in the bench's lock folders are the bench's: it would read ours as
    somebody else's and refuse."""
    own = {(d / n).resolve() for d in MB.lock_dirs()
           for n in MB.lock_names(cfg.get("backend", "cuda"), config_gpus(cfg) or None)}
    return [p for p in m.get("locks") or [] if Path(p).expanduser().resolve() not in own]


def busy_reasons(cfg: dict, m: dict) -> list:
    """Why this model cannot run now ([] = free): the bench's own quiet check for its GPUs, and its extra locks."""
    return MB.machine_quiet(cfg.get("backend", "cuda"), config_gpus(cfg) or None) + \
        extra_locks_held(extra_locks(cfg, m))


def run_bench(root: Path, m: dict, cfg_path: Path, base: str, st: dict, tokens_file: Path, log: Path) -> int:
    """The --bench --compare run (its own quiet wait and GPU locks), its output in `log`; a "smoke" model's short
    run has no other build."""
    cmd = MG.bench_command(root, cfg_path, tokens_file, text_ref=base, compare=base, rounds=int(st["rounds"]),
                           wait_min=float(st["wait_min"]), smoke=bool(m.get("smoke")))
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=str(root), stdout=f, stderr=subprocess.STDOUT).returncode


def run_model(root: Path, m: dict, base: str, st: dict, say, bench=run_bench, busy=busy_reasons) -> dict:
    """One model: quiet check, locks, the compare run, the saved answers.  -> {'status', 'why', 'exact', 'speeds',
    'golden', 'secs'}"""
    t = time.time()
    out = {"status": "ERROR", "why": [], "exact": "?", "speeds": {}, "golden": "-", "secs": 0.0}
    cfg_path = (root / m["config"]) if not Path(m["config"]).expanduser().is_absolute() else \
        Path(m["config"]).expanduser()
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return dict(out, why=[f"config {cfg_path.name}: {e}"])
    why = busy(cfg, m)                           # (a passing load - a build's last step - gets "wait_min" to go away)
    while why and time.time() - t < float(st["wait_min"]) * 60:
        time.sleep(15)
        why = busy(cfg, m)
    if why:
        return dict(out, status="BUSY", exact="-", why=why[:3], secs=time.time() - t)
    note = f"maya nightly pid {os.getpid()} on {socket.gethostname()} {time.strftime('%Y-%m-%d %H:%M')} {m['name']}"
    mine = []
    try:
        for p in extra_locks(cfg, m):
            Path(p).expanduser().write_text(note + "\n")
            mine.append(Path(p).expanduser())
        # (the run's tokens are kept beside its log: maya_golden --from-tokens can check or capture from them later)
        tokens_file = root / f"maya-nightly-{MG.slug(m['name'])}.tokens.json"
        log = root / f"maya-nightly-{MG.slug(m['name'])}.log"
        first_try = []
        for attempt in (1, 2):
            tokens_file.unlink(missing_ok=True)
            code = bench(root, m, cfg_path, base, st, tokens_file, log)
            if code or not tokens_file.exists():
                tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-2:] \
                    if log.exists() else []
                quiet = any("not quiet enough" in x for x in tail)
                return dict(out, status="BUSY" if quiet else "ERROR", secs=time.time() - t, why=first_try + [
                    f"--bench stopped (exit {code}, {log.name}): " + " / ".join(x.strip() for x in tail)])
            tokens = json.loads(tokens_file.read_text(encoding="utf-8"))
            exact, speeds = ("-", {}) if m.get("smoke") else compare_parts((tokens.get("records") or [""])[-1])
            res, _ = MG.check(tokens, cfg, MG.golden_dir(root), m.get("layout", ""))
            status, why = verdict(exact, speeds, res, float(st["slow_pct"]), bool(m.get("answers_vary")))
            # (slow with equal answers: once more before it is called SLOW - a job that started beside the run, or
            #  the model's files gone cold in the page cache, slows one build's first round only)
            if status != "SLOW" or attempt == 2 or not st.get("retry_slow", True):
                break
            first_try = ["first try " + why[-1].split(": ", 1)[-1] + " - run again"]
            say(f"{m['name']}: {first_try[0]}")
        if first_try and status == "PASS":
            first_try = [first_try[0].replace(" - run again", ", not on the second try (a disturbed run)")]
        return {"status": status, "why": first_try + why, "exact": exact, "speeds": speeds,
                "secs": time.time() - t, "golden": MG.verdict_line(res) if res else "none saved"}
    except (OSError, ValueError, KeyError) as e:
        return dict(out, why=[f"{type(e).__name__}: {e}"], secs=time.time() - t)
    finally:
        for p in mine:
            try:
                if p.read_text(errors="replace").strip() == note:
                    p.unlink()
            except OSError:
                pass


def night(root: Path, doc: dict, say, only=(), dry=False, run=run_model) -> int:
    st, models = load_plan(doc, only)
    machine = socket.gethostname().split(".")[0]
    date = time.strftime("%Y-%m-%d")
    base, commit = base_ref(root, st["base"]), MB.commit_of(root)
    results = root / RESULTS
    old = results.read_text(encoding="utf-8", errors="replace").splitlines() if results.exists() else []
    rows_plan, total = budget(models, st, old, machine)
    say(f"maya nightly: {machine}, {commit} against {base}, {len(models)} model(s)")
    for x in budget_lines(rows_plan, total, st):
        say(x)
    if dry:
        for m, _, fits in rows_plan:
            if fits:
                say("  would run: " + " ".join(MG.bench_command(root, Path(m["config"]), Path("<tokens>"), base, base,
                                                              int(st["rounds"]), float(st["wait_min"]),
                                                              bool(m.get("smoke")))[1:]) +
                    (f"  [locks: {', '.join(m['locks'])}]" if m.get("locks") else ""))
        say("dry run: nothing started, nothing written")
        return 0
    if not take_run_lock(root):
        say("another nightly is running in this folder: stopping")
        return 3
    t0, rows = time.time(), []
    try:
        for m, est, fits in rows_plan:
            if not fits or time.time() - t0 + est > float(st["budget_min"]) * 60 * 1.25:
                r = {"status": "SKIPPED", "why": ["over tonight's time budget"], "exact": "-", "speeds": {},
                     "golden": "-", "secs": 0.0}
            else:
                say(f"{m['name']}: starting ({time.strftime('%H:%M:%S')})")
                r = run(root, m, base, st, say)
            say(f"{m['name']}: {r['status']}" + (" - " + "; ".join(r["why"]) if r["why"] else "") +
                f" ({r['secs']:.0f} s)")
            rows.append((m["name"], m.get("layout", ""), r["status"], r["why"], r["secs"]))
            with open(results, "a", encoding="utf-8") as f:
                f.write(result_line(date, machine, m, commit, base, r["exact"], r["speeds"], r["golden"], r["secs"],
                                    r["status"], r["why"]) + "\n")
    finally:
        (root / STATUS).write_text(status_text(date, machine, commit, base, rows, time.time() - t0),
                                   encoding="utf-8")
        (root / RUN_LOCK).unlink(missing_ok=True)
    say((root / STATUS).read_text(encoding="utf-8").rstrip())
    return exit_code(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The unattended nightly check of this machine's models.")
    ap.add_argument("--once", action="store_true", help="run tonight's check now")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the commands; start nothing")
    ap.add_argument("--only", action="append", default=[], metavar="NAME", help="only this model (repeatable)")
    ap.add_argument("--plan", type=Path, default=None, help=f"the machine's list (default ./{CONFIG})")
    ap.add_argument("--base", default="", help="compare with this release tag (default: the list's, else auto)")
    a = ap.parse_args(argv)
    if not (a.once or a.dry_run):
        ap.error("give --once (run now) or --dry-run")
    plan_file = a.plan or ROOT / CONFIG
    try:
        doc = json.loads(plan_file.read_text(encoding="utf-8"))
        if a.base:
            doc["base"] = a.base
        return night(ROOT, doc, lambda s="": print(s, flush=True), a.only, a.dry_run)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"maya_nightly: {plan_file.name}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
